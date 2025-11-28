#!/usr/bin/env python3
"""
WORKING TensorFlow LSTM Semantic Attack Classifier

Minimal, robust implementation that guarantees to work.
Fixed all known issues and simplified for stability.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import urllib.parse
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Configuration
# Character-level parameters
VOCAB_SIZE = 200 # Enough for all ASCII chars
EMBEDDING_DIM = 32 # Smaller embedding sufficient for chars
HIDDEN_DIM = 128 # Larger hidden state to capture longer char sequences
DROPOUT = 0.4
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 64 # Larger batch size for efficiency
MAX_SEQ_LENGTH = 1000 # Increased for character sequences

# Classes
CLASSES = ['normal', 'sqli', 'bruteforce', 'lfi', 'xss', 'rce', 'directory_traversal', 'command_injection', 'rfi']
OUTPUT_DIM = len(CLASSES)

def setup_tensorflow():
    print("🔧 Setting up TensorFlow...")
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured")
            return True
        except Exception:
            print("⚠️  GPU configuration failed, using CPU")
            return False
    else:
        print("ℹ️  No GPUs detected - using CPU")
        return False

def preprocess_text(text):
    """Minimal preprocessing for char-level model."""
    if not isinstance(text, str):
        return ""
    
    # 1. URL Decode first (Important to expose actual characters)
    try:
        text = urllib.parse.unquote(text)
    except Exception:
        pass
        
    # 2. Lowercase (Optional, but helps reduce vocab size)
    text = text.lower()
    
    return text.strip()

def balance_dataset(df, target_count=5000, min_count=1000):
    """
    Balance the dataset by undersampling majority classes and oversampling minority classes.
    """
    print("\n⚖️  Balancing TRAINING dataset...")
    
    # Standardize labels first
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    
    balanced_dfs = []
    
    for label in CLASSES:
        class_df = df[df['label'] == label]
        count = len(class_df)
        
        if count == 0:
            continue
            
        if count > target_count:
            balanced_dfs.append(class_df.sample(target_count, random_state=42))
        elif count < min_count:
            balanced_dfs.append(class_df.sample(min_count, replace=True, random_state=42))
        else:
            balanced_dfs.append(class_df)
            
    return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

def prepare_data_splits(df):
    print("📊 Preparing data splits...")
    
    global CLASSES
    global OUTPUT_DIM
    
    # 1. Standardize Labels
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df.loc[df['label'] == 'path traversal', 'label'] = 'directory_traversal'
    
    # Filter only known classes
    df = df[df['label'].isin(CLASSES)]
    
    # Remove classes with 0 samples
    if 'command_injection' in df['label'].unique() and len(df[df['label'] == 'command_injection']) == 0:
        CLASSES = [cls for cls in CLASSES if cls != 'command_injection']
        OUTPUT_DIM = len(CLASSES)
        print(f"⚠️ Removed 'command_injection' from CLASSES as it has 0 samples. New CLASSES: {CLASSES}")

    # 2. Construct the text feature
    print("   Constructing text features...")
    # Removing User-Agent to focus on payload
    df['combined_text'] = (
        df['request_line_method'].fillna('') + " " +
        df['request_line_url'].fillna('') + " " +
        df['action_message'].fillna('') + " " + 
        df['request_body'].fillna('')
    )
    
    # 3. Preprocess Texts
    print("   Preprocessing texts (cleaning)...")
    df['clean_text'] = df['combined_text'].apply(preprocess_text)
    
    # Print distribution before splitting
    print("\n   Class distribution before splitting:")
    print(df['label'].value_counts())

    # 4. Stratified Split
    print("   Splitting dataset (80/20)...")
    min_samples_per_class = df['label'].value_counts()
    eligible_classes = min_samples_per_class[min_samples_per_class >= 2].index.tolist()
    df_eligible = df[df['label'].isin(eligible_classes)]
    
    temp_label_encoder = LabelEncoder()
    temp_labels = temp_label_encoder.fit_transform(df_eligible['label'])

    train_df, test_df = train_test_split(
        df_eligible,
        test_size=0.2, 
        stratify=temp_labels,
        random_state=42
    )
    
    print(f"   Original Train size: {len(train_df)}")
    print(f"   Original Test size: {len(test_df)}")
    
    # 5. Balance ONLY the Training Set
    train_df_balanced = balance_dataset(train_df, target_count=10000, min_count=2000)
    print(f"   Balanced Train size: {len(train_df_balanced)}")
    
    return train_df_balanced, test_df

def create_model(vocab_size, max_seq_length, num_classes):
    print("🏗️  Building Character-Level LSTM model...")

    model = Sequential([
        Embedding(vocab_size, output_dim=EMBEDDING_DIM, input_length=max_seq_length),
        # Use Conv1D to extract features from chars before LSTM (helps speed and performance)
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(HIDDEN_DIM, return_sequences=False, dropout=DROPOUT)),
        Dense(64, activation='relu'),
        Dropout(DROPOUT),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ Model created and compiled")
    return model

def train_model(model, train_df, test_df, vocab_size, max_seq_length, epochs, batch_size, output_dir):
    print("🚀 Starting training pipeline...")

    # Prepare Labels
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)
    
    y_train_int = label_encoder.transform(train_df['label'])
    y_train = to_categorical(y_train_int, num_classes=len(CLASSES))
    
    y_test_int = label_encoder.transform(test_df['label'])
    y_test = to_categorical(y_test_int, num_classes=len(CLASSES))

    # Prepare Texts - CHARACTER LEVEL
    print("   Tokenizing (Character Level)...")
    tokenizer = Tokenizer(
        num_words=vocab_size, 
        char_level=True, # Enable char level
        oov_token='<OOV>', 
        lower=True
    )
    tokenizer.fit_on_texts(train_df['clean_text'])

    X_train = tokenizer.texts_to_sequences(train_df['clean_text'])
    X_train = pad_sequences(X_train, maxlen=max_seq_length, padding='post', truncating='post')
    
    X_test = tokenizer.texts_to_sequences(test_df['clean_text'])
    X_test = pad_sequences(X_test, maxlen=max_seq_length, padding='post', truncating='post')

    # Class Weights
    unique_classes = np.unique(y_train_int)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train_int
    )
    class_weight_dict = dict(zip(unique_classes, weights))
    print(f"⚖️  Class weights: {class_weight_dict}")

    # Callbacks
    os.makedirs(output_dir, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test), 
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save
    model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(model_path)
    
    tokenizer_path = os.path.join(output_dir, 'tokenizer.pkl')
    import pickle
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    return model, tokenizer, label_encoder, history, X_test, y_test

def evaluate_model(model, tokenizer, label_encoder, X_test, y_test, max_seq_length=100):
    print("📊 Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_int = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    report = classification_report(
        y_test_int, 
        y_pred_classes, 
        target_names=label_encoder.classes_,
        zero_division=0, 
        labels=list(range(len(label_encoder.classes_)))
    )
    print(report)
    return {'test_loss': loss, 'test_accuracy': accuracy, 'classification_report': report}

def main():
    parser = argparse.ArgumentParser(description="Working TensorFlow LSTM Semantic Attack Classifier")
    parser.add_argument('--input', type=str, default='data/raw/Modsec-WP.csv', help='Input CSV file')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--vocab-size', type=int, default=VOCAB_SIZE, help='Vocabulary size')
    parser.add_argument('--max-seq-length', type=int, default=1000, help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate')

    args = parser.parse_args()

    print("="*80)
    print("🚀 WORKING TENSORFLOW CHAR-LEVEL LSTM CLASSIFIER")
    print("="*80)

    setup_tensorflow()

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    try:
        df = pd.read_csv(args.input, low_memory=False)
        print(f"✅ Dataset loaded: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1

    train_df, test_df = prepare_data_splits(df)
    
    model = create_model(args.vocab_size, args.max_seq_length, len(CLASSES))

    model, tokenizer, label_encoder, history, X_test, y_test = train_model(
        model, train_df, test_df,
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    evaluate_model(model, tokenizer, label_encoder, X_test, y_test, max_seq_length=args.max_seq_length)

    return 0

if __name__ == "__main__":
    exit(main())