#!/usr/bin/env python3
"""
WORKING TensorFlow Subword (BPE) Semantic Attack Classifier

Uses Byte-Pair Encoding (BPE) with Engineered Feature Flags for robust detection.
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Tokenizers (Hugging Face)
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# Configuration
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
DROPOUT = 0.4
LEARNING_RATE = 0.0001
EPOCHS = 20
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256 # Increased to 500

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
    """Basic Preprocessing (without Feature Injection)."""
    if not isinstance(text, str):
        return ""
    
    # 1. URL Decode
    try:
        text = urllib.parse.unquote(text)
    except Exception:
        pass
        
    # 2. Lowercase and strip whitespace
    text = text.lower().strip()
            
    return text

def balance_dataset(df, classes, target_count=5000, min_count=1000):
    print("\n⚖️  Balancing TRAINING dataset...")
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    
    balanced_dfs = []
    for label in classes:
        class_df = df[df['label'] == label]
        count = len(class_df)
        if count == 0: continue
            
        if count > target_count:
            balanced_dfs.append(class_df.sample(target_count, random_state=42))
        elif count < min_count:
            balanced_dfs.append(class_df.sample(min_count, replace=True, random_state=42))
        else:
            balanced_dfs.append(class_df)
            
    return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

def prepare_data_splits(df, initial_classes, initial_output_dim):
    print("📊 Preparing data splits...")
    
    current_classes = list(initial_classes)
    current_output_dim = initial_output_dim
    
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df.loc[df['label'] == 'path traversal', 'label'] = 'directory_traversal'
    df = df[df['label'].isin(current_classes)]
    
    if 'command_injection' in df['label'].unique() and len(df[df['label'] == 'command_injection']) == 0:
        current_classes = [cls for cls in current_classes if cls != 'command_injection']
        current_output_dim = len(current_classes)
        print(f"⚠️ Removed 'command_injection'. New CLASSES: {current_classes}")

    print("   Constructing text features...")
    df['combined_text'] = (
                df['request_line_url'].fillna('') + " " +
                df['action_message'].fillna('') + " " +
                df['request_body'].fillna('')    )
    
    print("   Preprocessing texts (without feature injection)...")
    df['clean_text'] = df['combined_text'].apply(preprocess_text)
    
    print("\n   Class distribution before splitting:")
    print(df['label'].value_counts())

    print("   Splitting dataset (80/20)...")
    min_samples = df['label'].value_counts()
    eligible = min_samples[min_samples >= 2].index.tolist()
    df = df[df['label'].isin(eligible)]
    
    temp_le = LabelEncoder()
    temp_y = temp_le.fit_transform(df['label'])

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=temp_y, random_state=42)
    
    train_df_balanced = balance_dataset(train_df, current_classes, target_count=10000, min_count=2000)
    print(f"   Balanced Train size: {len(train_df_balanced)}")
    
    return train_df_balanced, test_df, current_classes, current_output_dim

def create_model(vocab_size, max_seq_length, num_classes, learning_rate):
    print("🏗️  Building Subword LSTM model...")
    model = Sequential([
        Embedding(vocab_size, output_dim=EMBEDDING_DIM, input_length=max_seq_length),
        Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(HIDDEN_DIM, return_sequences=False, dropout=DROPOUT)),
        Dense(64, activation='relu'),
        Dropout(DROPOUT),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_tokenizer(texts, vocab_size):
    print("   Training BPE Tokenizer...")
    with open("temp_corpus.txt", "w") as f:
        for text in texts:
            f.write(str(text) + "\n")
            
    tokenizer = ByteLevelBPETokenizer()
    # No custom flags in special tokens after removing feature injection
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] 
    
    tokenizer.train(files=["temp_corpus.txt"], vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)
    
    os.remove("temp_corpus.txt")
    return tokenizer

def encode_texts(tokenizer, texts, max_len):
    encodings = tokenizer.encode_batch(texts.tolist())
    sequences = [e.ids for e in encodings]
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

def evaluate_model(model, tokenizer, label_encoder, X_test, y_test, classes):
    print("📊 Evaluating model on Real-World Test Set...")
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
        target_names=classes,
        zero_division=0, 
        labels=list(range(len(classes)))
    )
    print(report)

def train_model(model, train_df, test_df, vocab_size, max_seq_length, epochs, batch_size, output_dir, classes):
    print("🚀 Starting training pipeline...")

    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    
    y_train = to_categorical(label_encoder.transform(train_df['label']), num_classes=len(classes))
    y_test = to_categorical(label_encoder.transform(test_df['label']), num_classes=len(classes))

    tokenizer = train_tokenizer(train_df['clean_text'], vocab_size)
    
    X_train = encode_texts(tokenizer, train_df['clean_text'], max_seq_length)
    X_test = encode_texts(tokenizer, test_df['clean_text'], max_seq_length)

    y_train_int = np.argmax(y_train, axis=1)
    weights = compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
    class_weight_dict = dict(zip(np.unique(y_train_int), weights))
    print(f"⚖️  Class weights: {class_weight_dict}")

    os.makedirs(output_dir, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(os.path.join(output_dir, 'best_model.keras'), monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test), 
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    model.save(os.path.join(output_dir, 'final_model.keras'))
    tokenizer.save_model(output_dir)
    
    import pickle
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    return model, tokenizer, label_encoder, X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/Modsec-WP.csv')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--output-dir', type=str, default='results')
    
    args = parser.parse_args()
    setup_tensorflow()
    
    if not os.path.exists(args.input): return 1
    df = pd.read_csv(args.input, low_memory=False)
    
    global CLASSES
    global OUTPUT_DIM
    
    train_df, test_df, CLASSES, OUTPUT_DIM = prepare_data_splits(df, CLASSES, OUTPUT_DIM)
    model = create_model(VOCAB_SIZE, MAX_SEQ_LENGTH, len(CLASSES), LEARNING_RATE)
    
    model, tokenizer, label_encoder, X_test, y_test = train_model(model, train_df, test_df, VOCAB_SIZE, MAX_SEQ_LENGTH, args.epochs, args.batch_size, args.output_dir, CLASSES)
    
    evaluate_model(model, tokenizer, label_encoder, X_test, y_test, CLASSES)
    
    return 0

if __name__ == "__main__":
    exit(main())
