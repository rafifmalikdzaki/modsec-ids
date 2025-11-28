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
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
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
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.4
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 500 # Increased to 500

# Classes
CLASSES = ['normal', 'sqli', 'bruteforce', 'lfi', 'xss', 'rce', 'directory_traversal', 'command_injection', 'rfi']
OUTPUT_DIM = len(CLASSES)

# Special Feature Flags
FEATURE_FLAGS = [
    "[FLAG_RFI]", "[FLAG_LFI]", "[FLAG_XSS]", "[FLAG_SQLI]", "[FLAG_RCE]", "[FLAG_TRAVERSAL]"
]

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
    """Preprocessing with Feature Injection."""
    if not isinstance(text, str):
        return ""
    
    # 1. URL Decode
    try:
        text = urllib.parse.unquote(text)
    except Exception:
        pass
        
    # 2. Lowercase
    text = text.lower().strip()
    
    # 3. Feature Injection (Heuristic hints)
    flags = []
    
    # RFI: http/https in parameters
    if "http://" in text or "https://" in text or "ftp://" in text:
        flags.append("[FLAG_RFI]")
        
    # Traversal / LFI
    if "../" in text or "..\\\\" in text or "/etc/passwd" in text or "win.ini" in text:
        flags.append("[FLAG_TRAVERSAL]")
        
    # XSS
    if "<script" in text or "javascript:" in text or "onerror=" in text or "onload=" in text:
        flags.append("[FLAG_XSS]")
        
    # SQLi
    if "union select" in text or " or 1=1" in text or "'--" in text or "information_schema" in text:
        flags.append("[FLAG_SQLI]")
        
    # RCE
    if "; cat" in text or "| ls" in text or "$(whoami)" in text or "; system" in text:
        flags.append("[FLAG_RCE]")

    # Append flags to text
    if flags:
        text = " ".join(flags) + " " + text
        
    return text

def balance_dataset(df, target_count=5000, min_count=1000):
    print("\n⚖️  Balancing TRAINING dataset...")
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    
    balanced_dfs = []
    for label in CLASSES:
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

def prepare_data_splits(df):
    print("📊 Preparing data splits...")
    global CLASSES
    global OUTPUT_DIM
    
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df.loc[df['label'] == 'path traversal', 'label'] = 'directory_traversal'
    df = df[df['label'].isin(CLASSES)]
    
    if 'command_injection' in df['label'].unique() and len(df[df['label'] == 'command_injection']) == 0:
        CLASSES = [cls for cls in CLASSES if cls != 'command_injection']
        OUTPUT_DIM = len(CLASSES)
        print(f"⚠️ Removed 'command_injection'. New CLASSES: {CLASSES}")

    print("   Constructing text features...")
    df['combined_text'] = (
        df['request_line_method'].fillna('') + " " +
        df['request_line_url'].fillna('') + " " +
        df['action_message'].fillna('') + " " + 
        df['request_body'].fillna('')
    )
    
    print("   Preprocessing texts (with feature injection)...")
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
    
    train_df_balanced = balance_dataset(train_df, target_count=10000, min_count=2000)
    print(f"   Balanced Train size: {len(train_df_balanced)}")
    
    return train_df_balanced, test_df

def create_model(vocab_size, max_seq_length, num_classes):
    print("🏗️  Building Subword LSTM model...")
    model = Sequential([
        Embedding(vocab_size, output_dim=EMBEDDING_DIM, input_length=max_seq_length),
        Bidirectional(LSTM(HIDDEN_DIM, return_sequences=False, dropout=DROPOUT)),
        Dense(64, activation='relu'),
        Dropout(DROPOUT),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_tokenizer(texts, vocab_size):
    print("   Training BPE Tokenizer...")
    with open("temp_corpus.txt", "w") as f:
        for text in texts:
            f.write(str(text) + "\n")
            
    tokenizer = ByteLevelBPETokenizer()
    # Add custom flags to special tokens so they are learned as single tokens
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + FEATURE_FLAGS
    
    tokenizer.train(files=["temp_corpus.txt"], vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)
    
    os.remove("temp_corpus.txt")
    return tokenizer

def encode_texts(tokenizer, texts, max_len):
    encodings = tokenizer.encode_batch(texts.tolist())
    sequences = [e.ids for e in encodings]
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

def evaluate_model(model, tokenizer, label_encoder, X_test, y_test):
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
        target_names=label_encoder.classes_,
        zero_division=0, 
        labels=list(range(len(label_encoder.classes_)))
    )
    print(report)

def train_model(model, train_df, test_df, vocab_size, max_seq_length, epochs, batch_size, output_dir):
    print("🚀 Starting training pipeline...")

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)
    
    y_train = to_categorical(label_encoder.transform(train_df['label']), num_classes=len(CLASSES))
    y_test = to_categorical(label_encoder.transform(test_df['label']), num_classes=len(CLASSES))

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
    
    train_df, test_df = prepare_data_splits(df)
    model = create_model(VOCAB_SIZE, MAX_SEQ_LENGTH, len(CLASSES))
    
    model, tokenizer, label_encoder, X_test, y_test = train_model(model, train_df, test_df, VOCAB_SIZE, MAX_SEQ_LENGTH, args.epochs, args.batch_size, args.output_dir)
    
    evaluate_model(model, tokenizer, label_encoder, X_test, y_test)
    
    return 0

if __name__ == "__main__":
    exit(main())
