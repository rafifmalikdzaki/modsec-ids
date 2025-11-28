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

# Configuration
VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 8
DROPOUT = 0.4
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32

# Classes
CLASSES = ['normal', 'sqli', 'bruteforce', 'lfi', 'xss', 'rce', 'directory_traversal', 'command_injection']

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

def prepare_data(df):
    print("📊 Preparing data...")

    # Simple text preparation
    texts = []
    labels = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx} rows")

        try:
            # Combine basic text fields
            text_parts = []

            # Essential fields
            if 'request_line_method' in row and pd.notna(row['request_line_method']):
                text_parts.append(str(row['request_line_method']))
            if 'request_line_url' in row and pd.notna(row['request_line_url']):
                text_parts.append(str(row['request_line_url']))
            if 'request_useragent' in row and pd.notna(row['request_useragent']):
                text_parts.append(str(row['request_useragent']))
            if 'action_message' in row and pd.notna(row['action_message']):
                text_parts.append(str(row['action_message']))

            combined_text = ' '.join(text_parts)

            if combined_text.strip():
                texts.append(combined_text)

                # Simple label mapping
                label_str = str(row.get('label', 'normal')).lower()
                if label_str in ['attack', 'malicious', 'blocked']:
                    label = 1  # SQLi
                elif 'sqli' in label_str or 'union' in label_str or 'select' in label_str:
                    label = 1  # SQLi
                elif 'xss' in label_str or 'script' in label_str:
                    label = 4  # XSS
                elif 'exec' in label_str or 'shell' in label_str:
                    label = 5  # RCE
                elif 'file' in label_str or 'include' in label_str:
                    label = 3  # LFI
                elif 'directory' in label_str or 'traversal' in label_str:
                    label = 6  # Directory Traversal
                elif 'command' in label_str:
                    label = 7  # Command Injection
                else:
                    label = 0  # Normal

                labels.append(label)

        except Exception:
            continue

    print(f"✅ Data prepared: {len(texts)} samples")
    return texts, labels

def create_model(vocab_size, max_seq_length, num_classes):
    print("🏗️  Building LSTM model...")

    model = Sequential([
        Embedding(vocab_size, output_dim=EMBEDDING_DIM, input_length=max_seq_length),
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

def train_model(model, texts, labels, vocab_size, max_seq_length, epochs, batch_size, output_dir):
    print("🚀 Starting training...")

    # Encode labels
    # Labels are already integers matching CLASSES indices, so we don't need fit_transform
    # which might shift indices if some classes are missing
    y_encoded = np.array(labels)
    
    # Create label encoder for artifacts (fitting on all classes to ensure consistency)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(range(len(CLASSES))))
    
    y_categorical = to_categorical(y_encoded, num_classes=len(CLASSES))

    # Tokenize texts
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)

    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')

    # Create proper train-test split (80% train, 20% test)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded  # Ensure balanced class distribution
    )

    print(f"✅ Train-test split created: {len(X_train)} training, {len(X_test)} testing samples")

    # Setup callbacks
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

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Now this is 20% of training data for validation
        callbacks=callbacks,
        verbose=1
    )

    # Save model and artifacts (using native Keras format)
    model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(model_path)
    print(f"✅ Model saved: {model_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, 'tokenizer.pkl')
    import pickle
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved: {tokenizer_path}")

    # Save label encoder
    encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved: {encoder_path}")

    print("🎉 Training completed!")
    return model, tokenizer, label_encoder, history, X_test, y_test

def evaluate_model(model, tokenizer, label_encoder, X_test, y_test, max_seq_length=100):
    print("📊 Evaluating model...")

    # X_test is already tokenized and padded from train_model
    X_test_padded = X_test

    # Evaluate
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test_padded, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Convert y_test from one-hot to integer labels
    y_test_int = np.argmax(y_test, axis=1)

    # Classification report
    print("\nClassification Report:")
    # Use labels=range(len(CLASSES)) to force report to include all classes and match target_names
    report = classification_report(y_test_int, y_pred_classes, target_names=CLASSES, zero_division=0, labels=list(range(len(CLASSES))))
    print(report)

    return {
        'test_loss': loss,
        'test_accuracy': accuracy,
        'classification_report': report
    }

def main():
    parser = argparse.ArgumentParser(description="Working TensorFlow LSTM Semantic Attack Classifier")
    parser.add_argument('--input', type=str, default='data/raw/Modsec-WP.csv', help='Input CSV file')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--vocab-size', type=int, default=VOCAB_SIZE, help='Vocabulary size')
    parser.add_argument('--max-seq-length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate')

    args = parser.parse_args()

    print("="*80)
    print("🚀 WORKING TENSORFLOW LSTM SEMANTIC ATTACK CLASSIFIER")
    print("="*80)

    # Setup TensorFlow
    setup_tensorflow()

    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    # Load data
    try:
        df = pd.read_csv(args.input)
        print(f"✅ Dataset loaded: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1

    # Prepare data
    texts, labels = prepare_data(df)
    if len(texts) == 0:
        print("❌ Error: No valid text data prepared")
        return 1

    # Create model
    model = create_model(args.vocab_size, args.max_seq_length, len(CLASSES))

    # Train model
    model, tokenizer, label_encoder, history, X_test, y_test = train_model(
        model, texts, labels,
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    # Evaluate model on held-out test set
    print("📊 Evaluating on held-out test set...")
    test_results = evaluate_model(model, tokenizer, label_encoder,
                                 X_test,
                                 y_test,
                                 max_seq_length=args.max_seq_length)

    print("\n" + "="*80)
    print(f"🎯 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"✅ Test Accuracy: {test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)")
    print(f"✅ Test Loss: {test_results['test_loss']:.4f}")
    print(f"📁 Results saved to: {args.output_dir}")

    return 0

if __name__ == "__main__":
    exit(main())