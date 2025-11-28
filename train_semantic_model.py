#!/usr/bin/env python3
"""
Training script for LSTM Semantic Attack Classifier.

This script trains a pure LSTM model that learns semantic information
from HTTP headers, payloads, responses, and messages.

Usage:
    python train_semantic_model.py [--input data/dataset.csv] [--epochs 10] [--batch-size 32]
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import our semantic model
from enhanced_security_model import LSTMMultiClassSemanticAttackClassifier, ATTACK_CLASSES, ATTACK_COLORS

def prepare_training_data(csv_file):
    """Prepare training data from CSV file for semantic analysis."""
    print(f"Loading training data from {csv_file}...")

    if not os.path.exists(csv_file):
        print(f"Error: Dataset file not found: {csv_file}")
        print("Please provide a CSV file with HTTP log data")
        return [], []

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from dataset")

        # Prepare texts and labels
        texts = []
        labels = []

        print("Preparing training data for semantic analysis...")
        for idx, row in df.iterrows():
            # Combine all text fields for comprehensive semantic analysis
            text_parts = []

            # Request data
            if 'request_line_method' in row and pd.notna(row['request_line_method']):
                text_parts.append(str(row['request_line_method']))
            if 'request_line_url' in row and pd.notna(row['request_line_url']):
                text_parts.append(str(row['request_line_url']))
            if 'request_useragent' in row and pd.notna(row['request_useragent']):
                text_parts.append(str(row['request_useragent']))
            if 'request_body' in row and pd.notna(row['request_body']) and str(row['request_body']).strip():
                text_parts.append(str(row['request_body']))

            # Response data
            if 'response_body' in row and pd.notna(row['response_body']) and str(row['response_body']).strip():
                text_parts.append(str(row['response_body']))

            # Security messages
            if 'action_message' in row and pd.notna(row['action_message']) and str(row['action_message']).strip():
                text_parts.append(str(row['action_message']))
            if 'message_msg' in row and pd.notna(row['message_msg']) and str(row['message_msg']).strip():
                text_parts.append(str(row['message_msg']))
            if 'message_description' in row and pd.notna(row['message_description']) and str(row['message_description']).strip():
                text_parts.append(str(row['message_description']))

            # Additional fields if available
            if 'request_host' in row and pd.notna(row['request_host']):
                text_parts.append(str(row['request_host']))
            if 'full_message_line' in row and pd.notna(row['full_message_line']) and str(row['full_message_line']).strip():
                text_parts.append(str(row['full_message_line']))

            # Combine all text fields
            combined_text = ' '.join(filter(None, text_parts))

            if combined_text.strip():
                texts.append(combined_text.strip())

                # Extract label - handle different label formats for multi-class
                label_str = str(row.get('label', 'normal')).strip().lower()

                # Map label to class index
                if label_str == 'normal':
                    labels.append(0)  # Normal
                elif label_str == 'sqli':
                    labels.append(1)  # SQL Injection
                elif label_str == 'bruteforce':
                    labels.append(2)  # Brute Force
                elif label_str == 'lfi':
                    labels.append(3)  # Local File Inclusion
                elif label_str == 'path traversal':
                    labels.append(4)  # Path Traversal
                elif label_str == 'rfi':
                    labels.append(5)  # Remote File Inclusion
                elif label_str == 'rce':
                    labels.append(6)  # Remote Code Execution
                elif label_str == 'xss':
                    labels.append(7)  # Cross-Site Scripting
                else:
                    labels.append(0)  # Unknown/Normal fallback

            if idx % 5000 == 0 and idx > 0:
                print(f"Processed {idx} records...")

        if len(texts) == 0:
            print("Error: No valid text data found in dataset")
            return [], []

        print(f"\nTraining data prepared:")
        print(f"  Total samples: {len(texts)}")
        print(f"  Attack samples: {sum(labels)}")
        print(f"  Normal samples: {len(labels) - sum(labels)}")
        print(f"  Attack percentage: {100 * sum(labels) / len(labels):.1f}%")
        print(f"  Average text length: {np.mean([len(text.split()) for text in texts]):.1f} words")

        # Show some example texts
        print(f"\nSample texts for training:")
        for i in range(min(3, len(texts))):
            label_text = "ATTACK" if labels[i] == 1 else "NORMAL"
            preview = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
            print(f"  [{i+1}] {label_text}: {preview}")

        return texts, labels

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], []

def train_multiclass_semantic_model(texts, labels, epochs=10, batch_size=32, learning_rate=0.001):
    """Train the LSTM multi-class semantic model."""
    print(f"\nTraining LSTM Multi-Class Semantic Attack Classifier...")
    print(f"Configuration: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Initialize and train model
    model = LSTMMultiClassSemanticAttackClassifier()

    # Train the model
    model.train_model(texts, labels, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    return model

def test_multiclass_model(model, test_texts, test_labels):
    """Test the multi-class trained model on test data."""
    print(f"\nTesting multi-class model on {len(test_texts)} samples...")

    correct = 0
    total = len(test_labels)
    class_correct = {i: 0 for i in range(8)}  # 8 classes total

    for i, text in enumerate(test_texts):
        prediction = model.predict(text)
        predicted_label = prediction.get('predicted_label', -1)
        actual_label = test_labels[i]

        if predicted_label == actual_label:
            correct += 1
            class_correct[actual_label] = class_correct.get(actual_label, 0) + 1

        if i % 1000 == 0:
            print(f"Tested {i}/{total} samples...")

    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}% ({correct}/{total} correct)")

    # Per-class accuracy
    print(f"\nPer-Class Test Accuracy:")
    for i in range(8):
        if i < len(model.attack_classes):
            class_name = model.attack_classes[i]
            class_accuracy = 100 * class_correct.get(i, 0) / max(1, sum(1 for label in test_labels if label == i))
            print(f"  {class_name:12s}: {class_accuracy:5.1f}%")

    # Create confusion matrix summary
    print(f"\nDetailed Prediction Summary:")
    for i, text in enumerate(test_texts[:5]):  # Show first 5 predictions
        prediction = model.predict(text)
        predicted_class = prediction.get('predicted_class', 'unknown')
        confidence = prediction.get('confidence', 0.0)
        actual_class = model.attack_classes.get(test_labels[i], 'unknown')
        print(f"  Sample {i+1}: Actual={actual_class:12s}, Predicted={predicted_class:12s}, Confidence={confidence:.3f}")

    return overall_accuracy

def main():
    parser = argparse.ArgumentParser(description="Train LSTM Semantic Attack Classifier")
    parser.add_argument('--input', type=str, default='data/raw/Modsec-WP.csv',
                       help='Input CSV file with HTTP log data')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test set split ratio')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing model, do not train')

    args = parser.parse_args()

    print("=" * 60)
    print("LSTM SEMANTIC ATTACK CLASSIFIER")
    print("=" * 60)

    if args.test_only:
        # Test existing model
        print("Testing existing LSTM semantic model...")

        # Load test data
        texts, labels = prepare_training_data(args.input)
        if not texts:
            print("No training data available")
            return

        # Test on full dataset (or we could split)
        test_texts = texts
        test_labels = labels

        model = LSTMSemanticAttackClassifier()
        if model.is_loaded:
            test_model(model, test_texts, test_labels)
        else:
            print("No trained model found. Please train first using --train mode.")
    else:
        # Train new model
        print("Training new LSTM semantic model...")

        # Load and prepare training data
        texts, labels = prepare_training_data(args.input)
        if not texts:
            print("No training data available")
            return

        # Split data for training and testing
        split_idx = int(len(texts) * (1 - args.test_split))
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        test_texts = texts[split_idx:]
        test_labels = labels[split_idx:]

        print(f"\nData split:")
        print(f"  Training: {len(train_texts)} samples")
        print(f"  Testing: {len(test_texts)} samples")

        # Train multi-class model
        model = train_multiclass_semantic_model(train_texts, train_labels,
                                               epochs=args.epochs,
                                               batch_size=args.batch_size,
                                               learning_rate=args.learning_rate)

        # Test model
        if model.is_loaded:
            test_multiclass_model(model, test_texts, test_labels)
        else:
            print("Model training failed")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Provide usage instructions
    print("\nTo use the trained multi-class semantic model:")
    print("1. Run the dashboard with multi-class model:")
    print("   python idsdashboard.py --multiclass")
    print("2. Or run with both multi-class and semantic:")
    print("   python idsdashboard.py --multiclass --semantic")
    print("\nModel files saved to:")
    print(f"  - models/lstm_multiclass_semantic_classifier.pth")
    print(f"  - data/processed/lstm_multiclass_semantic_preprocessor.pkl")

if __name__ == "__main__":
    main()