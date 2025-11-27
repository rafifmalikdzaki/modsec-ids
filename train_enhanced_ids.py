#!/usr/bin/env python3
"""
Training script for Enhanced LSTM-based Intrusion Detection System
Using ModSecurity WordPress dataset
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Enhanced training function that works with multi-class data
def train_multiclass_lstm(processed_path, epochs=20, batch_size=64, use_multiclass=True):
    """Train LSTM model on preprocessed data with multi-class support."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    print("Starting multi-class LSTM training...")

    # Load data and preprocessor
    with np.load(processed_path, allow_pickle=True) as data:
        features = data['features']
        labels = data['labels']

    # Load the preprocessor separately
    import joblib
    preprocessor_path = processed_path.replace('.npz', '_preprocessor.pkl')
    preprocessor_data = joblib.load(preprocessor_path)

    # Handle both cases: preprocessor object or dictionary
    if hasattr(preprocessor_data, 'label_encoder'):
        label_encoder = preprocessor_data.label_encoder
    elif isinstance(preprocessor_data, dict) and 'label_encoder' in preprocessor_data:
        label_encoder = preprocessor_data['label_encoder']
    else:
        # Try to load label encoder separately
        try:
            label_encoder = joblib.load('models/label_encoder.pkl')
        except:
            raise ValueError("Could not load label encoder from preprocessor")
    num_classes = len(label_encoder.classes_)

    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {list(label_encoder.classes_)}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Simple dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = torch.FloatTensor(features)
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    # Create dataset
    dataset = SimpleDataset(features, labels)

    # Split dataset: 70% train, 15% validation, 15% test (stratified to preserve class distribution)
    from sklearn.model_selection import train_test_split
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.15,
        random_state=42,
        stratify=labels
    )

    # Second split: separate train and validation from remaining data
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.15/0.85,  # 15% of original dataset = 17.6% of train_val set
        random_state=42,
        stratify=[labels[i] for i in train_val_indices]
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Save test set for later use with dashboard
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    test_data_path = processed_path.replace('.npz', '_test.npz')
    np.savez(
        test_data_path,
        features=test_features,
        labels=test_labels,
        test_indices=test_indices,
        label_encoder=label_encoder.classes_
    )
    print(f"💾 Test set saved to: {test_data_path}")
    print(f"📊 Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Enhanced LSTM model for multi-class classification
    class MulticlassLSTM(nn.Module):
        def __init__(self, input_size=15, hidden_size=64, output_size=8):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            # x shape: (batch, seq_len, features) - need to add seq dimension
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension

            lstm_out, _ = self.lstm(x)
            # Use the last output
            last_out = lstm_out[:, -1, :]
            last_out = self.dropout(last_out)
            output = self.fc(last_out)
            return output

    # Initialize model with correct number of output classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = MulticlassLSTM(input_size=features.shape[1], hidden_size=64, output_size=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('models').mkdir(exist_ok=True)

            # Save model with additional metadata
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'input_size': features.shape[1],
                'hidden_size': 64,
                'output_size': num_classes,
                'class_names': label_encoder.classes_,
                'label_encoder': label_encoder
            }, 'models/multiclass_lstm_classifier.pth')

            # Also save the label encoder separately for easy loading
            joblib.dump(label_encoder, 'models/label_encoder.pkl')

            print(f'  ✅ New best model saved! Val Acc: {val_acc:.2f}%')

            # Print detailed classification report for best model
            print(f'\n📊 Classification Report (Epoch {epoch+1}):')
            class_report = classification_report(
                all_targets,
                all_predictions,
                target_names=label_encoder.classes_,
                zero_division=0
            )
            print(class_report)

    print("\n✅ Multi-class LSTM training completed!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: models/multiclass_lstm_classifier.pth")
    print(f"💾 Label encoder saved to: models/label_encoder.pkl")

    return best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Enhanced LSTM-based IDS")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess raw dataset first")
    parser.add_argument("--train", action="store_true", help="Train LSTM model")
    parser.add_argument("--input", type=str, default="data/raw/Modsec-WP.csv",
                       help="Input ModSecurity dataset CSV file")
    parser.add_argument("--processed", type=str, default="data/processed/modsec_processed.npz",
                       help="Output path for processed data")
    parser.add_argument("--sample", type=int, default=None,
                       help="Sample size for training (use None for full dataset)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Training batch size")

    args = parser.parse_args()

    print("Enhanced LSTM-based IDS Training Pipeline")
    print("=" * 50)

    # Step 1: Preprocess dataset if requested
    if args.preprocess:
        print("Step 1: Preprocessing dataset...")
        try:
            from data_preprocessor import clean_and_preprocess_dataset

            # Create processed data directory
            Path(args.processed).parent.mkdir(parents=True, exist_ok=True)

            # Preprocess the dataset
            clean_and_preprocess_dataset(args.input, args.processed, args.sample)
            print(f"✅ Dataset preprocessed successfully: {args.processed}")
        except Exception as e:
            print(f"❌ Error preprocessing dataset: {e}")
            return 1

    # Step 2: Train LSTM model if requested
    if args.train:
        print("\nStep 2: Training LSTM model...")

        # Check if processed data exists
        if not os.path.exists(args.processed):
            print(f"❌ Processed dataset not found at {args.processed}")
            print("Run with --preprocess first or specify correct --processed path")
            return 1

        try:
            # Import required modules
            import torch
            import torch.utils.data

            # Temporarily modify the LSTM script parameters
            import lstm_security_model
            lstm_security_model.EPOCHS = args.epochs
            lstm_security_model.BATCH_SIZE = args.batch_size
            lstm_security_model.ModSecurityDataset.csv_file = args.processed

            # Update the ModSecurityDataset to work with processed .npz file
            class ProcessedModSecurityDataset(torch.utils.data.Dataset):
                def __init__(self, processed_path, sequence_length=50):
                    data = np.load(processed_path)
                    self.features = data['features']
                    self.labels = data['labels']
                    self.sequence_length = sequence_length

                    # Create sequences for LSTM
                    self.sequences, self.sequence_labels = self._create_sequences()

                def _create_sequences(self):
                    sequences = []
                    labels = []

                    # Pad with zeros at beginning if needed
                    padded_features = np.zeros((len(self.features) + self.sequence_length - 1, self.features.shape[1]))
                    padded_features[self.sequence_length - 1:] = self.features

                    # Create sliding window sequences
                    for i in range(len(self.features)):
                        seq = padded_features[i:i + self.sequence_length]
                        sequences.append(seq)
                        labels.append(self.labels[i])

                    return np.array(sequences), np.array(labels)

                def __len__(self):
                    return len(self.sequences)

                def __getitem__(self, idx):
                    return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.sequence_labels[idx]])

            # Monkey-patch the dataset class
            lstm_security_model.ModSecurityDataset = ProcessedModSecurityDataset

            # Train the model with multi-class support
            best_val_acc = train_multiclass_lstm(args.processed, args.epochs, args.batch_size)
            print(f"✅ LSTM model trained successfully!")

        except Exception as e:
            print(f"❌ Error training LSTM model: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Step 3: Test integration with existing dashboard
    if os.path.exists("models/lstm_attack_classifier.pth"):
        print("\nStep 3: Testing model integration...")

        try:
            from enhanced_security_model import EnhancedAttackClassifier, EnhancedFeatureExtractor

            # Test enhanced model loading
            print("Testing enhanced model loading...")
            model = EnhancedAttackClassifier(model_path="models/lstm_attack_classifier.pth")
            print("✅ Enhanced model loaded successfully")

            # Test feature extraction
            if os.path.exists("data/processed/modsec_processed_preprocessor.pkl"):
                print("Testing enhanced feature extraction...")
                extractor = EnhancedFeatureExtractor("data/processed/modsec_processed_preprocessor.pkl")
                print("✅ Enhanced feature extractor loaded successfully")

                # Test with dummy data
                test_row = {
                    'request_line_url': '/wp-admin/admin-ajax.php?action=test',
                    'request_line_method': 'POST',
                    'response_status': '200',
                    'request_useragent': 'Mozilla/5.0',
                    'action': 'allowed'
                }

                features = extractor.extract_from_modsecurity_row(test_row)
                if features is not None:
                    print(f"✅ Feature extraction works: extracted {len(features)} features")
                else:
                    print("❌ Feature extraction failed")

        except Exception as e:
            print(f"❌ Error testing integration: {e}")

    print("\n" + "=" * 50)
    print("Training pipeline completed!")

    if args.preprocess and args.train:
        print("\n🎯 Training Summary:")
        print("✅ Preprocessed dataset with multi-class support")
        print("✅ Trained LSTM model with proper train/val/test split")
        print("✅ Saved model and label encoder separately")

        print("\n🚀 Next Steps to test with dashboard:")
        print("1. Test the trained model with sample data:")
        print("   python test_log_producer.py --stats-only")
        print("")
        print("2. Start streaming test data to dashboard:")
        print("   python test_log_producer.py --limit 50 --delay 0.5")
        print("")
        print("3. In another terminal, start the dashboard:")
        print("   python idsdashboard.py")
        print("")
        print("📁 Important files created:")
        print("   models/multiclass_lstm_classifier.pth  # Trained model")
        print("   models/label_encoder.pkl               # Class labels")
        print("   data/processed/modsec_processed_test.npz  # Test dataset")

        print("\n💡 Dashboard Integration:")
        print("   The current idsdashboard.py uses a binary classifier.")
        print("   To use the new multi-class model, you would need to:")
        print("   - Load the multiclass model and label encoder")
        print("   - Update the prediction logic to handle 8 classes")
        print("   - Map class indices back to attack types")

    return 0

if __name__ == "__main__":
    sys.exit(main())