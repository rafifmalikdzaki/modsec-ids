#!/usr/bin/env python3
"""
Simple LSTM Training Script for ModSecurity IDS
Fixes issues with the original training script
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import argparse

class SimpleModSecurityDataset(Dataset):
    """Simple dataset for preprocessed ModSecurity data."""

    def __init__(self, processed_path, sequence_length=50):
        print(f"Loading preprocessed data from {processed_path}")

        # Load the data safely
        with np.load(processed_path, allow_pickle=True) as data:
            self.features = data['features']
            self.labels = data['labels']

        self.sequence_length = sequence_length

        print(f"Loaded {len(self.features)} samples with {self.features.shape[1]} features")

        # Create sequences for LSTM
        self.sequences, self.sequence_labels = self._create_sequences()
        print(f"Created {len(self.sequences)} sequences for LSTM training")

    def _create_sequences(self):
        """Create sequences for LSTM training."""
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

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.sequence_labels[idx]])

class SimpleLSTMClassifier(nn.Module):
    """Simplified LSTM classifier."""

    def __init__(self, input_features=15, hidden_dim=64, num_layers=2, output_dim=2):
        super(SimpleLSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_features, hidden_dim, num_layers,
                           batch_first=True, dropout=0.3, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)

        # Global max pooling over sequence dimension
        pooled = torch.max(lstm_out, dim=1)[0]
        # pooled shape: (batch_size, hidden_dim * 2)

        # Fully connected layers
        x = self.fc1(pooled)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_lstm_model(processed_path, epochs=20, batch_size=64):
    """Train LSTM model on preprocessed data."""

    print("Starting LSTM Model Training for ModSecurity IDS")
    print(f"Configuration: Sequence length=50, Features=15, Hidden=64")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")

    # Load dataset
    print("Loading dataset...")
    dataset = SimpleModSecurityDataset(processed_path)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleLSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.squeeze().to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sequence_length': 50,
                'input_features': 15,
                'hidden_dim': 64,
                'num_layers': 2,
                'val_acc': val_acc
            }, 'models/lstm_attack_classifier.pth')
            print(f'  New best model saved! Val Acc: {val_acc:.2f}%')

        print('-' * 50)

    # Final evaluation on validation set
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.squeeze().to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print("\nFinal Classification Report:")
    target_names = ['Normal', 'Attack']
    print(classification_report(all_targets, all_predictions, target_names=target_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions))

    print("\n✅ Training completed successfully!")
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Simple LSTM Training for ModSecurity IDS")
    parser.add_argument("--processed", type=str, default="data/processed/modsec_processed.npz",
                       help="Path to preprocessed data")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Training batch size")
    parser.add_argument("--output", type=str, default="models/lstm_attack_classifier.pth",
                       help="Output model path")

    args = parser.parse_args()

    # Check if processed data exists
    if not os.path.exists(args.processed):
        print(f"❌ Processed dataset not found at {args.processed}")
        print("Please run preprocessing first:")
        print("python train_enhanced_ids.py --preprocess")
        return 1

    try:
        # Train the model
        best_loss = train_lstm_model(args.processed, args.epochs, args.batch_size)

        print(f"\n🎯 Best validation loss: {best_loss:.4f}")
        print(f"💾 Model saved to: {args.output}")

        print(f"\n🚀 You can now use the enhanced model:")
        print(f"python idsdashboard.py --enhanced")

        return 0

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())