#!/usr/bin/env python3
"""
Fixed Enhanced LSTM-based IDS Training Script
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

def train_simple_lstm(processed_path, epochs=20, batch_size=64):
    """Train LSTM model on preprocessed data."""
    print("Starting simplified LSTM training...")

    # Load data
    with np.load(processed_path, allow_pickle=True) as data:
        features = data['features']
        labels = data['labels']

    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
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

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Simple LSTM model
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size=15, hidden_size=64, output_size=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # x shape: (batch, seq_len, features) - need to add seq dimension
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension

            lstm_out, _ = self.lstm(x)
            # Use the last output
            last_out = lstm_out[:, -1, :]
            output = self.fc(last_out)
            return output

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = SimpleLSTM().to(device)
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

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
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

        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('models').mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'input_size': 15,
                'hidden_size': 64,
                'output_size': 2
            }, 'models/simple_lstm_classifier.pth')
            print(f'  ✅ New best model saved! Val Acc: {val_acc:.2f}%')

    print("\n✅ Simplified LSTM training completed!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: models/simple_lstm_classifier.pth")

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

            # Preprocess dataset
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
            best_val_acc = train_simple_lstm(args.processed, args.epochs, args.batch_size)
            print(f"\n🎯 Best validation accuracy: {best_val_acc:.2f}%")
            print(f"💾 Model saved to: models/simple_lstm_classifier.pth")

            print("\n🚀 You can now use the enhanced model:")
            print("python idsdashboard.py --enhanced")

            return 0

        except Exception as e:
            print(f"❌ Error training LSTM model: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("Training pipeline completed!")

if __name__ == "__main__":
    sys.exit(main())