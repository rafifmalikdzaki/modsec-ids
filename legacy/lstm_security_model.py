import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import numpy as np
import re
import urllib.parse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# Model Configuration
SEQUENCE_LENGTH = 50  # Process sequences of 50 requests
INPUT_FEATURES = 15   # Extended feature set for LSTM
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20

class ModSecurityDataset(Dataset):
    """Custom Dataset for ModSecurity data with sequence processing."""

    def __init__(self, processed_path=None, csv_file=None, sequence_length=SEQUENCE_LENGTH, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform

        if processed_path:
            # Load preprocessed data
            data = np.load(processed_path)
            self.features = data['features']
            self.labels = data['labels']

            # Create sequences for LSTM
            print("Creating sequences for LSTM training...")
            self.sequences, self.sequence_labels = self._create_sequences()
        elif csv_file:
            # Load raw CSV and process
            self.data = pd.read_csv(csv_file)

            # Initialize feature extractors
            self.ip_encoder = LabelEncoder()
            self.method_encoder = LabelEncoder()
            self.host_encoder = LabelEncoder()
            self.scaler = StandardScaler()

            # Process data
            self._process_data()
        else:
            raise ValueError("Either processed_path or csv_file must be provided")

    def _extract_features(self, row):
        """Extract comprehensive features from ModSecurity log entry."""

        # URL-based features
        url = str(row.get('request_line_url', '')).lower()
        decoded_url = urllib.parse.unquote(url)

        # User-agent features
        user_agent = str(row.get('request_useragent', '')).lower()

        # Host features
        host = str(row.get('request_host', '')).lower()

        # Status code features
        status = int(row.get('response_status', 200))

        # Action/message features
        action = str(row.get('action', '-'))
        message = str(row.get('action_message', '')).lower()

        # Create feature vector (15 features)
        features = [
            # URL-based features (8)
            float(len(url)) / 200.0,  # URL length (normalized)
            float(len(decoded_url)) / 200.0,  # Decoded URL length
            1.0 if 'wp-admin' in decoded_url else 0.0,  # Admin area
            1.0 if 'admin-ajax' in decoded_url else 0.0,  # Ajax endpoint
            1.0 if any(char in decoded_url for char in ['<', '>', '\'', '"', '&']) else 0.0,  # XSS chars
            1.0 if 'base64' in decoded_url else 0.0,  # Base64 encoding
            1.0 if 'union' in decoded_url else 0.0,  # SQL injection
            1.0 if any(payload in decoded_url for payload in ['exec', 'system', 'shell', 'cmd']) else 0.0,  # RCE

            # HTTP features (3)
            1.0 if str(row.get('request_line_method', 'GET')) == 'POST' else 0.0,  # POST method
            float(status) / 600.0,  # Status code (normalized)
            1.0 if status >= 400 else 0.0,  # Error status

            # User-agent features (2)
            1.0 if any(bot in user_agent for bot in ['bot', 'crawler', 'spider', 'scraper']) else 0.0,  # Bot
            float(len(user_agent)) / 500.0,  # User-agent length

            # Attack pattern features (2)
            1.0 if 'blocked' in action.lower() else 0.0,  # Was blocked
            1.0 if any(pattern in message for pattern in ['sqli', 'xss', 'rce', 'lfi', 'rfi']) else 0.0,  # Attack pattern
        ]

        return np.array(features, dtype=np.float32)

    def _process_data(self):
        """Process the entire dataset into sequences."""
        features_list = []
        labels_list = []

        print("Extracting features from dataset...")
        for idx, row in self.data.iterrows():
            if idx % 10000 == 0:
                print(f"Processed {idx} records...")

            # Extract features
            features = self._extract_features(row)

            # Create label (binary: 0 for normal, 1 for malicious)
            label_str = str(row.get('label', 'normal')).strip().lower()
            label = 1 if 'malicious' in label_str or 'attack' in label_str else 0

            features_list.append(features)
            labels_list.append(label)

        # Convert to numpy arrays
        self.features = np.array(features_list)
        self.labels = np.array(labels_list)

        # Normalize features
        print("Normalizing features...")
        self.features = self.scaler.fit_transform(self.features)

        # Create sequences for LSTM
        print("Creating sequences...")
        self.sequences, self.sequence_labels = self._create_sequences()

        print(f"Dataset processed: {len(self.sequences)} sequences")
        print(f"Label distribution: {np.bincount(self.sequence_labels)}")

    def _create_sequences(self):
        """Create sequences for LSTM training."""
        sequences = []
        labels = []

        # Pad with zeros at the beginning if needed
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

class LSTMAttackClassifier(nn.Module):
    """LSTM-based neural network for attack classification."""

    def __init__(self, input_features=INPUT_FEATURES, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM):
        super(LSTMAttackClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_features, hidden_dim, num_layers,
                           batch_first=True, dropout=0.3, bidirectional=True)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Global max pooling over attention output
        pooled = torch.max(attn_out, dim=1)[0]

        # Fully connected layers
        x = self.fc1(pooled)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_model():
    """Train the LSTM model on ModSecurity dataset."""

    print("Starting LSTM Model Training for ModSecurity IDS")
    print(f"Configuration: SEQ_LEN={SEQUENCE_LENGTH}, FEATURES={INPUT_FEATURES}, HIDDEN={HIDDEN_DIM}")

    # Check if dataset exists
    csv_path = 'data/raw/Modsec-WP.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    # Load dataset
    print("Loading dataset...")
    dataset = ModSecurityDataset(csv_path)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LSTMAttackClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
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

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}')

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

        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': dataset.scaler,
                'sequence_length': SEQUENCE_LENGTH,
                'input_features': INPUT_FEATURES,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'val_acc': val_acc
            }, 'models/lstm_attack_classifier.pth')
            print(f'  New best model saved! Val Acc: {val_acc:.2f}%')

        print('-' * 50)

    print("Training completed!")

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
    print(classification_report(all_targets, all_predictions, target_names=['Normal', 'Attack']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions))

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    train_model()