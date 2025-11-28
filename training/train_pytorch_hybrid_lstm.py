# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed untuk reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
# ====== 1. LOAD DATA ======
df = pd.read_excel('../data/Modsec-WP.xlsx')
print(f"Dataset loaded: {df.shape}")

# %%
# ====== 2. PREPROCESSING ======

# 2.1 Handle Missing Values
df = df.replace('-', np.nan)
df['request_body'] = df['request_body'].fillna('')
df['response_body'] = df['response_body'].fillna('')
df['full_message_line'] = df['full_message_line'].fillna('')

# 2.2 Feature Selection
text_features = ['request_useragent', 'request_line_url', 'full_message_line']
categorical_features = ['request_line_method', 'action', 'message_type']
numerical_features = ['response_status']

# 2.3 Encode Label
print(f"Unique labels: {df['label'].unique()}")
print(f"Label counts:\n{df['label'].value_counts()}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

print(f"Encoded labels: {np.unique(y)}")
print(f"Number of classes: {len(label_encoder.classes_)}")

# PENTING: Jika lebih dari 2 kelas, gunakan CrossEntropyLoss
# Jika 2 kelas, pastikan label adalah 0 dan 1
if len(label_encoder.classes_) > 2:
    print("⚠️ WARNING: More than 2 classes detected! Need to use CrossEntropyLoss")
    print("Classes:", label_encoder.classes_)

# 2.4 Build Vocabulary untuk Text Features
MAX_WORDS = 10000
MAX_LEN = 100

class TextVectorizer:
    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx = 2
    
    def fit(self, texts):
        word_freq = {}
        for text in texts:
            for word in str(text).lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Ambil top max_words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_words - 2]:
            self.word2idx[word] = self.idx
            self.idx += 1
    
    def transform(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in str(text).lower().split()[:self.max_len]:
                seq.append(self.word2idx.get(word, 1))  # 1 = <UNK>
            # Padding
            seq = seq + [0] * (self.max_len - len(seq))
            sequences.append(seq[:self.max_len])
        return np.array(sequences)

# Vectorize text features
vectorizers = {}
text_sequences = {}

for feature in text_features:
    vectorizer = TextVectorizer(max_words=MAX_WORDS, max_len=MAX_LEN)
    vectorizer.fit(df[feature].astype(str))
    sequences = vectorizer.transform(df[feature].astype(str))
    
    vectorizers[feature] = vectorizer
    text_sequences[feature] = sequences

print(f"Text features vectorized. Vocab size: {MAX_WORDS}")

# 2.5 Encode Categorical Features
categorical_encoders = {}
categorical_encoded = {}

for feature in categorical_features:
    le = LabelEncoder()
    categorical_encoded[feature] = le.fit_transform(df[feature].fillna('unknown'))
    categorical_encoders[feature] = le

print(f"Categorical features encoded")

# 2.6 Scale Numerical Features
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df[numerical_features].fillna(0))

print(f"Numerical features scaled")

# %%
# ====== 3. CREATE PYTORCH DATASET ======

class IDSDataset(Dataset):
    def __init__(self, text_data, cat_data, num_data, labels):
        self.text_data = text_data
        self.cat_data = cat_data
        self.num_data = torch.FloatTensor(num_data)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Text features
        text_dict = {
            name: torch.LongTensor(data[idx]) 
            for name, data in self.text_data.items()
        }
        
        # Categorical features
        cat_dict = {
            name: torch.LongTensor([data[idx]]) 
            for name, data in self.cat_data.items()
        }
        
        # Numerical features
        num = self.num_data[idx]
        
        # Label
        label = self.labels[idx]
        
        return text_dict, cat_dict, num, label

# Split data
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

# Prepare train data
text_train = {name: data[train_idx] for name, data in text_sequences.items()}
cat_train = {name: data[train_idx] for name, data in categorical_encoded.items()}
num_train = numerical_scaled[train_idx]
y_train = y[train_idx]

# Prepare test data
text_test = {name: data[test_idx] for name, data in text_sequences.items()}
cat_test = {name: data[test_idx] for name, data in categorical_encoded.items()}
num_test = numerical_scaled[test_idx]
y_test = y[test_idx]

# Create datasets
train_dataset = IDSDataset(text_train, cat_train, num_train, y_train)
test_dataset = IDSDataset(text_test, cat_test, num_test, y_test)

# Create dataloaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# %%
# ====== 4. BUILD PYTORCH MODEL ======

class HybridIDSModel(nn.Module):
    def __init__(self, text_features, cat_features, num_features_dim,
                 vocab_size=10000, embed_dim=128, lstm_hidden=64, 
                 cat_embed_dim=10, num_classes=8):  # ADD num_classes
        super(HybridIDSModel, self).__init__()
        
        self.text_features = text_features
        self.cat_features = cat_features
        
        # Text Embeddings dan LSTM
        self.text_embeddings = nn.ModuleDict()
        self.text_lstms = nn.ModuleDict()
        
        for feature in text_features:
            self.text_embeddings[feature] = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.text_lstms[feature] = nn.LSTM(
                embed_dim, lstm_hidden, batch_first=True, dropout=0.2
            )
        
        # Categorical Embeddings
        self.cat_embeddings = nn.ModuleDict()
        for feature, n_unique in cat_features.items():
            self.cat_embeddings[feature] = nn.Embedding(n_unique, cat_embed_dim)
        
        # Calculate total input dimension
        total_dim = (
            len(text_features) * lstm_hidden +  # LSTM outputs
            len(cat_features) * cat_embed_dim +  # Cat embeddings
            num_features_dim  # Numerical features
        )
        
        # Classification layers
        self.fc1 = nn.Linear(total_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, num_classes)  # CHANGE: output num_classes
        
        self.relu = nn.ReLU()
    
    def forward(self, text_dict, cat_dict, num_features):
        # Process text features
        text_outputs = []
        for feature in self.text_features:
            embedded = self.text_embeddings[feature](text_dict[feature])
            lstm_out, (hidden, _) = self.text_lstms[feature](embedded)
            # Use last hidden state
            text_outputs.append(hidden[-1])
        
        # Process categorical features
        cat_outputs = []
        for feature in self.cat_features:
            embedded = self.cat_embeddings[feature](cat_dict[feature])
            cat_outputs.append(embedded.squeeze(1))
        
        # Reshape numerical features to [batch_size, num_features_dim]
        if num_features.dim() == 1:
            num_features = num_features.unsqueeze(1)
        
        # Concatenate all features
        combined = torch.cat(text_outputs + cat_outputs + [num_features], dim=1)
        
        # Classification
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        output = self.fc3(x)
        
        return output

# Get categorical feature unique counts
cat_features_config = {
    feature: len(categorical_encoders[feature].classes_) + 1
    for feature in categorical_features
}

# Initialize model
num_classes = len(label_encoder.classes_)
model = HybridIDSModel(
    text_features=text_features,
    cat_features=cat_features_config,
    num_features_dim=len(numerical_features),
    vocab_size=MAX_WORDS,
    embed_dim=128,
    lstm_hidden=64,
    cat_embed_dim=10,
    num_classes=num_classes  # ADD THIS
)

model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel initialized")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\nModel Architecture:")
print(model)

# %%
# ====== 5. TRAINING SETUP ======

# Loss function dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

early_stopping = EarlyStopping(patience=5)

print("Training setup complete")

# %%
# ====== 6. TRAINING LOOP ======

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for text_dict, cat_dict, num, labels in dataloader:
        # Move to device
        text_dict = {k: v.to(device) for k, v in text_dict.items()}
        cat_dict = {k: v.to(device) for k, v in cat_dict.items()}
        num = num.to(device)
        labels = labels.to(device).long()  # CHANGE: .long() not .unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(text_dict, cat_dict, num)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)  # CHANGE: argmax instead of > 0.5
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text_dict, cat_dict, num, labels in dataloader:
            # Move to device
            text_dict = {k: v.to(device) for k, v in text_dict.items()}
            cat_dict = {k: v.to(device) for k, v in cat_dict.items()}
            num = num.to(device)
            labels = labels.to(device).long()
            
            # Forward pass
            outputs = model(text_dict, cat_dict, num)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Collect predictions
            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    # CHANGE: Add average='weighted' for multiclass
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

# Training
NUM_EPOCHS = 50
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'val_precision': [], 'val_recall': [], 'val_f1': []
}

print("Starting training...\n")

for epoch in range(NUM_EPOCHS):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Evaluate
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
        model, test_loader, criterion, device
    )
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_precision'].append(val_prec)
    history['val_recall'].append(val_rec)
    history['val_f1'].append(val_f1)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    early_stopping(val_loss, model)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"  Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}")
    print()
    
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        model.load_state_dict(early_stopping.best_model)
        break

print("Training completed!")

# Save best model
torch.save(model.state_dict(), 'best_ids_model.pth')
print("Model saved to: best_ids_model.pth")

# %%
# ====== 7. FINAL EVALUATION ======

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for text_dict, cat_dict, num, labels in test_loader:
        text_dict = {k: v.to(device) for k, v in text_dict.items()}
        cat_dict = {k: v.to(device) for k, v in cat_dict.items()}
        num = num.to(device)
        
        outputs = model(text_dict, cat_dict, num)
        
        # CHANGE: Use argmax for multiclass
        predicted = outputs.argmax(dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics - ADD average='weighted'
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print("=" * 50)
print("📊 FINAL MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=" * 50)

# Classification report
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, 
                          target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Training History
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss')
axes[0, 0].plot(history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Model Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy
axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
axes[0, 1].set_title('Model Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision & Recall
axes[1, 0].plot(history['val_precision'], label='Precision', marker='o')
axes[1, 0].plot(history['val_recall'], label='Recall', marker='s')
axes[1, 0].set_title('Precision & Recall')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Score')
axes[1, 0].legend()
axes[1, 0].grid(True)

# F1 Score
axes[1, 1].plot(history['val_f1'], label='F1-Score', color='green', marker='^')
axes[1, 1].set_title('F1-Score')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# %%
"""
Save all artifacts properly dengan module import yang benar
Jalankan cell ini SETELAH training selesai
"""

import pickle
import json
import sys
from pathlib import Path
import shutil

# ⭐ PENTING: Import TextVectorizer dari module agar pickle compatibility baik
project_root = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
sys.path.append(str(project_root))

# Import dari module (bukan define ulang di notebook)
from src.model_architecture import TextVectorizer as ModuleTextVectorizer

print("=" * 60)
print("💾 SAVING PREPROCESSING ARTIFACTS FOR INFERENCE")
print("=" * 60)

# Create models directory if not exists
models_dir = project_root / 'models'
models_dir.mkdir(exist_ok=True)

# ⭐ FIX: Re-create vectorizers dengan class dari module
print("\n1️⃣ Re-creating vectorizers with correct module path...")
new_vectorizers = {}

for feature_name, old_vectorizer in vectorizers.items():
    # Create new vectorizer dengan class dari module
    new_vec = ModuleTextVectorizer(max_words=MAX_WORDS, max_len=MAX_LEN)
    
    # Copy vocabulary dari old vectorizer
    new_vec.word2idx = old_vectorizer.word2idx.copy()
    new_vec.idx = old_vectorizer.idx
    
    new_vectorizers[feature_name] = new_vec
    print(f"   ✓ Re-created vectorizer for '{feature_name}' (vocab size: {len(new_vec.word2idx)})")

# Replace dengan new vectorizers
vectorizers = new_vectorizers

# 1. Save Vectorizers (dengan module path yang benar)
print("\n2️⃣ Saving vectorizers...")
with open(models_dir / 'vectorizers.pkl', 'wb') as f:
    pickle.dump(vectorizers, f)
print(f"   ✓ Vectorizers saved: {models_dir / 'vectorizers.pkl'}")

# Verify immediately
print("\n3️⃣ Verifying vectorizers...")
with open(models_dir / 'vectorizers.pkl', 'rb') as f:
    test_load = pickle.load(f)
print(f"   ✓ Verification successful! Loaded {len(test_load)} vectorizers")

# 2. Save Categorical Encoders
print("\n4️⃣ Saving categorical encoders...")
with open(models_dir / 'categorical_encoders.pkl', 'wb') as f:
    pickle.dump(categorical_encoders, f)
print(f"   ✓ Categorical encoders saved")

# 3. Save Label Encoder
print("\n5️⃣ Saving label encoder...")
with open(models_dir / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"   ✓ Label encoder saved")

# 4. Save Scaler
print("\n6️⃣ Saving scaler...")
with open(models_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Scaler saved")

# 5. Save Model Configuration
print("\n7️⃣ Saving model configuration...")
model_config = {
    'text_features': text_features,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'cat_features_config': cat_features_config,
    'num_classes': num_classes,
    'vocab_size': MAX_WORDS,
    'max_len': MAX_LEN,
    'embed_dim': 128,
    'lstm_hidden': 64,
    'cat_embed_dim': 10
}

with open(models_dir / 'model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)
print(f"   ✓ Model config saved")

# 6. Save Feature Info
print("\n8️⃣ Saving feature info...")
feature_info = {
    'text_features': text_features,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'label_classes': label_encoder.classes_.tolist()
}

with open(models_dir / 'feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=4)
print(f"   ✓ Feature info saved")

# 7. Move model weights
print("\n9️⃣ Moving model weights...")
current_model_path = Path('best_ids_model.pth')
target_model_path = models_dir / 'best_ids_model.pth'

if current_model_path.exists():
    if target_model_path.exists():
        target_model_path.unlink()  # Delete old one
    shutil.move(str(current_model_path), str(target_model_path))
    print(f"   ✓ Model weights moved to: {target_model_path}")
else:
    print(f"   ⚠️  Model weights not found at: {current_model_path}")

# 8. Final verification
print("\n" + "=" * 60)
print("🔍 FINAL VERIFICATION")
print("=" * 60)

files_to_check = [
    'best_ids_model.pth',
    'vectorizers.pkl',
    'categorical_encoders.pkl',
    'label_encoder.pkl',
    'scaler.pkl',
    'model_config.json',
    'feature_info.json'
]

all_good = True
for filename in files_to_check:
    filepath = models_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size
        if size > 0:
            print(f"   ✅ {filename:30s} ({size:,} bytes)")
        else:
            print(f"   ❌ {filename:30s} (0 bytes - CORRUPT!)")
            all_good = False
    else:
        print(f"   ❌ {filename:30s} (NOT FOUND)")
        all_good = False

print("\n" + "=" * 60)
if all_good:
    print("✅ ALL ARTIFACTS SAVED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📁 All files saved in: {models_dir}")
    print("\n🚀 You can now use inference!")
    print("\nNext steps:")
    print("  1. Run: jupyter notebook inference_demo.ipynb")
    print("  2. Or run: python scripts/inference_example.py")
else:
    print("❌ SOME FILES ARE MISSING OR CORRUPT!")
    print("=" * 60)
    print("\n⚠️  Please check the errors above and re-run this cell")

# %%
"""
🔍 INFERENCE DEMO - REAL-TIME PREDICTION
Test model dengan sample dari test set dan data custom
Jalankan cell ini setelah training selesai
"""

import random
import torch
import numpy as np

print("=" * 80)
print("🔍 INFERENCE DEMO - TESTING MODEL PREDICTIONS")
print("=" * 80)

# ====================================================================
# INFERENCE FUNCTION
# ====================================================================

def predict_single_sample(text_dict, cat_dict, num_features, 
                         model, device, label_encoder):
    """
    Predict a single sample
    
    Args:
        text_dict: Dict of text sequences {feature_name: sequence}
        cat_dict: Dict of categorical values {feature_name: encoded_value}
        num_features: Numerical feature values
        model: Trained model
        device: torch device
        label_encoder: Label encoder for decoding predictions
    
    Returns:
        predicted_label: Predicted class name
        confidence: Confidence score (0-1)
        probabilities: Dict of all class probabilities
    """
    model.eval()
    
    with torch.no_grad():
        # Convert to tensors
        text_tensors = {
            name: torch.LongTensor(seq).unsqueeze(0).to(device)
            for name, seq in text_dict.items()
        }
        
        cat_tensors = {
            name: torch.LongTensor([val]).to(device)
            for name, val in cat_dict.items()
        }
        
        num_tensor = torch.FloatTensor(num_features).unsqueeze(0).to(device)
        
        # Forward pass
        outputs = model(text_tensors, cat_tensors, num_tensor)
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)[0]
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        # Decode label
        predicted_label = label_encoder.classes_[pred_idx]
        
        # All class probabilities
        all_probs = {
            label_encoder.classes_[i]: probs[i].item()
            for i in range(len(label_encoder.classes_))
        }
    
    return predicted_label, confidence, all_probs

# ====================================================================
# FUNCTION: Prepare sample for inference
# ====================================================================

def prepare_sample_for_inference(idx, df, text_sequences, categorical_encoded, 
                                numerical_scaled, vectorizers):
    """
    Prepare a single sample from dataset for inference
    """
    # Text features (sequences)
    text_dict = {
        name: sequences[idx] 
        for name, sequences in text_sequences.items()
    }
    
    # Categorical features (encoded values)
    cat_dict = {
        name: encoded[idx] 
        for name, encoded in categorical_encoded.items()
    }
    
    # Numerical features
    num_features = numerical_scaled[idx]
    
    # Original raw data for display
    raw_data = {
        'request_useragent': df.iloc[idx]['request_useragent'],
        'request_line_url': df.iloc[idx]['request_line_url'],
        'full_message_line': df.iloc[idx]['full_message_line'],
        'request_line_method': df.iloc[idx]['request_line_method'],
        'action': df.iloc[idx]['action'],
        'message_type': df.iloc[idx]['message_type'],
        'response_status': df.iloc[idx]['response_status'],
        'true_label': df.iloc[idx]['label']
    }
    
    return text_dict, cat_dict, num_features, raw_data

# ====================================================================
# 1. DEMO: Random Samples from Test Set
# ====================================================================

# Use the same split as in training
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=label_encoder.transform(df['label'])
)

print("\n" + "=" * 80)
print("1️⃣ DEMO: RANDOM SAMPLES FROM TEST SET")
print("=" * 80)

# Select random samples
num_samples = 15
sample_indices = np.random.choice(test_idx, min(num_samples, len(test_idx)), replace=False)

correct_predictions = 0
total_predictions = 0

for i, idx in enumerate(sample_indices, 1):
    print(f"\n{'='*80}")
    print(f"📊 SAMPLE {i}/{len(sample_indices)} (Index: {idx})")
    print(f"{'='*80}")
    
    # Prepare sample
    text_dict, cat_dict, num_features, raw_data = prepare_sample_for_inference(
        idx, df, text_sequences, categorical_encoded, numerical_scaled, vectorizers
    )
    
    # Predict
    predicted_label, confidence, all_probs = predict_single_sample(
        text_dict, cat_dict, num_features, model, device, label_encoder
    )
    
    # Display raw data (truncated)
    print(f"\n📝 REQUEST DATA:")
    print(f"   URL: {str(raw_data['request_line_url'])[:80]}...")
    print(f"   Method: {raw_data['request_line_method']}")
    print(f"   User Agent: {str(raw_data['request_useragent'])[:60]}...")
    print(f"   Action: {raw_data['action']}")
    print(f"   Message: {str(raw_data['full_message_line'])[:80]}...")
    print(f"   Response Status: {raw_data['response_status']}")
    
    # Display prediction
    print(f"\n🎯 PREDICTION:")
    print(f"   True Label: {raw_data['true_label']}")
    print(f"   Predicted: {predicted_label}")
    print(f"   Confidence: {confidence:.2%}")
    
    # Check if correct
    is_correct = predicted_label == raw_data['true_label']
    correct_predictions += int(is_correct)
    total_predictions += 1
    
    if is_correct:
        print(f"   ✅ CORRECT!")
    else:
        print(f"   ❌ INCORRECT!")
    
    # Top 3 predictions
    print(f"\n📊 TOP 3 PREDICTIONS:")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for rank, (label, prob) in enumerate(sorted_probs[:3], 1):
        bar = '█' * int(prob * 50)
        marker = '👈' if label == predicted_label else ''
        print(f"   {rank}. {label:25s}: {prob:6.2%} {bar} {marker}")
    
    print(f"\n{'='*80}")

# Summary
print(f"\n" + "=" * 80)
print(f"📈 DEMO SUMMARY")
print(f"=" * 80)
print(f"   Total Samples: {total_predictions}")
print(f"   Correct: {correct_predictions}")
print(f"   Incorrect: {total_predictions - correct_predictions}")
print(f"   Accuracy: {correct_predictions/total_predictions:.2%}")
print(f"=" * 80)

# ====================================================================
# 2. DEMO: Inference by Attack Type
# ====================================================================

print("\n\n" + "=" * 80)
print("2️⃣ DEMO: INFERENCE BY ATTACK TYPE")
print("=" * 80)

# Get unique labels
unique_labels = df['label'].unique()

print(f"\nTesting 2 samples from each attack type...\n")

for attack_type in unique_labels:
    print(f"\n{'─'*80}")
    print(f"🔴 ATTACK TYPE: {attack_type}")
    print(f"{'─'*80}")
    
    # Get indices for this attack type in test set
    attack_indices = test_idx[df.iloc[test_idx]['label'] == attack_type]
    
    if len(attack_indices) == 0:
        print(f"   ⚠️  No samples of '{attack_type}' in test set")
        continue
    
    # Sample 2 instances
    sample_count = min(2, len(attack_indices))
    samples = np.random.choice(attack_indices, sample_count, replace=False)
    
    for j, idx in enumerate(samples, 1):
        print(f"\n   Sample {j}:")
        
        # Prepare and predict
        text_dict, cat_dict, num_features, raw_data = prepare_sample_for_inference(
            idx, df, text_sequences, categorical_encoded, numerical_scaled, vectorizers
        )
        
        predicted_label, confidence, all_probs = predict_single_sample(
            text_dict, cat_dict, num_features, model, device, label_encoder
        )
        
        # Display compact info
        print(f"      URL: {str(raw_data['request_line_url'])[:60]}...")
        print(f"      True: {raw_data['true_label']}")
        print(f"      Pred: {predicted_label} ({confidence:.1%})")
        print(f"      {'✅ CORRECT' if predicted_label == raw_data['true_label'] else '❌ INCORRECT'}")

print(f"\n{'='*80}")

# %%
# ====================================================================
# 3. DEMO: Custom Inference Function
# ====================================================================

print("\n\n" + "=" * 80)
print("3️⃣ DEMO: CUSTOM INFERENCE FUNCTION")
print("=" * 80)

def predict_from_raw_input(url, method, user_agent, message, action, 
                           message_type, response_status,
                           model, vectorizers, categorical_encoders, 
                           scaler, label_encoder, device):
    """
    Predict attack type from raw input data
    
    Args:
        url: Request URL string
        method: HTTP method (GET, POST, etc.)
        user_agent: User agent string
        message: Full message line
        action: Action type
        message_type: Message type
        response_status: HTTP response status code
        ... (preprocessors and model)
    
    Returns:
        predicted_label, confidence, all_probs
    """
    # 1. Vectorize text features
    text_dict = {}
    text_inputs = {
        'request_useragent': user_agent,
        'request_line_url': url,
        'full_message_line': message
    }
    
    for feature_name, text in text_inputs.items():
        vectorizer = vectorizers[feature_name]
        sequence = vectorizer.transform([text])[0]
        text_dict[feature_name] = sequence
    
    # 2. Encode categorical features
    cat_dict = {}
    cat_inputs = {
        'request_line_method': method,
        'action': action,
        'message_type': message_type
    }
    
    for feature_name, value in cat_inputs.items():
        encoder = categorical_encoders[feature_name]
        try:
            encoded = encoder.transform([value])[0]
        except:
            # Unknown value, use 0 (or most common)
            encoded = 0
        cat_dict[feature_name] = encoded
    
    # 3. Scale numerical features
    num_features = scaler.transform([[response_status]])[0]
    
    # 4. Predict
    predicted_label, confidence, all_probs = predict_single_sample(
        text_dict, cat_dict, num_features, model, device, label_encoder
    )
    
    return predicted_label, confidence, all_probs

# Test custom inference
print("\n💡 Testing custom inference function with synthetic examples:\n")

# Example 1: SQL Injection
print("Example 1: SQL Injection Attack")
print("-" * 80)
url1 = "/admin/login.php?id=1' OR '1'='1"
method1 = "GET"
ua1 = "Mozilla/5.0 (Windows NT 10.0)"
message1 = "SQL Injection detected in query parameters"
action1 = "block"
msg_type1 = "error"
status1 = 403

pred1, conf1, probs1 = predict_from_raw_input(
    url1, method1, ua1, message1, action1, msg_type1, status1,
    model, vectorizers, categorical_encoders, scaler, label_encoder, device
)

print(f"URL: {url1}")
print(f"Predicted: {pred1} (Confidence: {conf1:.2%})")
print(f"Top 3:")
for label, prob in sorted(probs1.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"   {label:25s}: {prob:.2%}")

# Example 2: XSS
print("\n\nExample 2: Cross-Site Scripting (XSS)")
print("-" * 80)
url2 = "/search?q=<script>alert('XSS')</script>"
method2 = "GET"
ua2 = "Mozilla/5.0"
message2 = "XSS attempt detected"
action2 = "block"
msg_type2 = "warning"
status2 = 403

pred2, conf2, probs2 = predict_from_raw_input(
    url2, method2, ua2, message2, action2, msg_type2, status2,
    model, vectorizers, categorical_encoders, scaler, label_encoder, device
)

print(f"URL: {url2}")
print(f"Predicted: {pred2} (Confidence: {conf2:.2%})")
print(f"Top 3:")
for label, prob in sorted(probs2.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"   {label:25s}: {prob:.2%}")

# Example 3: Path Traversal
print("\n\nExample 3: Path Traversal Attack")
print("-" * 80)
url3 = "/download?file=../../../../etc/passwd"
method3 = "GET"
ua3 = "curl/7.68.0"
message3 = "Path traversal attempt detected"
action3 = "block"
msg_type3 = "error"
status3 = 403

pred3, conf3, probs3 = predict_from_raw_input(
    url3, method3, ua3, message3, action3, msg_type3, status3,
    model, vectorizers, categorical_encoders, scaler, label_encoder, device
)

print(f"URL: {url3}")
print(f"Predicted: {pred3} (Confidence: {conf3:.2%})")
print(f"Top 3:")
for label, prob in sorted(probs3.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"   {label:25s}: {prob:.2%}")

print("\n" + "=" * 80)

# ====================================================================
# 4. CONFIDENCE DISTRIBUTION ANALYSIS
# ====================================================================

print("\n\n" + "=" * 80)
print("4️⃣ CONFIDENCE DISTRIBUTION ANALYSIS")
print("=" * 80)

print("\nAnalyzing prediction confidence across test set...\n")

# Get all predictions with confidence
all_confidences = []
all_correct = []

for idx in test_idx[:min(500, len(test_idx))]:  # Analyze up to 500 samples
    text_dict, cat_dict, num_features, raw_data = prepare_sample_for_inference(
        idx, df, text_sequences, categorical_encoded, numerical_scaled, vectorizers
    )
    
    predicted_label, confidence, _ = predict_single_sample(
        text_dict, cat_dict, num_features, model, device, label_encoder
    )
    
    is_correct = predicted_label == raw_data['true_label']
    
    all_confidences.append(confidence)
    all_correct.append(is_correct)

# Analyze confidence by correctness
correct_confidences = [c for c, correct in zip(all_confidences, all_correct) if correct]
incorrect_confidences = [c for c, correct in zip(all_confidences, all_correct) if not correct]

print(f"📊 Confidence Statistics:")
print(f"   Overall:")
print(f"      Mean Confidence: {np.mean(all_confidences):.2%}")
print(f"      Median Confidence: {np.median(all_confidences):.2%}")
print(f"      Min Confidence: {np.min(all_confidences):.2%}")
print(f"      Max Confidence: {np.max(all_confidences):.2%}")

if len(correct_confidences) > 0:
    print(f"\n   Correct Predictions:")
    print(f"      Mean Confidence: {np.mean(correct_confidences):.2%}")
    print(f"      Median Confidence: {np.median(correct_confidences):.2%}")

if len(incorrect_confidences) > 0:
    print(f"\n   Incorrect Predictions:")
    print(f"      Mean Confidence: {np.mean(incorrect_confidences):.2%}")
    print(f"      Median Confidence: {np.median(incorrect_confidences):.2%}")

# Confidence bins
print(f"\n📊 Confidence Distribution:")
bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
bin_labels = ['0-50%', '50-70%', '70-80%', '80-90%', '90-95%', '95-100%']

for i in range(len(bins)-1):
    count = sum(1 for c in all_confidences if bins[i] <= c < bins[i+1])
    pct = count / len(all_confidences) * 100
    bar = '█' * int(pct / 2)
    print(f"   {bin_labels[i]:10s}: {count:4d} ({pct:5.1f}%) {bar}")

print("\n" + "=" * 80)
print("✅ INFERENCE DEMO COMPLETED!")
print("=" * 80)

print("\n💡 Tips:")
print("   - High confidence (>90%) usually indicates strong prediction")
print("   - Low confidence (<70%) suggests model uncertainty")
print("   - Check top 3 predictions when confidence is low")
print("   - Model performs best on attack types well-represented in training data")
print("\n" + "=" * 80)

# %%
"""
📊 VISUALIZATION: Confidence Distribution
"""

import matplotlib.pyplot as plt
import seaborn as sns

print("Generating confidence distribution visualizations...")

# Prepare data
correct_confidences = [c for c, correct in zip(all_confidences, all_correct) if correct]
incorrect_confidences = [c for c, correct in zip(all_confidences, all_correct) if not correct]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Overall confidence distribution
ax1 = axes[0, 0]
ax1.hist(all_confidences, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(all_confidences), color='red', linestyle='--', 
           label=f'Mean: {np.mean(all_confidences):.2%}')
ax1.axvline(np.median(all_confidences), color='green', linestyle='--',
           label=f'Median: {np.median(all_confidences):.2%}')
ax1.set_xlabel('Confidence')
ax1.set_ylabel('Frequency')
ax1.set_title('Overall Confidence Distribution', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Correct vs Incorrect confidence
ax2 = axes[0, 1]
if len(correct_confidences) > 0:
    ax2.hist(correct_confidences, bins=20, color='green', alpha=0.5, 
            label=f'Correct ({len(correct_confidences)})', edgecolor='black')
if len(incorrect_confidences) > 0:
    ax2.hist(incorrect_confidences, bins=20, color='red', alpha=0.5,
            label=f'Incorrect ({len(incorrect_confidences)})', edgecolor='black')
ax2.set_xlabel('Confidence')
ax2.set_ylabel('Frequency')
ax2.set_title('Confidence: Correct vs Incorrect Predictions', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Box plot comparison
ax3 = axes[1, 0]
box_data = []
box_labels = []
if len(correct_confidences) > 0:
    box_data.append(correct_confidences)
    box_labels.append('Correct')
if len(incorrect_confidences) > 0:
    box_data.append(incorrect_confidences)
    box_labels.append('Incorrect')

bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2))
ax3.set_ylabel('Confidence')
ax3.set_title('Confidence Comparison (Box Plot)', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. Confidence bins
ax4 = axes[1, 1]
bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
bin_labels = ['0-50%', '50-70%', '70-80%', '80-90%', '90-95%', '95-100%']
bin_counts = [sum(1 for c in all_confidences if bins[i] <= c < bins[i+1]) 
              for i in range(len(bins)-1)]

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bin_counts)))
bars = ax4.bar(bin_labels, bin_counts, color=colors, edgecolor='black')
ax4.set_xlabel('Confidence Range')
ax4.set_ylabel('Count')
ax4.set_title('Confidence Distribution by Bins', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.suptitle('📊 Model Confidence Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('inference_confidence_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: inference_confidence_analysis.png")
plt.show()

print("\n" + "=" * 80)
print("✅ CONFIDENCE ANALYSIS VISUALIZATION COMPLETED!")
print("=" * 80)


