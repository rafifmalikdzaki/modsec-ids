"""
Hybrid LSTM IDS Model Architecture untuk ModSecurity
Compatible dengan inference engine yang ada
"""

import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np


class TextVectorizer:
    """
    Text vectorizer untuk preprocessing - HARUS SAMA dengan saat training
    """
    
    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx = 2
    
    def fit(self, texts):
        """Build vocabulary dari training texts"""
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
        """Convert texts ke sequences"""
        if isinstance(texts, str):
            texts = [texts]
            
        sequences = []
        for text in texts:
            seq = []
            for word in str(text).lower().split()[:self.max_len]:
                seq.append(self.word2idx.get(word, 1))  # 1 = <UNK>
            # Padding
            seq = seq + [0] * (self.max_len - len(seq))
            sequences.append(seq[:self.max_len])
        return sequences


class HybridIDSModel(nn.Module):
    """
    Hybrid LSTM model untuk IDS dengan:
    - Multiple text features (request_useragent, request_line_url, full_message_line)
    - Categorical features (method, action, message_type)
    - Numerical features (response_status)
    """
    
    def __init__(self, text_features, cat_features, num_features_dim,
                 vocab_size=10000, embed_dim=128, lstm_hidden=64, 
                 cat_embed_dim=10, num_classes=8):
        super(HybridIDSModel, self).__init__()
        
        self.text_features = text_features
        self.cat_features = cat_features
        
        # Text Embeddings dan LSTM
        self.text_embeddings = nn.ModuleDict()
        self.text_lstms = nn.ModuleDict()
        
        for feature in text_features:
            self.text_embeddings[feature] = nn.Embedding(
                vocab_size, embed_dim, padding_idx=0
            )
            self.text_lstms[feature] = nn.LSTM(
                embed_dim, lstm_hidden, batch_first=True, dropout=0.2
            )
        
        # Categorical Embeddings
        self.cat_embeddings = nn.ModuleDict()
        for feature, n_unique in cat_features.items():
            self.cat_embeddings[feature] = nn.Embedding(n_unique, cat_embed_dim)
        
        # Calculate total input dimension
        total_dim = (
            len(text_features) * lstm_hidden +
            len(cat_features) * cat_embed_dim +
            num_features_dim
        )
        
        # Classification layers
        self.fc1 = nn.Linear(total_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, text_dict, cat_dict, num_features):
        """
        Forward pass
        
        Args:
            text_dict: Dict {feature_name: LongTensor (batch, seq_len)}
            cat_dict: Dict {feature_name: LongTensor (batch, 1)}
            num_features: FloatTensor (batch, num_features_dim)
        """
        # Process text features
        text_outputs = []
        for feature in self.text_features:
            embedded = self.text_embeddings[feature](text_dict[feature])
            lstm_out, (hidden, _) = self.text_lstms[feature](embedded)
            text_outputs.append(hidden[-1])
        
        # Process categorical features
        cat_outputs = []
        for feature in self.cat_features:
            embedded = self.cat_embeddings[feature](cat_dict[feature])
            cat_outputs.append(embedded.squeeze(1))
        
        # Reshape numerical features if needed
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