import torch
import torch.nn as nn
import numpy as np
import re
import urllib.parse
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

class EnhancedAttackClassifier(nn.Module):
    """Enhanced LSTM-based attack classifier with preloaded weights."""

    def __init__(self,
                 input_features=15,
                 hidden_dim=64,
                 num_layers=2,
                 output_dim=2,
                 model_path: Optional[str] = None):
        super(EnhancedAttackClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_features = input_features

        # LSTM layers with bidirectional processing
        self.lstm = nn.LSTM(input_features, hidden_dim, num_layers,
                           batch_first=True, dropout=0.3, bidirectional=True)

        # Attention mechanism for better feature importance
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            self.eval()
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_features)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)

        # Apply self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_out shape: (batch_size, seq_len, hidden_dim * 2)

        # Global max pooling over sequence dimension
        pooled = torch.max(attn_out, dim=1)[0]
        # pooled shape: (batch_size, hidden_dim * 2)

        # Batch normalization
        pooled = self.batch_norm(pooled)

        # Fully connected layers
        x = self.fc1(pooled)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {model_path}")
                print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
            else:
                self.load_state_dict(checkpoint)
                print(f"Model weights loaded from {model_path}")

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Using randomly initialized weights")

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[bool, float]:
        """Make prediction with confidence score."""
        self.eval()

        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension

            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims

            # Ensure sequence length dimension
            if x.size(1) != 50:  # Expected sequence length
                # Pad or truncate to sequence length 50
                if x.size(1) < 50:
                    padding = torch.zeros(x.size(0), 50 - x.size(1), x.size(2))
                    x = torch.cat([padding, x], dim=1)
                else:
                    x = x[:, :50, :]

            outputs = self(x)
            probabilities = torch.softmax(outputs, dim=1)

            # Get attack probability and prediction
            attack_prob = probabilities[0, 1].item()
            is_attack = attack_prob > threshold

            return is_attack, attack_prob

class EnhancedFeatureExtractor:
    """Enhanced feature extractor for both raw logs and ModSecurity format."""

    def __init__(self, preprocessor_path: Optional[str] = None):
        self.preprocessor = None
        self.fallback_mode = True

        # Try to load preprocessor
        if preprocessor_path and Path(preprocessor_path).exists():
            try:
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                self.fallback_mode = False
                print(f"Loaded preprocessor from {preprocessor_path}")
            except Exception as e:
                print(f"Error loading preprocessor: {e}, using fallback mode")
        else:
            print("No preprocessor found, using fallback feature extraction")

    def extract_from_modsecurity_row(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from ModSecurity dataset row."""
        if not self.fallback_mode and self.preprocessor:
            try:
                import pandas as pd
                df_row = pd.Series([row])
                features = self.preprocessor.extract_features(df_row)
                return features
            except Exception as e:
                print(f"Error using preprocessor: {e}, falling back")

        # Fallback feature extraction
        return self._extract_features_fallback(row)

    def _extract_features_fallback(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        """Fallback feature extraction method."""
        try:
            # URL-based features
            url = str(row.get('request_line_url', '')).lower()
            decoded_url = urllib.parse.unquote(url)

            # User-agent features
            user_agent = str(row.get('request_useragent', '')).lower()

            # HTTP features
            method = str(row.get('request_line_method', 'GET')).upper()
            status_code = str(row.get('response_status', '200'))

            try:
                status_int = int(float(status_code))
            except (ValueError, TypeError):
                status_int = 200

            # Action features
            action = str(row.get('action', '')).lower()
            action_message = str(row.get('action_message', '')).lower()

            # Create 15-dimensional feature vector
            features = np.array([
                # URL features
                float(len(url)) / 200.0,  # URL length (normalized)
                float(len(decoded_url)) / 200.0,  # Decoded URL length
                1.0 if 'wp-admin' in decoded_url or 'wp-login' in decoded_url else 0.0,  # Admin area
                1.0 if 'admin-ajax' in decoded_url or 'ajax' in decoded_url else 0.0,  # Ajax endpoint
                1.0 if any(char in decoded_url for char in ['<', '>', '\'', '"', '&']) else 0.0,  # XSS chars
                1.0 if 'base64' in decoded_url or 'bm' in decoded_url else 0.0,  # Base64 encoding
                1.0 if any(kw in decoded_url for kw in ['union', 'select', 'insert', 'delete']) else 0.0,  # SQL keywords
                1.0 if any(kw in decoded_url for kw in ['exec', 'system', 'shell', 'cmd']) else 0.0,  # RCE keywords

                # HTTP features
                1.0 if method == 'POST' else 0.0,  # POST method
                float(status_int) / 600.0,  # Status code (normalized)
                1.0 if status_int >= 400 else 0.0,  # Error status

                # User-agent features
                1.0 if any(bot in user_agent for bot in ['bot', 'crawler', 'spider']) else 0.0,  # Bot detection
                float(len(user_agent)) / 500.0,  # User-agent length

                # Attack pattern features
                1.0 if 'blocked' in action or 'deny' in action else 0.0,  # Was blocked
                1.0 if any(pattern in action_message for pattern in ['sqli', 'xss', 'rce', 'lfi']) else 0.0,  # Attack pattern
            ], dtype=np.float32)

            return features

        except Exception as e:
            print(f"Error in fallback feature extraction: {e}")
            return None

    def extract_from_raw_log(self, log_line: str) -> Optional[np.ndarray]:
        """Extract features from raw log line (for compatibility)."""
        # This is the original feature extraction from security_model.py
        LOG_PATTERN = re.compile(
            r'(?P<ip>[\d\.]+) - - \[(?P<timestamp>.*?)\] "(?P<method>\w+) (?P<uri>.*?) (?P<protocol>HTTP\/[\d\.]+)" (?P<status>\d+) (?P<size>\d+)'
        )

        match = LOG_PATTERN.match(log_line)
        if not match:
            return None

        data = match.groupdict()
        decoded_uri = urllib.parse.unquote(data['uri']).lower()

        # Create a row dict similar to ModSecurity format
        row = {
            'request_line_url': data['uri'],
            'request_line_method': data['method'],
            'response_status': data['status'],
            'request_useragent': '-',  # Not available in raw log
            'action': '-',  # Not available in raw log
            'action_message': '-',  # Not available in raw log
        }

        return self._extract_features_fallback(row)

# Model singleton for global access
_model_instance = None
_feature_extractor = None

def get_enhanced_model(model_path: str = "models/lstm_attack_classifier.pth") -> EnhancedAttackClassifier:
    """Get or create enhanced model instance."""
    global _model_instance

    if _model_instance is None:
        _model_instance = EnhancedAttackClassifier(model_path=model_path)

    return _model_instance

def get_enhanced_feature_extractor(preprocessor_path: str = "data/processed/modsec_processed_preprocessor.pkl") -> EnhancedFeatureExtractor:
    """Get or create enhanced feature extractor instance."""
    global _feature_extractor

    if _feature_extractor is None:
        _feature_extractor = EnhancedFeatureExtractor(preprocessor_path)

    return _feature_extractor

def predict_attack(features: np.ndarray,
                   model_path: str = "models/lstm_attack_classifier.pth",
                   threshold: float = 0.5) -> Tuple[bool, float]:
    """Convenience function to make attack prediction."""
    model = get_enhanced_model(model_path)
    features_tensor = torch.FloatTensor(features)

    is_attack, confidence = model.predict(features_tensor, threshold)
    return is_attack, confidence

if __name__ == "__main__":
    # Test the enhanced model
    print("Testing Enhanced Security Model")

    # Create dummy features for testing
    dummy_features = np.random.rand(15).astype(np.float32)

    try:
        is_attack, confidence = predict_attack(dummy_features)
        print(f"Test prediction: Attack={is_attack}, Confidence={confidence:.4f}")
    except Exception as e:
        print(f"Error in test prediction: {e}")
        print("Model will be available after training with python lstm_security_model.py")