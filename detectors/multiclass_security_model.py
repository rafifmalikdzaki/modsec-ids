import torch
import torch.nn as nn
import joblib
from pathlib import Path

# Multi-class Model Configuration
MULTICLASS_MODEL_PATH = "models/multiclass_lstm_classifier.pth"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

class MultiClassLSTM(nn.Module):
    """Multi-class LSTM model for attack classification."""
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

class MultiClassAttackClassifier:
    """Multi-class attack classifier that handles 8 attack types."""

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        """Load the multi-class model and label encoder."""
        try:
            # Load label encoder
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)

            # Load model checkpoint with safety settings
            checkpoint = torch.load(
                MULTICLASS_MODEL_PATH,
                map_location='cpu',
                weights_only=False  # Required for PyTorch 2.6+
            )

            # Extract model parameters
            input_size = checkpoint['input_size']
            hidden_size = checkpoint['hidden_size']
            output_size = checkpoint['output_size']

            # Create model and load state dict
            self.model = MultiClassLSTM(input_size, hidden_size, output_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode

            self.is_loaded = True
            print(f"✅ Loaded multi-class model: {input_size} features, {output_size} classes")
            print(f"📋 Classes: {list(self.label_encoder.classes_)}")

        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            print("   Run 'python train_enhanced_ids.py --preprocess --train' first")
            self.is_loaded = False
        except Exception as e:
            print(f"❌ Error loading multi-class model: {e}")
            self.is_loaded = False

    def predict(self, features):
        """
        Make prediction on features.

        Args:
            features: List or numpy array of features (should be 15-dimensional)

        Returns:
            dict: {
                'predicted_class': str,  # e.g., 'sqli', 'xss', 'normal'
                'class_index': int,      # 0-7
                'confidence': float,      # 0.0-1.0
                'all_probabilities': dict,  # {class: prob} for all classes
                'is_attack': bool         # True if not 'normal'
            }
        """
        if not self.is_loaded:
            # Fallback to binary prediction if model not loaded
            return {
                'predicted_class': 'unknown',
                'class_index': -1,
                'confidence': 0.0,
                'all_probabilities': {},
                'is_attack': False
            }

        try:
            # Convert features to tensor
            if isinstance(features, list):
                features = torch.FloatTensor([features])  # Add batch dimension
            else:
                features = torch.FloatTensor([features.tolist()])

            # Make prediction
            with torch.no_grad():
                output = self.model(features)
                probabilities = torch.softmax(output, dim=1)

                # Get prediction
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()

                # Get class name
                predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]

                # Create probabilities dictionary
                all_probs = {}
                for i, class_name in enumerate(self.label_encoder.classes_):
                    all_probs[class_name] = probabilities[0][i].item()

                # Determine if it's an attack
                is_attack = predicted_class != 'normal'

                return {
                    'predicted_class': predicted_class,
                    'class_index': predicted_idx,
                    'confidence': confidence,
                    'all_probabilities': all_probs,
                    'is_attack': is_attack
                }

        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return {
                'predicted_class': 'error',
                'class_index': -1,
                'confidence': 0.0,
                'all_probabilities': {},
                'is_attack': False
            }

    def get_attack_colors(self):
        """Return color mappings for different attack types."""
        return {
            'normal': 'green',
            'sqli': 'red',
            'xss': 'orange',
            'lfi': 'yellow',
            'rfi': 'purple',
            'rce': 'bright_red',
            'path_traversal': 'magenta',
            'bruteforce': 'red'
        }

    def get_attack_descriptions(self):
        """Return descriptions for different attack types."""
        return {
            'normal': 'Normal traffic',
            'sqli': 'SQL Injection',
            'xss': 'Cross-Site Scripting',
            'lfi': 'Local File Inclusion',
            'rfi': 'Remote File Inclusion',
            'rce': 'Remote Code Execution',
            'path_traversal': 'Directory Traversal',
            'bruteforce': 'Brute Force Attack'
        }