"""
Real-time inference engine untuk Hybrid LSTM IDS Model
Compatible dengan ModSecurity log format
"""

import torch
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from .hybrid_ids_model import HybridIDSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridLSTMDetector:
    """
    Production-ready inference engine untuk multiclass attack detection
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Load semua artifacts untuk inference
        
        Args:
            models_dir: Path ke folder models/ yang berisi artifacts
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🔧 Initializing HybridLSTMDetector...")
        logger.info(f"   Device: {self.device}")
        
        # Load artifacts
        self._load_config()
        self._load_preprocessors()
        self._load_model()
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Classes: {self.label_classes}")
    
    def _load_config(self):
        """Load model configuration"""
        config_path = self.models_dir / 'model_config.json'
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please ensure model artifacts are in {self.models_dir}/"
            )
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"   ✓ Config loaded")
    
    def _load_preprocessors(self):
        """Load vectorizers, encoders, dan scaler"""
        # Vectorizers untuk text features
        vectorizer_path = self.models_dir / 'vectorizers.pkl'
        with open(vectorizer_path, 'rb') as f:
            self.vectorizers = pickle.load(f)
        
        # Categorical encoders
        cat_encoder_path = self.models_dir / 'categorical_encoders.pkl'
        with open(cat_encoder_path, 'rb') as f:
            self.categorical_encoders = pickle.load(f)
        
        # Label encoder
        label_encoder_path = self.models_dir / 'label_encoder.pkl'
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Scaler untuk numerical features
        scaler_path = self.models_dir / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.label_classes = self.label_encoder.classes_.tolist()
        
        logger.info(f"   ✓ Preprocessors loaded")
        logger.info(f"      - Vectorizers: {list(self.vectorizers.keys())}")
        logger.info(f"      - Categorical features: {list(self.categorical_encoders.keys())}")
        logger.info(f"      - Classes: {self.label_classes}")
    
    def _load_model(self):
        """Load trained model weights"""
        # Initialize model architecture
        self.model = HybridIDSModel(
            text_features=self.config['text_features'],
            cat_features=self.config['cat_features_config'],
            num_features_dim=len(self.config['numerical_features']),
            vocab_size=self.config['vocab_size'],
            embed_dim=self.config['embed_dim'],
            lstm_hidden=self.config['lstm_hidden'],
            cat_embed_dim=self.config['cat_embed_dim'],
            num_classes=self.config['num_classes']
        )
        
        # Load weights
        model_path = self.models_dir / 'best_ids_model.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                f"Please train model first using train_pytorch_hybrid_lstm.py"
            )
        
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"   ✓ Model weights loaded from {model_path.name}")
    
    def preprocess_modsec_log(self, log_entry: Dict) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Preprocess ModSecurity log entry untuk inference
        
        Args:
            log_entry: Dict dengan format ModSecurity log
                {
                    'request_line_url': str,
                    'request_line_method': str,
                    'request_useragent': str,
                    'full_message_line': str,
                    'action': str,
                    'message_type': str,
                    'response_status': int
                }
        
        Returns:
            (text_dict, cat_dict, num_features)
        """
        # 1. Extract dan default values
        url = log_entry.get('request_line_url', '/')
        method = log_entry.get('request_line_method', 'GET')
        user_agent = log_entry.get('request_useragent', 'Unknown')
        message = log_entry.get('full_message_line', '')
        action = log_entry.get('action', 'pass')
        message_type = log_entry.get('message_type', 'info')
        response_status = int(log_entry.get('response_status', 200))
        
        # 2. Vectorize text features
        text_dict = {}
        text_inputs = {
            'request_useragent': user_agent,
            'request_line_url': url,
            'full_message_line': message
        }
        
        for feature_name, text in text_inputs.items():
            vectorizer = self.vectorizers[feature_name]
            sequence = vectorizer.transform([text])[0]  # Returns list of sequences
            text_dict[feature_name] = sequence
        
        # 3. Encode categorical features
        cat_dict = {}
        cat_inputs = {
            'request_line_method': method,
            'action': action,
            'message_type': message_type
        }
        
        for feature_name, value in cat_inputs.items():
            encoder = self.categorical_encoders[feature_name]
            try:
                # Handle unknown values
                encoded = encoder.transform([value])[0]
            except ValueError:
                # If value not in training set, use 0 (or last class)
                logger.warning(f"Unknown value '{value}' for {feature_name}, using default")
                encoded = 0
            cat_dict[feature_name] = encoded
        
        # 4. Scale numerical features
        num_features = self.scaler.transform([[response_status]])[0]
        
        return text_dict, cat_dict, num_features
    
    def predict(self, log_entry: Dict, return_probs: bool = True) -> Dict:
        """
        Predict attack type dari ModSecurity log entry
        
        Args:
            log_entry: Dict dengan format ModSec log
            return_probs: Include all class probabilities in output
        
        Returns:
            {
                'predicted_label': str,
                'predicted_index': int,
                'confidence': float,
                'is_attack': bool,
                'probabilities': Dict[str, float],  # if return_probs=True
                'input_summary': Dict
            }
        """
        # Preprocess
        text_dict, cat_dict, num_features = self.preprocess_modsec_log(log_entry)
        
        # Convert to tensors
        text_tensors = {
            name: torch.LongTensor([seq]).to(self.device)
            for name, seq in text_dict.items()
        }
        
        cat_tensors = {
            name: torch.LongTensor([val]).to(self.device)
            for name, val in cat_dict.items()
        }
        
        num_tensor = torch.FloatTensor([num_features]).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(text_tensors, cat_tensors, num_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            predicted_label = self.label_classes[pred_idx]
        
        # Build result
        result = {
            'predicted_label': predicted_label,
            'predicted_index': pred_idx,
            'confidence': confidence,
            'is_attack': predicted_label.lower() not in ['normal', 'clean', 'benign'],
            'input_summary': {
                'url': log_entry.get('request_line_url', '/')[:100],
                'method': log_entry.get('request_line_method', 'GET'),
                'status': log_entry.get('response_status', 200)
            }
        }
        
        # Add all probabilities if requested
        if return_probs:
            result['probabilities'] = {
                self.label_classes[i]: float(probs[i].item())
                for i in range(len(self.label_classes))
            }
        
        return result
    
    def batch_predict(self, log_entries: List[Dict]) -> List[Dict]:
        """
        Batch prediction untuk multiple log entries
        
        Args:
            log_entries: List of ModSec log dicts
        
        Returns:
            List of prediction results
        """
        results = []
        for log in log_entries:
            try:
                result = self.predict(log)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing log: {e}")
                results.append({
                    'error': str(e),
                    'input': log
                })
        
        return results
    
    def get_top_k_predictions(self, log_entry: Dict, k: int = 3) -> List[Dict]:
        """
        Get top-k predictions dengan probabilities
        
        Args:
            log_entry: ModSec log dict
            k: Number of top predictions to return
        
        Returns:
            List of {label, probability} dicts, sorted by probability
        """
        result = self.predict(log_entry, return_probs=True)
        
        # Sort probabilities
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'label': label, 'probability': prob}
            for label, prob in sorted_probs[:k]
        ]