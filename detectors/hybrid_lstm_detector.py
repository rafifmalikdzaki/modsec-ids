"""
Hybrid LSTM Inference Engine
Compatible dengan api_log_producer.py dan existing infrastructure
"""

import torch
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import sys
import warnings

from .hybrid_ids_model import HybridIDSModel, TextVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# ⭐ FIX: Custom unpickler untuk handle module path changes
class RenameUnpickler(pickle.Unpickler):
    """
    Custom unpickler untuk redirect old module paths ke new paths
    Mengatasi error 'No module named src'
    """
    def find_class(self, module, name):
        # Redirect old module paths
        if module == 'src.model_architecture':
            module = 'detectors.hybrid_ids_model'
        elif module.startswith('src.'):
            module = module.replace('src.', 'detectors.')
        
        return super().find_class(module, name)


def renamed_load(file_obj):
    """Helper function to load pickle with module renaming"""
    return RenameUnpickler(file_obj).load()


class HybridLSTMDetector:
    """
    Inference engine compatible dengan existing api_log_producer.py
    Implements predict() method dengan output format yang sama
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Load model artifacts
        
        Args:
            models_dir: Path ke folder models/ berisi artifacts
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
        # ⭐ FIX: Use renamed_load untuk vectorizers
        vectorizer_path = self.models_dir / 'vectorizers.pkl'
        with open(vectorizer_path, 'rb') as f:
            self.vectorizers = renamed_load(f)
        
        # Categorical encoders
        cat_encoder_path = self.models_dir / 'categorical_encoders.pkl'
        with open(cat_encoder_path, 'rb') as f:
            self.categorical_encoders = renamed_load(f)
        
        # ⭐ NEW: Build mapping untuk categorical values
        self.categorical_mappings = {}
        for feature_name, encoder in self.categorical_encoders.items():
            # Get all valid classes dari encoder
            valid_classes = list(encoder.classes_)
            self.categorical_mappings[feature_name] = {
                'valid_values': valid_classes,
                'default_value': valid_classes[0] if valid_classes else None
            }
        
        # Label encoder
        label_encoder_path = self.models_dir / 'label_encoder.pkl'
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.label_classes = self.label_encoder.classes_.tolist()
        
        logger.info(f"   ✓ Preprocessors loaded")
        logger.info(f"      - Vectorizers: {list(self.vectorizers.keys())}")
        logger.info(f"      - Categorical features: {list(self.categorical_encoders.keys())}")
    
    def _load_model(self):
        """Load trained model weights"""
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
        
        model_path = self.models_dir / 'best_ids_model.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                f"Please train model first"
            )
        
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"   ✓ Model weights loaded")
    
    def _parse_log_line(self, log_line: str) -> Dict:
        """
        Parse raw log line ke structured data
        Compatible dengan ModSecurity log format
        
        Args:
            log_line: Raw log string
            
        Returns:
            Dict dengan fields yang diperlukan
        """
        # ⭐ FIX: Use valid default values dari categorical mappings
        default_method = self.categorical_mappings.get('request_line_method', {}).get('default_value', 'GET')
        default_action = self.categorical_mappings.get('action', {}).get('default_value', 'pass')
        default_msg_type = self.categorical_mappings.get('message_type', {}).get('default_value', 'info')
        
        # Default values
        parsed = {
            'request_line_url': log_line[:200],
            'request_line_method': default_method,
            'request_useragent': 'Unknown',
            'full_message_line': log_line,
            'action': default_action,
            'message_type': default_msg_type,
            'response_status': 200
        }
        
        # Try to extract basic info from log
        parts = log_line.split()
        if len(parts) >= 2:
            potential_method = parts[0].upper()
            # Validate method adalah salah satu yang valid
            valid_methods = self.categorical_mappings.get('request_line_method', {}).get('valid_values', [])
            if potential_method in valid_methods:
                parsed['request_line_method'] = potential_method
                parsed['request_line_url'] = parts[1]
        
        return parsed
    
    def preprocess_log_entry(self, log_entry: Dict) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Preprocess log entry untuk inference
        
        Args:
            log_entry: Dict dengan keys dari ModSec log
            
        Returns:
            (text_dict, cat_dict, num_features)
        """
        # Extract dengan default values yang valid
        url = log_entry.get('request_line_url', '/')
        method = log_entry.get('request_line_method', 
                              self.categorical_mappings.get('request_line_method', {}).get('default_value', 'GET'))
        user_agent = log_entry.get('request_useragent', 'Unknown')
        message = log_entry.get('full_message_line', '')
        action = log_entry.get('action', 
                              self.categorical_mappings.get('action', {}).get('default_value', 'pass'))
        message_type = log_entry.get('message_type', 
                                    self.categorical_mappings.get('message_type', {}).get('default_value', 'info'))
        response_status = int(log_entry.get('response_status', 200))
        
        # Vectorize text features
        text_dict = {}
        text_inputs = {
            'request_useragent': user_agent,
            'request_line_url': url,
            'full_message_line': message
        }
        
        for feature_name, text in text_inputs.items():
            vectorizer = self.vectorizers[feature_name]
            sequence = vectorizer.transform([text])[0]
            text_dict[feature_name] = sequence
        
        # ⭐ FIX: Encode categorical features dengan validation
        cat_dict = {}
        cat_inputs = {
            'request_line_method': method,
            'action': action,
            'message_type': message_type
        }
        
        for feature_name, value in cat_inputs.items():
            encoder = self.categorical_encoders[feature_name]
            
            # Validate value ada di valid classes
            valid_values = self.categorical_mappings[feature_name]['valid_values']
            
            if value not in valid_values:
                # Use default value instead
                default_val = self.categorical_mappings[feature_name]['default_value']
                logger.debug(f"Value '{value}' not in training data for {feature_name}, using '{default_val}'")
                value = default_val
            
            encoded = encoder.transform([value])[0]
            cat_dict[feature_name] = encoded
        
        # ⭐ FIX: Scale numerical features without feature names
        num_features = self.scaler.transform([[response_status]])[0]
        
        return text_dict, cat_dict, num_features
    
    def predict(self, log_line: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Main prediction method - COMPATIBLE dengan api_log_producer.py
        
        Args:
            log_line: Raw log string atau structured dict
            
        Returns:
            (predicted_label, confidence, probabilities_dict)
        """
        # Parse input
        if isinstance(log_line, str):
            log_entry = self._parse_log_line(log_line)
        elif isinstance(log_line, dict):
            log_entry = log_line
        else:
            raise ValueError("Input must be string or dict")
        
        # Preprocess
        text_dict, cat_dict, num_features = self.preprocess_log_entry(log_entry)
        
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
            
            # All probabilities
            all_probs = {
                self.label_classes[i]: float(probs[i].item())
                for i in range(len(self.label_classes))
            }
        
        return predicted_label, confidence, all_probs
    
    def get_supported_values(self) -> Dict[str, List[str]]:
        """
        Get list of supported categorical values
        Useful untuk debugging dan validation
        
        Returns:
            Dict mapping feature name to list of valid values
        """
        return {
            feature_name: mapping['valid_values']
            for feature_name, mapping in self.categorical_mappings.items()
        }