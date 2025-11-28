"""
IDS Model Inference Module
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Union
from pathlib import Path

from .model_architecture import HybridIDSModel, TextVectorizer


class IDSInference:
    """Main inference class for IDS model"""
    
    def __init__(self, model_path='models/best_ids_model.pth', 
                 artifacts_dir='models', device=None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to .pth model weights
            artifacts_dir: Directory containing preprocessing artifacts
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Initializing IDS Inference Engine")
        print(f"   Device: {self.device}")
        print(f"   Model path: {self.model_path}")
        print(f"   Artifacts dir: {self.artifacts_dir}")
        
        # Load all artifacts
        self._load_artifacts()
        
        # Load model
        self._load_model()
        
        print(f"✅ Inference engine ready!\n")
    
    def _load_artifacts(self):
        """Load all preprocessing artifacts"""
        print("📦 Loading preprocessing artifacts...")
        
        # 1. Load vectorizers
        with open(self.artifacts_dir / 'vectorizers.pkl', 'rb') as f:
            self.vectorizers = pickle.load(f)
        print("   ✓ Vectorizers loaded")
        
        # 2. Load categorical encoders
        with open(self.artifacts_dir / 'categorical_encoders.pkl', 'rb') as f:
            self.categorical_encoders = pickle.load(f)
        print("   ✓ Categorical encoders loaded")
        
        # 3. Load label encoder
        with open(self.artifacts_dir / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("   ✓ Label encoder loaded")
        
        # 4. Load scaler
        with open(self.artifacts_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print("   ✓ Scaler loaded")
        
        # 5. Load model config
        with open(self.artifacts_dir / 'model_config.json', 'r') as f:
            self.model_config = json.load(f)
        print("   ✓ Model config loaded")
        
        # 6. Load feature info
        with open(self.artifacts_dir / 'feature_info.json', 'r') as f:
            self.feature_info = json.load(f)
        print("   ✓ Feature info loaded")
    
    def _load_model(self):
        """Load trained model"""
        print(f"🧠 Loading model...")
        
        # Initialize model with same architecture
        self.model = HybridIDSModel(
            text_features=self.model_config['text_features'],
            cat_features=self.model_config['cat_features_config'],
            num_features_dim=len(self.model_config['numerical_features']),
            vocab_size=self.model_config['vocab_size'],
            embed_dim=self.model_config['embed_dim'],
            lstm_hidden=self.model_config['lstm_hidden'],
            cat_embed_dim=self.model_config['cat_embed_dim'],
            num_classes=self.model_config['num_classes']
        )
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        print("   ✓ Model loaded and ready")
    
    def preprocess_input(self, data: Union[Dict, pd.DataFrame]) -> Dict:
        """Preprocess input data"""
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Handle missing values (same as training)
        df = df.replace('-', np.nan)
        for col in ['request_useragent', 'request_line_url', 'full_message_line']:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # 1. Process text features
        text_data = {}
        for feature in self.feature_info['text_features']:
            sequences = self.vectorizers[feature].transform(df[feature].astype(str))
            text_data[feature] = torch.LongTensor(sequences).to(self.device)
        
        # 2. Process categorical features
        cat_data = {}
        for feature in self.feature_info['categorical_features']:
            values = df[feature].fillna('unknown')
            
            encoded = []
            for val in values:
                if val in self.categorical_encoders[feature].classes_:
                    encoded.append(self.categorical_encoders[feature].transform([val])[0])
                else:
                    encoded.append(len(self.categorical_encoders[feature].classes_))
            
            cat_data[feature] = torch.LongTensor(encoded).unsqueeze(1).to(self.device)
        
        # 3. Process numerical features
        num_values = df[self.feature_info['numerical_features']].fillna(0).values
        num_scaled = self.scaler.transform(num_values)
        num_data = torch.FloatTensor(num_scaled).to(self.device)
        
        return {
            'text': text_data,
            'categorical': cat_data,
            'numerical': num_data
        }
    
    def predict(self, data: Union[Dict, pd.DataFrame], 
                return_proba: bool = False) -> Union[List[str], List[Dict]]:
        """Make prediction on new data"""
        processed = self.preprocess_input(data)
        
        with torch.no_grad():
            outputs = self.model(
                processed['text'],
                processed['categorical'],
                processed['numerical']
            )
            
            probs = torch.softmax(outputs, dim=1)
            predicted_indices = outputs.argmax(dim=1).cpu().numpy()
            predicted_labels = self.label_encoder.inverse_transform(predicted_indices)
        
        if return_proba:
            probs_np = probs.cpu().numpy()
            result = []
            for i, label in enumerate(predicted_labels):
                result.append({
                    'prediction': label,
                    'confidence': float(probs_np[i, predicted_indices[i]]),
                    'probabilities': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.label_encoder.classes_, probs_np[i])
                    }
                })
            return result
        else:
            return list(predicted_labels)
    
    def predict_single(self, request_useragent: str, request_line_url: str,
                      full_message_line: str, request_line_method: str,
                      action: str, message_type: str, 
                      response_status: int) -> Dict:
        """Predict single request"""
        data = {
            'request_useragent': request_useragent,
            'request_line_url': request_line_url,
            'full_message_line': full_message_line,
            'request_line_method': request_line_method,
            'action': action,
            'message_type': message_type,
            'response_status': response_status
        }
        
        result = self.predict(data, return_proba=True)[0]
        return result
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'device': str(self.device),
            'num_classes': self.model_config['num_classes'],
            'classes': self.feature_info['label_classes'],
            'text_features': self.feature_info['text_features'],
            'categorical_features': self.feature_info['categorical_features'],
            'numerical_features': self.feature_info['numerical_features'],
            'vocab_size': self.model_config['vocab_size'],
            'max_length': self.model_config['max_len']
        }


def load_inference_engine(model_path='models/best_ids_model.pth', 
                          artifacts_dir='models') -> IDSInference:
    """Quick helper to load inference engine"""
    return IDSInference(model_path, artifacts_dir)


def batch_predict_from_csv(csv_path: str, model_path='models/best_ids_model.pth',
                           artifacts_dir='models', output_path=None) -> pd.DataFrame:
    """Predict from CSV file"""
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded {len(df)} samples from {csv_path}")
    
    engine = IDSInference(model_path, artifacts_dir)
    
    print("🔮 Making predictions...")
    results = engine.predict(df, return_proba=True)
    
    df['prediction'] = [r['prediction'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to {output_path}")
    
    return df