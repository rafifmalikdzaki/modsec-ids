import os
import pickle
import numpy as np
import urllib.parse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import ByteLevelBPETokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TensorFlowSemanticInference:
    def __init__(self, model_dir='results'):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_seq_length = 500  # Increased to match training
        self.classes = []
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, tokenizer, and label encoder."""
        try:
            # Load Model
            model_path = os.path.join(self.model_dir, 'best_model.keras')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.model_dir, 'final_model.keras')
            
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logging.info(f"✅ Model loaded from {model_path}")
            else:
                logging.error(f"Model not found at {model_path}")

            # Load BPE Tokenizer
            vocab_path = os.path.join(self.model_dir, 'vocab.json')
            merges_path = os.path.join(self.model_dir, 'merges.txt')
            
            if os.path.exists(vocab_path) and os.path.exists(merges_path):
                self.tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
                logging.info(f"✅ BPE Tokenizer loaded")
            else:
                logging.error(f"BPE Tokenizer files not found in {self.model_dir}")

            # Load Label Encoder
            encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                if hasattr(self.label_encoder, 'classes_'):
                    self.classes = self.label_encoder.classes_.tolist()
                logging.info(f"✅ Label encoder loaded")
            else:
                logging.warning(f"Label encoder not found.")

        except Exception as e:
            logging.error(f"Error loading artifacts: {e}")

    def _preprocess_text(self, text):
        """Preprocessing with Feature Injection (Must match training)."""
        if not isinstance(text, str):
            return ""
        
        # 1. URL Decode
        try:
            text = urllib.parse.unquote(text)
        except Exception:
            pass
            
        # 2. Lowercase
        text = text.lower().strip()
        
        # 3. Feature Injection (Heuristic hints)
        flags = []
        
        # RFI: http/https in parameters
        if "http://" in text or "https://" in text or "ftp://" in text:
            flags.append("[FLAG_RFI]")
            
        # Traversal / LFI
        if "../" in text or "..\\" in text or "/etc/passwd" in text or "win.ini" in text:
            flags.append("[FLAG_TRAVERSAL]")
            
        # XSS
        if "<script" in text or "javascript:" in text or "onerror=" in text or "onload=" in text:
            flags.append("[FLAG_XSS]")
            
        # SQLi
        if "union select" in text or " or 1=1" in text or "'--" in text or "information_schema" in text:
            flags.append("[FLAG_SQLI]")
            
        # RCE
        if "; cat" in text or "| ls" in text or "$(whoami)" in text or "; system" in text:
            flags.append("[FLAG_RCE]")

        # Append flags to text
        if flags:
            text = " ".join(flags) + " " + text
            
        return text

    def predict(self, text):
        if not self.model or not self.tokenizer:
            return "error", 0.0, {}

        try:
            clean_text = self._preprocess_text(text)
            
            # BPE Encoding
            encoded = self.tokenizer.encode(clean_text)
            sequence = encoded.ids
            
            # Padding
            padded = pad_sequences([sequence], maxlen=self.max_seq_length, padding='post', truncating='post')

            # Inference
            preds = self.model.predict(padded, verbose=0)[0]
            
            class_idx = np.argmax(preds)
            confidence = float(preds[class_idx])
            
            predicted_label = "unknown"
            if class_idx < len(self.classes):
                predicted_label = self.classes[class_idx]

            probs = {cls: float(preds[i]) for i, cls in enumerate(self.classes) if i < len(preds)}

            return predicted_label, confidence, probs

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return "error", 0.0, {}
