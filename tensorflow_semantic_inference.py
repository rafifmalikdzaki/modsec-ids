import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TensorFlowSemanticInference:
    def __init__(self, model_dir='results'):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_seq_length = 100  # Must match training
        self.classes = ['normal', 'sqli', 'bruteforce', 'lfi', 'xss', 'rce', 'directory_traversal', 'command_injection']
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, tokenizer, and label encoder."""
        try:
            # Load Model
            model_path = os.path.join(self.model_dir, 'final_model.keras')
            if not os.path.exists(model_path):
                logging.error(f"Model not found at {model_path}")
                return

            self.model = load_model(model_path)
            logging.info(f"✅ Model loaded from {model_path}")

            # Load Tokenizer
            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logging.info(f"✅ Tokenizer loaded from {tokenizer_path}")
            else:
                logging.error(f"Tokenizer not found at {tokenizer_path}")

            # Load Label Encoder
            encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logging.info(f"✅ Label encoder loaded from {encoder_path}")
            else:
                logging.warning(f"Label encoder not found. Using default classes.")

        except Exception as e:
            logging.error(f"Error loading artifacts: {e}")

    def predict(self, text):
        """
        Predict the class of a given log line/text.
        Returns: (predicted_class_name, confidence_score, all_probabilities)
        """
        if not self.model or not self.tokenizer:
            return "error", 0.0, {}

        try:
            # Preprocess
            sequences = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post', truncating='post')

            # Inference
            preds = self.model.predict(padded, verbose=0)[0]
            
            # Get result
            class_idx = np.argmax(preds)
            confidence = float(preds[class_idx])
            
            if self.label_encoder:
                predicted_label = self.label_encoder.inverse_transform([class_idx])[0]
            else:
                predicted_label = self.classes[class_idx] if class_idx < len(self.classes) else "unknown"

            # Format probabilities
            probs = {cls: float(preds[i]) for i, cls in enumerate(self.classes) if i < len(preds)}

            return predicted_label, confidence, probs

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return "error", 0.0, {}

if __name__ == "__main__":
    # Simple test
    detector = TensorFlowSemanticInference()
    test_log = "GET /wp-admin/admin-ajax.php?action=revslider_show_image&img=../wp-config.php HTTP/1.1"
    label, conf, _ = detector.predict(test_log)
    print(f"Test Log: {test_log}")
    print(f"Prediction: {label} ({conf:.2%})")
