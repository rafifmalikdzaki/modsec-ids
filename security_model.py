import torch
import torch.nn as nn
import urllib.parse
import re
from pathlib import Path

# Model Configuration
INPUT_DIM = 10  # We extract 10 numerical features from every log line
HIDDEN_DIM = 16
OUTPUT_DIM = 2  # [Probability of Normal, Probability of Attack]

# Enhanced LSTM Model Configuration
LSTM_MODEL_PATH = "models/lstm_attack_classifier.pth"
LSTM_SEMANTIC_MODEL_PATH = "models/lstm_semantic_classifier.pth"
PREPROCESSOR_PATH = "data/processed/modsec_processed_preprocessor.pkl"
SEMANTIC_PREPROCESSOR_PATH = "data/processed/lstm_semantic_preprocessor.pkl"

class AttackClassifier(nn.Module):
    """Attack Classifier with support for simple FFNN, enhanced LSTM, and LSTM semantic models."""
    def __init__(self, use_enhanced=False, use_semantic=False):
        super(AttackClassifier, self).__init__()
        self.use_enhanced = use_enhanced
        self.use_semantic = use_semantic

        # Try to load LSTM semantic model first (highest priority)
        if use_semantic and Path(LSTM_SEMANTIC_MODEL_PATH).exists():
            try:
                from enhanced_security_model import LSTMSemanticAttackClassifier
                self.semantic_model = LSTMSemanticAttackClassifier()
                print("Loaded LSTM semantic model")
                self.use_semantic = True
            except Exception as e:
                print(f"Error loading semantic model: {e}, falling back to enhanced model")
                self.use_semantic = False
                self.semantic_model = None
        else:
            self.semantic_model = None

        # Try to load enhanced LSTM model
        if use_enhanced and Path(LSTM_MODEL_PATH).exists():
            try:
                from enhanced_security_model import EnhancedAttackClassifier
                self.enhanced_model = EnhancedAttackClassifier(model_path=LSTM_MODEL_PATH)
                print("Loaded enhanced LSTM model")
                self.use_enhanced = True
            except Exception as e:
                print(f"Error loading enhanced model: {e}, falling back to simple model")
                self.use_enhanced = False
                self.enhanced_model = None
        else:
            self.enhanced_model = None

        # Initialize simple model as fallback
        if not self.use_enhanced and not self.use_semantic:
            # Simple Feed-Forward Neural Network
            # Layer 1: Input -> Hidden
            self.layer1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
            self.relu = nn.ReLU()
            # Layer 2: Hidden -> Output
            self.layer2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
            self.softmax = nn.Softmax(dim=1)
            self.layer2_output = None

    def forward(self, x):
        if self.use_semantic:
            # Semantic model handles its own forward pass
            return self.semantic_model(x)
        elif self.use_enhanced:
            return self.enhanced_model(x)
        else:
            # Simple FFNN forward pass
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            self.layer2_output = x
            return self.softmax(x)

    def predict(self, features, threshold=0.5):
        """Make prediction with confidence score."""
        if self.use_semantic:
            # Use semantic model for text-based prediction
            # Note: semantic model expects text, not numerical features
            if self.semantic_model and self.semantic_model.preprocessor.is_fitted:
                # Features should contain semantic text data for this mode
                if isinstance(features, str):
                    return self.semantic_model.predict(features, threshold)
                elif isinstance(features, dict) and 'text' in features:
                    return self.semantic_model.predict(features['text'], threshold)
            # Fallback to simple model
            features_tensor = torch.FloatTensor([features]) if len(features.shape) == 1 else features
            with torch.no_grad():
                outputs = self.forward(features_tensor)
                probabilities = outputs.tolist()[0]

                is_attack = probabilities[1] > threshold
                confidence = probabilities[1] if is_attack else probabilities[0]

                return is_attack, confidence
        elif self.use_enhanced:
            features_tensor = torch.FloatTensor(features)
            if features_tensor.dim() == 1:
                features_tensor = features_tensor.unsqueeze(0)
            return self.enhanced_model.predict(features_tensor, threshold)
        else:
            # Use simple FFNN
            features_tensor = torch.FloatTensor([features]) if len(features.shape) == 1 else features
            with torch.no_grad():
                outputs = self.forward(features_tensor)
                probabilities = outputs.tolist()[0]

                is_attack = probabilities[1] > threshold
                confidence = probabilities[1] if is_attack else probabilities[0]

                return is_attack, confidence

class FeatureExtractor:
    """Helper class to parse raw logs and ModSecurity dataset and convert them into Tensors."""

    # Regex to parse standard Combined Log Format
    LOG_PATTERN = re.compile(
        r'(?P<ip>[\d\.]+) - - \[(?P<timestamp>.*?)\] "(?P<method>\w+) (?P<uri>.*?) (?P<protocol>HTTP\/[\d\.]+)" (?P<status>\d+) (?P<size>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
    )

    def __init__(self, use_enhanced=False, use_semantic=False):
        self.use_enhanced = use_enhanced
        self.use_semantic = use_semantic

        # Load semantic feature extractor
        if use_semantic and Path(SEMANTIC_PREPROCESSOR_PATH).exists():
            try:
                from enhanced_security_model import LSTMSemanticAttackClassifier
                self.semantic_model = LSTMSemanticAttackClassifier()
                print("Loaded LSTM semantic model")
            except Exception as e:
                print(f"Error loading semantic model: {e}, using fallback")
                self.semantic_model = None
                self.use_semantic = False
        else:
            self.semantic_model = None

        # Load enhanced feature extractor
        if use_enhanced and Path(PREPROCESSOR_PATH).exists():
            try:
                from enhanced_security_model import EnhancedFeatureExtractor
                self.enhanced_extractor = EnhancedFeatureExtractor(PREPROCESSOR_PATH)
                print("Loaded enhanced feature extractor")
            except Exception as e:
                print(f"Error loading enhanced feature extractor: {e}, using fallback")
                self.enhanced_extractor = None
        else:
            self.enhanced_extractor = None

    def parse_and_extract(self, log_data):
        """
        Parse and extract features from various log formats.

        Args:
            log_data: Can be either a raw log line (string) or a dict-like object from ModSecurity dataset

        Returns:
            features (list): The numerical features for the model.
            metadata (dict): Raw data for display in the UI.
        """
        # Try semantic extraction first (highest priority)
        if self.use_semantic and self.semantic_model:
            try:
                if isinstance(log_data, str):
                    # Raw log line - extract text for semantic analysis
                    features, metadata = self._semantic_extract_from_raw_log(log_data)
                    if features is not None:
                        return features, metadata
                elif isinstance(log_data, dict):
                    # ModSecurity format - extract all text fields
                    features, metadata = self._semantic_extract_from_modsecurity_row(log_data)
                    if features is not None:
                        return features, metadata
            except Exception as e:
                print(f"Error using semantic extractor: {e}, falling back to enhanced extraction")

        # Try enhanced extraction
        if self.use_enhanced and self.enhanced_extractor:
            try:
                if isinstance(log_data, str):
                    # Raw log line - convert to dict format for enhanced extractor
                    features = self.enhanced_extractor.extract_from_raw_log(log_data)
                    if features is not None:
                        metadata = self._extract_metadata_from_raw_log(log_data)
                        return features, metadata
                elif isinstance(log_data, dict):
                    # ModSecurity format
                    features = self.enhanced_extractor.extract_from_modsecurity_row(log_data)
                    if features is not None:
                        metadata = self._extract_metadata_from_modsecurity_row(log_data)
                        return features, metadata
            except Exception as e:
                print(f"Error using enhanced extractor: {e}, falling back to simple extraction")

        # Fallback to simple extraction for raw logs
        if isinstance(log_data, str):
            return self._simple_extract_from_raw_log(log_data)
        elif isinstance(log_data, dict):
            return self._simple_extract_from_modsecurity_row(log_data)
        else:
            return None, None

    def _semantic_extract_from_raw_log(self, log_line):
        """Semantic feature extraction for raw log lines using LSTM."""
        if not self.semantic_model or not self.semantic_model.preprocessor.is_fitted:
            return None, None

        match = self.LOG_PATTERN.match(log_line)
        if not match:
            return None, None

        data = match.groupdict()

        # Combine text fields for semantic analysis
        semantic_text = f"{data['method']} {data['uri']} {data.get('user_agent', '')}"

        # Get semantic prediction
        semantic_result = self.semantic_model.predict(semantic_text)

        # Combine traditional 10 features with semantic features
        decoded_uri = urllib.parse.unquote(data['uri']).lower()

        # Traditional features (same as original)
        traditional_features = [
            1.0 if 'wp-admin' in decoded_uri else 0.0,        # 1. Is Admin Area?
            1.0 if 'admin-ajax' in decoded_uri else 0.0,      # 2. Is Ajax Call?
            1.0 if data['method'] == 'POST' else 0.0,         # 3. Is POST Request?
            float(len(decoded_uri)) / 100.0,                  # 4. URI Length (Normalized)
            1.0 if any(k in decoded_uri for k in ['<', '>', "'", '"']) else 0.0, # 5. Special Chars (XSS)
            1.0 if 'base64' in decoded_uri else 0.0,          # 6. 'base64' keyword
            1.0 if 'exec' in decoded_uri else 0.0,            # 7. 'exec' keyword
            1.0 if 'union' in decoded_uri else 0.0,           # 8. 'union' keyword (SQLi)
            1.0 if int(data['status']) == 200 else 0.0,       # 9. Status 200 OK
            1.0 if int(data['status']) >= 400 else 0.0,       # 10. Status Error
        ]

        # Add semantic features
        if semantic_result and semantic_result.get('predicted_class') != 'unknown':
            semantic_features = [
                semantic_result.get('confidence', 0.0),            # 11. Semantic confidence
                1.0 if semantic_result.get('is_attack', False) else 0.0,  # 12. Semantic attack prediction
                semantic_result.get('all_probabilities', {}).get('attack', 0.0),  # 13. Attack probability
                semantic_result.get('all_probabilities', {}).get('safe', 0.0),   # 14. Safe probability
            ]
        else:
            # Default semantic features if model fails
            semantic_features = [0.0, 0.0, 0.0, 1.0]  # [conf, attack_pred, attack_prob, safe_prob]

        # Combine all features
        combined_features = traditional_features + semantic_features

        # Metadata for the UI (not used by the model, just for display)
        metadata = {
            'ip': data['ip'],
            'timestamp': data['timestamp'],
            'method': data['method'],
            'uri': data['uri'], # Keep original case for display
            'status': data['status'],
            'user_agent': data.get('user_agent', ''),
            'semantic_prediction': semantic_result.get('predicted_class', 'unknown') if semantic_result else 'unknown',
            'semantic_confidence': semantic_result.get('confidence', 0.0) if semantic_result else 0.0,
            'raw': log_line.strip()
        }

        return combined_features, metadata

    def _semantic_extract_from_modsecurity_row(self, row):
        """Semantic feature extraction for ModSecurity dataset rows using LSTM."""
        if not self.semantic_model or not self.semantic_model.preprocessor.is_fitted:
            return None, None

        try:
            # Extract all relevant text fields for semantic analysis
            text_parts = []

            # Request data
            if 'request_line_method' in row and row['request_line_method']:
                text_parts.append(str(row['request_line_method']))
            if 'request_line_url' in row and row['request_line_url']:
                text_parts.append(str(row['request_line_url']))
            if 'request_useragent' in row and row['request_useragent']:
                text_parts.append(str(row['request_useragent']))
            if 'request_body' in row and row['request_body'] and str(row['request_body']).strip():
                text_parts.append(str(row['request_body']))

            # Response data
            if 'response_body' in row and row['response_body'] and str(row['response_body']).strip():
                text_parts.append(str(row['response_body']))

            # Security messages
            if 'action_message' in row and row['action_message'] and str(row['action_message']).strip():
                text_parts.append(str(row['action_message']))
            if 'message_msg' in row and row['message_msg'] and str(row['message_msg']).strip():
                text_parts.append(str(row['message_msg']))

            # Combine all text for semantic analysis
            semantic_text = ' '.join(filter(None, text_parts))

            # Get semantic prediction
            semantic_result = self.semantic_model.predict(semantic_text)

            # Extract basic information for traditional features
            url = str(row.get('request_line_url', ''))
            method = str(row.get('request_line_method', 'GET'))
            status = str(row.get('response_status', '200'))
            ip = str(row.get('remote_address', 'unknown'))
            timestamp = str(row.get('event_time', 'unknown'))

            decoded_uri = urllib.parse.unquote(url).lower()

            try:
                status_int = int(float(status))
            except (ValueError, TypeError):
                status_int = 200

            # Traditional features (same as original)
            traditional_features = [
                1.0 if 'wp-admin' in decoded_uri else 0.0,        # 1. Is Admin Area?
                1.0 if 'admin-ajax' in decoded_uri else 0.0,      # 2. Is Ajax Call?
                1.0 if method == 'POST' else 0.0,               # 3. Is POST Request?
                float(len(decoded_uri)) / 100.0,                  # 4. URI Length (Normalized)
                1.0 if any(k in decoded_uri for k in ['<', '>', "'", '"']) else 0.0, # 5. Special Chars (XSS)
                1.0 if 'base64' in decoded_uri else 0.0,          # 6. 'base64' keyword
                1.0 if 'exec' in decoded_uri else 0.0,            # 7. 'exec' keyword
                1.0 if 'union' in decoded_uri else 0.0,           # 8. 'union' keyword (SQLi)
                1.0 if status_int == 200 else 0.0,                # 9. Status 200 OK
                1.0 if status_int >= 400 else 0.0,                # 10. Status Error
            ]

            # Add semantic features
            if semantic_result and semantic_result.get('predicted_class') != 'unknown':
                semantic_features = [
                    semantic_result.get('confidence', 0.0),            # 11. Semantic confidence
                    1.0 if semantic_result.get('is_attack', False) else 0.0,  # 12. Semantic attack prediction
                    semantic_result.get('all_probabilities', {}).get('attack', 0.0),  # 13. Attack probability
                    semantic_result.get('all_probabilities', {}).get('safe', 0.0),   # 14. Safe probability
                ]
            else:
                # Default semantic features if model fails
                semantic_features = [0.0, 0.0, 0.0, 1.0]  # [conf, attack_pred, attack_prob, safe_prob]

            # Combine all features
            combined_features = traditional_features + semantic_features

            # Metadata for UI display
            metadata = {
                'ip': ip,
                'timestamp': timestamp,
                'method': method,
                'uri': url, # Keep original case for display
                'status': str(status_int),
                'user_agent': str(row.get('request_useragent', '')),
                'semantic_prediction': semantic_result.get('predicted_class', 'unknown') if semantic_result else 'unknown',
                'semantic_confidence': semantic_result.get('confidence', 0.0) if semantic_result else 0.0,
                'raw': str(row)
            }

            return combined_features, metadata

        except Exception as e:
            print(f"Error in semantic extraction from ModSecurity row: {e}")
            return None, None

    def _simple_extract_from_raw_log(self, log_line):
        """Simple feature extraction for raw log lines (original method)."""
        match = self.LOG_PATTERN.match(log_line)
        if not match:
            return None, None

        data = match.groupdict()
        decoded_uri = urllib.parse.unquote(data['uri']).lower()

        # --- Feature Engineering ---
        # We convert the text log into 10 numbers the AI can understand
        features = [
            1.0 if 'wp-admin' in decoded_uri else 0.0,        # 1. Is Admin Area?
            1.0 if 'admin-ajax' in decoded_uri else 0.0,      # 2. Is Ajax Call?
            1.0 if data['method'] == 'POST' else 0.0,         # 3. Is POST Request?
            float(len(decoded_uri)) / 100.0,                  # 4. URI Length (Normalized)
            1.0 if any(k in decoded_uri for k in ['<', '>', "'", '"']) else 0.0, # 5. Special Chars (XSS)
            1.0 if 'base64' in decoded_uri else 0.0,          # 6. 'base64' keyword
            1.0 if 'exec' in decoded_uri else 0.0,            # 7. 'exec' keyword
            1.0 if 'union' in decoded_uri else 0.0,           # 8. 'union' keyword (SQLi)
            1.0 if int(data['status']) == 200 else 0.0,       # 9. Status 200 OK
            1.0 if int(data['status']) >= 400 else 0.0,       # 10. Status Error
        ]

        # Metadata for the UI (not used by the model, just for display)
        metadata = {
            'ip': data['ip'],
            'timestamp': data['timestamp'],
            'method': data['method'],
            'uri': data['uri'], # Keep original case for display
            'status': data['status'],
            'raw': log_line.strip()
        }

        return features, metadata

    def _simple_extract_from_modsecurity_row(self, row):
        """Simple feature extraction for ModSecurity dataset rows."""
        try:
            # Extract basic information
            url = str(row.get('request_line_url', ''))
            method = str(row.get('request_line_method', 'GET'))
            status = str(row.get('response_status', '200'))
            ip = str(row.get('remote_address', 'unknown'))
            timestamp = str(row.get('event_time', 'unknown'))

            decoded_uri = urllib.parse.unquote(url).lower()

            try:
                status_int = int(float(status))
            except (ValueError, TypeError):
                status_int = 200

            # Use same 10-feature approach as raw logs for consistency
            features = [
                1.0 if 'wp-admin' in decoded_uri else 0.0,        # 1. Is Admin Area?
                1.0 if 'admin-ajax' in decoded_uri else 0.0,      # 2. Is Ajax Call?
                1.0 if method == 'POST' else 0.0,               # 3. Is POST Request?
                float(len(decoded_uri)) / 100.0,                  # 4. URI Length (Normalized)
                1.0 if any(k in decoded_uri for k in ['<', '>', "'", '"']) else 0.0, # 5. Special Chars (XSS)
                1.0 if 'base64' in decoded_uri else 0.0,          # 6. 'base64' keyword
                1.0 if 'exec' in decoded_uri else 0.0,            # 7. 'exec' keyword
                1.0 if 'union' in decoded_uri else 0.0,           # 8. 'union' keyword (SQLi)
                1.0 if status_int == 200 else 0.0,                # 9. Status 200 OK
                1.0 if status_int >= 400 else 0.0,                # 10. Status Error
            ]

            # Metadata for UI display
            metadata = {
                'ip': ip,
                'timestamp': timestamp,
                'method': method,
                'uri': url, # Keep original case for display
                'status': str(status_int),
                'raw': str(row)
            }

            return features, metadata

        except Exception as e:
            print(f"Error extracting from ModSecurity row: {e}")
            return None, None

    def _extract_metadata_from_raw_log(self, log_line):
        """Extract metadata from raw log line."""
        match = self.LOG_PATTERN.match(log_line)
        if match:
            data = match.groupdict()
            return {
                'ip': data['ip'],
                'timestamp': data['timestamp'],
                'method': data['method'],
                'uri': data['uri'],
                'status': data['status'],
                'raw': log_line.strip()
            }
        return {'ip': 'unknown', 'timestamp': 'unknown', 'method': 'unknown', 'uri': 'unknown', 'status': 'unknown', 'raw': log_line.strip()}

    def _extract_metadata_from_modsecurity_row(self, row):
        """Extract metadata from ModSecurity dataset row."""
        return {
            'ip': str(row.get('remote_address', 'unknown')),
            'timestamp': str(row.get('event_time', 'unknown')),
            'method': str(row.get('request_line_method', 'unknown')),
            'uri': str(row.get('request_line_url', 'unknown')),
            'status': str(row.get('response_status', 'unknown')),
            'raw': str(row)
        }
