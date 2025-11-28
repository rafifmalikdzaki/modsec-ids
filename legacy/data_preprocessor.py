import pandas as pd
import numpy as np
import re
import urllib.parse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import torch

# TensorFlow imports for semantic model preprocessing
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available - semantic model preprocessing disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModSecurityPreprocessor:
    """Preprocessor for ModSecurity WordPress dataset."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.ip_encoder = LabelEncoder()
        self.method_encoder = LabelEncoder()
        self.feature_columns = [
            'url_length', 'decoded_url_length', 'is_admin', 'is_ajax',
            'has_xss_chars', 'has_base64', 'has_sql_keywords', 'has_rce_keywords',
            'is_post_method', 'status_code', 'is_error_status', 'is_bot', 'user_agent_length',
            'was_blocked', 'has_attack_pattern'
        ]
        self.fitted = False


class SemanticTextPreprocessor:
    """
    Preprocessor for TensorFlow semantic model that handles text-based features.

    This preprocessor is designed specifically for the LSTM semantic model
    that processes raw text from HTTP requests, responses, and security messages.
    """

    def __init__(self, vocab_size: int = 20000, max_sequence_length: int = 500):
        """
        Initialize semantic text preprocessor.

        Args:
            vocab_size: Maximum vocabulary size for tokenization
            max_sequence_length: Maximum sequence length for padding
        """
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.fitted = False

        # Multi-class attack labels for semantic model
        self.attack_classes = {
            'normal': 0,
            'sqli': 1,
            'bruteforce': 2,
            'lfi': 3,
            'xss': 4,
            'rce': 5,
            'directory_traversal': 6,
            'command_injection': 7
        }

        # Reverse mapping for predictions
        self.idx_to_class = {v: k for k, v in self.attack_classes.items()}

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data for semantic analysis.

        Args:
            text: Raw text string

        Returns:
            str: Cleaned text
        """
        if text is None or pd.isna(text):
            return ""

        # Convert to string and lowercase
        text = str(text).lower()

        # Remove excessive whitespace but keep important security characters
        text = ' '.join(text.split())

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def prepare_semantic_features(self, df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """
        Prepare comprehensive text features for semantic analysis.

        Args:
            df: Input DataFrame with HTTP log data

        Returns:
            Tuple of (texts, labels)
        """
        logger.info("Preparing semantic text features...")

        texts = []
        labels = []

        for idx, row in df.iterrows():
            # Combine all relevant text fields for comprehensive semantic analysis
            text_parts = []

            # HTTP Request components
            if 'request_line_method' in row and pd.notna(row['request_line_method']):
                text_parts.append(str(row['request_line_method']))
            if 'request_line_url' in row and pd.notna(row['request_line_url']):
                text_parts.append(str(row['request_line_url']))
            if 'request_useragent' in row and pd.notna(row['request_useragent']):
                text_parts.append(str(row['request_useragent']))
            if 'request_body' in row and pd.notna(row['request_body']) and str(row['request_body']).strip():
                text_parts.append(str(row['request_body']))

            # HTTP Request metadata
            if 'request_host' in row and pd.notna(row['request_host']):
                text_parts.append(str(row['request_host']))

            # Security detection components
            if 'action_message' in row and pd.notna(row['action_message']) and str(row['action_message']).strip():
                text_parts.append(str(row['action_message']))
            if 'message_msg' in row and pd.notna(row['message_msg']) and str(row['message_msg']).strip():
                text_parts.append(str(row['message_msg']))
            if 'message_description' in row and pd.notna(row['message_description']) and str(row['message_description']).strip():
                text_parts.append(str(row['message_description']))

            # Additional context fields
            if 'full_message_line' in row and pd.notna(row['full_message_line']) and str(row['full_message_line']).strip():
                text_parts.append(str(row['full_message_line']))

            # Combine all text fields
            combined_text = ' '.join(filter(None, text_parts))

            if combined_text.strip():
                # Clean combined text
                cleaned_text = self.clean_text(combined_text.strip())
                texts.append(cleaned_text)

                # Extract and normalize label for multi-class classification
                label_str = str(row.get('label', 'normal')).strip().lower()

                # Map label to class index using attack_classes
                label_idx = 0  # Default to normal
                for class_name, class_idx in self.attack_classes.items():
                    if label_str == class_name.lower() or label_str.replace(' ', '_').replace('-', '_') == class_name:
                        label_idx = class_idx
                        break

                # Handle common variations
                if label_str in ['attack', 'malicious', 'blocked']:
                    label_idx = 1  # Default to SQLi as most common
                elif label_str in ['normal', 'benign', 'safe', 'legitimate']:
                    label_idx = 0  # Normal

                labels.append(label_idx)

            # Progress reporting
            if (idx + 1) % 5000 == 0:
                processed_percentage = ((idx + 1) / len(df)) * 100
                logger.info(f"Processed {idx + 1:,}/{len(df):,} records ({processed_percentage:.1f}%)")

        logger.info(f"Semantic feature preparation completed: {len(texts):,} samples")

        # Label distribution
        from collections import Counter
        label_counts = Counter(labels)
        logger.info("Label distribution:")
        for label_idx, count in sorted(label_counts.items()):
            class_name = self.idx_to_class.get(label_idx, 'unknown')
            percentage = (count / len(labels)) * 100
            logger.info(f"   {class_name:20s}: {count:6,} ({percentage:5.1f}%)")

        return texts, labels

    def fit(self, df: pd.DataFrame) -> 'SemanticTextPreprocessor':
        """
        Fit preprocessor on dataset for semantic analysis.

        Args:
            df: Input DataFrame

        Returns:
            Fitted preprocessor instance
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for semantic preprocessing")

        logger.info("Fitting semantic text preprocessor...")

        # Prepare text features and labels
        texts, labels = self.prepare_semantic_features(df)

        if not texts:
            raise ValueError("No valid text data found in dataset")

        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )

        # Fit tokenizer on texts
        self.tokenizer.fit_on_texts(texts)

        logger.info(f"Tokenizer fitted: vocabulary size {len(self.tokenizer.word_index):,}")

        # Fit label encoder
        self.label_encoder.fit([self.idx_to_class[label] for label in labels])
        logger.info(f"Label encoder fitted for classes: {list(self.label_encoder.classes_)}")

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform dataset using fitted semantic preprocessor.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (padded_sequences, encoded_labels)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for semantic preprocessing")

        logger.info("Transforming dataset with semantic preprocessor...")

        # Prepare text features and labels
        texts, labels = self.prepare_semantic_features(df)

        if not texts:
            raise ValueError("No valid text data found in dataset")

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)

        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        # Encode labels
        label_names = [self.idx_to_class[label] for label in labels]
        encoded_labels = self.label_encoder.transform(label_names)

        logger.info(f"Transformed {len(padded_sequences):,} samples")
        logger.info(f"Sequence shape: {padded_sequences.shape}")
        logger.info(f"Label distribution: {np.bincount(encoded_labels)}")

        return padded_sequences, encoded_labels

    def save(self, filepath: str):
        """
        Save semantic preprocessor to disk.

        Args:
            filepath: Path to save preprocessor
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for semantic preprocessing")

        preprocessor_data = {
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'attack_classes': self.attack_classes,
            'idx_to_class': self.idx_to_class,
            'fitted': self.fitted,
            'preprocessor_type': 'semantic'
        }

        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Semantic preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SemanticTextPreprocessor':
        """
        Load semantic preprocessor from disk.

        Args:
            filepath: Path to load preprocessor from

        Returns:
            Loaded preprocessor instance
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for semantic preprocessing")

        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)

        # Validate preprocessor type
        if preprocessor_data.get('preprocessor_type') != 'semantic':
            raise ValueError(f"Invalid preprocessor type: {preprocessor_data.get('preprocessor_type')}")

        preprocessor = cls(
            vocab_size=preprocessor_data['vocab_size'],
            max_sequence_length=preprocessor_data['max_sequence_length']
        )

        preprocessor.tokenizer = preprocessor_data['tokenizer']
        preprocessor.label_encoder = preprocessor_data['label_encoder']
        preprocessor.attack_classes = preprocessor_data['attack_classes']
        preprocessor.idx_to_class = preprocessor_data['idx_to_class']
        preprocessor.fitted = preprocessor_data['fitted']

        logger.info(f"Semantic preprocessor loaded from {filepath}")
        return preprocessor

    def get_text_statistics(self, texts: List[str]) -> Dict:
        """
        Get comprehensive statistics about text data.

        Args:
            texts: List of text strings

        Returns:
            Dictionary with text statistics
        """
        if not texts:
            return {}

        # Calculate basic statistics
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]

        statistics = {
            'total_samples': len(texts),
            'mean_word_count': np.mean(word_counts),
            'median_word_count': np.median(word_counts),
            'std_word_count': np.std(word_counts),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts),
            'mean_char_count': np.mean(char_counts),
            'median_char_count': np.median(char_counts),
            'min_char_count': min(char_counts),
            'max_char_count': max(char_counts),
            'p95_word_count': np.percentile(word_counts, 95),
            'p99_word_count': np.percentile(word_counts, 99)
        }

        # Vocabulary analysis
        all_words = ' '.join(texts).split()
        unique_words = set(all_words)
        statistics['vocabulary_size'] = len(unique_words)
        statistics['total_words'] = len(all_words)

        return statistics

    def _extract_url_features(self, url: str) -> Dict[str, float]:
        """Extract URL-based features."""
        if not url or pd.isna(url):
            url = ''

        decoded_url = urllib.parse.unquote(str(url).lower())

        # XSS character patterns
        xss_chars = ['<', '>', '\'', '"', '&', 'javascript:', 'vbscript:', 'onload', 'onerror']

        # SQL injection patterns
        sql_keywords = ['union', 'select', 'insert', 'update', 'delete', 'drop',
                       'exec', 'execute', 'sp_', 'xp_', 'declare', 'cast']

        # Remote code execution patterns
        rce_keywords = ['exec', 'system', 'shell', 'cmd', 'powershell', 'bash',
                       'eval', 'passthru', 'proc_open']

        return {
            'url_length': len(str(url)),
            'decoded_url_length': len(decoded_url),
            'is_admin': float('wp-admin' in decoded_url or 'wp-login' in decoded_url),
            'is_ajax': float('admin-ajax' in decoded_url or 'ajax' in decoded_url),
            'has_xss_chars': float(any(char in decoded_url for char in xss_chars)),
            'has_base64': float('base64' in decoded_url or 'bm' in decoded_url),
            'has_sql_keywords': float(any(keyword in decoded_url for keyword in sql_keywords)),
            'has_rce_keywords': float(any(keyword in decoded_url for keyword in rce_keywords))
        }

    def _extract_http_features(self, method: str, status_code: str) -> Dict[str, float]:
        """Extract HTTP-based features."""
        if not method:
            method = 'GET'
        if not status_code:
            status_code = '200'

        try:
            status_int = int(float(str(status_code)))
        except (ValueError, TypeError):
            status_int = 200

        return {
            'is_post_method': float(method.upper() == 'POST'),
            'status_code': status_int / 600.0,  # Normalize status codes
            'is_error_status': float(status_int >= 400)
        }

    def _extract_user_agent_features(self, user_agent: str) -> Dict[str, float]:
        """Extract User-Agent based features."""
        if not user_agent or pd.isna(user_agent):
            user_agent = ''

        user_agent_lower = str(user_agent).lower()

        # Bot detection patterns
        bot_patterns = ['bot', 'crawler', 'spider', 'scraper', 'curl', 'wget',
                       'python', 'java', 'requests', 'http', 'scan']

        return {
            'user_agent_length': len(str(user_agent)),
            'is_bot': float(any(pattern in user_agent_lower for pattern in bot_patterns))
        }

    def _extract_attack_features(self, action: str, action_message: str,
                               message_type: str, message_msg: str) -> Dict[str, float]:
        """Extract attack pattern features."""
        action = str(action).lower() if action else ''
        action_message = str(action_message).lower() if action_message else ''
        message_type = str(message_type).lower() if message_type else ''
        message_msg = str(message_msg).lower() if message_msg else ''

        # Attack pattern detection
        attack_patterns = {
            'sqli': ['sqli', 'sql injection', 'union select', 'sqlmap'],
            'xss': ['xss', 'cross-site scripting', 'script injection', 'javascript'],
            'rce': ['rce', 'remote code execution', 'command injection', 'shell'],
            'lfi': ['lfi', 'local file inclusion', 'file inclusion'],
            'rfi': ['rfi', 'remote file inclusion', 'file inclusion'],
            'dos': ['dos', 'denial of service', 'flood']
        }

        has_attack_pattern = 0.0
        for pattern_name, keywords in attack_patterns.items():
            if any(keyword in action_message or keyword in message_msg for keyword in keywords):
                has_attack_pattern = 1.0
                break

        return {
            'was_blocked': float('blocked' in action or 'deny' in action),
            'has_attack_pattern': has_attack_pattern
        }

    def extract_features(self, row: pd.Series) -> np.ndarray:
        """Extract all features from a single row."""
        try:
            # URL features
            url = row.get('request_line_url', '')
            url_features = self._extract_url_features(url)

            # HTTP features
            method = row.get('request_line_method', '')
            status_code = row.get('response_status', '')
            http_features = self._extract_http_features(method, status_code)

            # User-Agent features
            user_agent = row.get('request_useragent', '')
            ua_features = self._extract_user_agent_features(user_agent)

            # Attack features
            action = row.get('action', '')
            action_message = row.get('action_message', '')
            message_type = row.get('message_type', '')
            message_msg = row.get('message_msg', '')
            attack_features = self._extract_attack_features(action, action_message,
                                                         message_type, message_msg)

            # Combine all features
            all_features = {**url_features, **http_features, **ua_features, **attack_features}

            # Return in the correct order
            return np.array([all_features[col] for col in self.feature_columns], dtype=np.float32)

        except Exception as e:
            logger.warning(f"Error extracting features from row: {e}")
            return np.zeros(len(self.feature_columns), dtype=np.float32)

    def fit(self, df: pd.DataFrame) -> 'ModSecurityPreprocessor':
        """Fit the preprocessor on the dataset."""
        logger.info("Fitting preprocessor on dataset...")

        # Extract features for all rows
        features = []
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                logger.info(f"Processing row {idx}...")

            feature_vector = self.extract_features(row)
            features.append(feature_vector)

        features = np.array(features)
        logger.info(f"Extracted features shape: {features.shape}")

        # Fit scaler
        self.scaler.fit(features)
        logger.info("Scaler fitted successfully")

        # Fit label encoder
        labels = self._extract_labels(df)
        self.label_encoder.fit(labels)
        logger.info(f"Label encoder fitted for classes: {self.label_encoder.classes_}")

        self.fitted = True
        return self

    def _extract_labels(self, df: pd.DataFrame) -> List[str]:
        """Extract multi-class labels from the dataset."""
        labels = []
        for _, row in df.iterrows():
            label_str = str(row.get('label', 'normal')).strip().lower()

            # Map common variations to standard labels
            if label_str in ['normal', 'benign', 'safe', 'legitimate']:
                labels.append('normal')
            elif label_str in ['attack', 'malicious', 'blocked']:
                # Default to most common attack type if not specified
                labels.append('sqli')  # Most common in our dataset
            else:
                # Use the label as-is for specific attack types
                # Normalize some common variations
                normalized_label = label_str.replace(' ', '_').replace('-', '_')
                labels.append(normalized_label)
        return labels

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform the dataset using fitted preprocessor."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")

        logger.info("Transforming dataset...")

        # Extract features
        features = []
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                logger.info(f"Transforming row {idx}...")

            feature_vector = self.extract_features(row)
            features.append(feature_vector)

        features = np.array(features)

        # Normalize features
        features_normalized = self.scaler.transform(features)

        # Extract and encode labels
        labels = self._extract_labels(df)
        labels_encoded = self.label_encoder.transform(labels)

        logger.info(f"Transformed {len(features)} samples")
        logger.info(f"Label distribution: {np.bincount(labels_encoded)}")

        return features_normalized, labels_encoded

    def save(self, filepath: str):
        """Save the preprocessor to disk."""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'fitted': self.fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ModSecurityPreprocessor':
        """Load a preprocessor from disk."""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)

        preprocessor = cls()
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.label_encoder = preprocessor_data['label_encoder']
        preprocessor.feature_columns = preprocessor_data['feature_columns']
        preprocessor.fitted = preprocessor_data['fitted']

        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor

def clean_and_preprocess_dataset(input_path: str, output_path: str, sample_size: Optional[int] = None):
    """Clean and preprocess the ModSecurity dataset."""

    logger.info(f"Loading dataset from {input_path}")

    try:
        # Read the CSV file with encoding handling
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                logger.info(f"Successfully loaded dataset with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to read with {encoding}: {e}")
                if encoding == encodings_to_try[-1]:  # Last encoding to try
                    raise e
                continue

        logger.info(f"Original dataset shape: {df.shape}")

        # Remove rows with missing critical fields and clean data
        critical_fields = ['request_line_url', 'response_status', 'request_line_method']

        # Remove rows with missing critical fields
        df = df.dropna(subset=critical_fields)

        # Filter out rows with HTML content in URL field (malformed entries)
        if 'request_line_url' in df.columns:
            # Remove rows where URL contains HTML tags
            html_indicators = ['<html', '<!DOCTYPE', '<script', 'alert(', 'function(']
            try:
                df = df[~df['request_line_url'].str.contains('|'.join(html_indicators), case=False, na=False)]
            except Exception as e:
                logger.warning(f"Error filtering HTML content: {e}")

        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled dataset shape: {df.shape}")

        # Initialize and fit preprocessor
        preprocessor = ModSecurityPreprocessor()
        preprocessor.fit(df)

        # Transform the data
        features, labels = preprocessor.transform(df)

        # Save processed data with safe pickling
        processed_data = {
            'features': features,
            'labels': labels,
            'feature_columns': preprocessor.feature_columns,
            'classes': preprocessor.label_encoder.classes_
        }

        # Save as numpy arrays (safe format)
        np.savez_compressed(output_path,
                           features=features,
                           labels=labels,
                           feature_columns=preprocessor.feature_columns,
                           classes=preprocessor.label_encoder.classes_)

        logger.info(f"Processed data saved to {output_path}")

        # Save preprocessor separately
        preprocessor_path = output_path.replace('.npz', '_preprocessor.pkl')
        preprocessor.save(preprocessor_path)

        # Print statistics
        logger.info("Dataset Statistics:")
        logger.info(f"  Total samples: {len(features)}")
        logger.info(f"  Feature dimensions: {features.shape[1]}")
        logger.info(f"  Classes: {preprocessor.label_encoder.classes_}")
        logger.info(f"  Class distribution: {np.bincount(labels)}")

        return processed_data

    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess ModSecurity dataset")
    parser.add_argument("--input", type=str, default="data/raw/Modsec-WP.csv",
                       help="Input CSV file path")
    parser.add_argument("--output", type=str, default="data/processed/modsec_processed.npz",
                       help="Output processed data path")
    parser.add_argument("--sample", type=int, default=None,
                       help="Sample size (use None for full dataset)")

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Process the dataset
    clean_and_preprocess_dataset(args.input, args.output, args.sample)