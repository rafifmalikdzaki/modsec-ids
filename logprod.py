import time
import zmq
import os
import sys
import json
import argparse
import glob
import numpy as np
from pathlib import Path

# Import the new TensorFlow inference engine
try:
    from detectors.tensorflow_semantic_inference import TensorFlowSemanticInference
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow inference module not found or failed to load.")
    TENSORFLOW_AVAILABLE = False

# Import existing feature extractor for metadata
from detectors.security_model import FeatureExtractor

# Configuration
DEFAULT_LOG_FILE = 'data/raw/access.txt'
ZMQ_PORT = 5555

def sanitize_for_json(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    return obj

class LogProducer:
    def __init__(self, port=ZMQ_PORT):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        try:
            self.socket.bind(f"tcp://*:{self.port}")
            print(f"🚀 Producer started on port {self.port}")
        except zmq.error.ZMQError:
            print(f"❌ Error: Port {self.port} is in use. Is the producer already running?")
            sys.exit(1)

        # Initialize Inference Engines
        self.extractor = FeatureExtractor(use_enhanced=False, use_semantic=True)
        
        if TENSORFLOW_AVAILABLE:
            print("🧠 Initializing TensorFlow Semantic Model...")
            self.tf_model = TensorFlowSemanticInference()
        else:
            self.tf_model = None

    def process_log_line(self, line):
        if not line.strip():
            return

        try:
            # 1. TensorFlow Semantic Inference (Primary)
            semantic_result = {}
            if self.tf_model:
                label, conf, probs = self.tf_model.predict(line.strip())
                semantic_result = {
                    'semantic_prediction': label,
                    'semantic_confidence': float(conf),
                    'semantic_probs': probs
                }

            # 2. Feature Extraction (Secondary/Metadata)
            # We still use this to get IP, method, status, etc. for the dashboard
            features, metadata = self.extractor.parse_and_extract(line)

            if features:
                # 3. Merge Results
                # We inject the semantic results into metadata so the dashboard can use them
                if semantic_result:
                    metadata.update(semantic_result)
                    # Also set a top-level 'is_attack' flag for easier consumption
                    metadata['is_attack'] = semantic_result['semantic_prediction'] != 'normal'

                payload = {
                    'features': features,
                    'metadata': metadata
                }
                
                # Sanitize payload for JSON serialization (fix int64 error)
                payload = sanitize_for_json(payload)

                # 4. Publish
                self.socket.send_string("logs", flags=zmq.SNDMORE)
                self.socket.send_json(payload)

                # Feedback
                self._print_status(metadata)
                
                # Rate limiting for demo purposes (prevent flooding)
                time.sleep(0.05)

        except Exception as e:
            print(f"⚠️  Error processing line: {e}")

    def _print_status(self, metadata):
        """Print a colorful status line."""
        prediction = str(metadata.get('semantic_prediction', 'unknown'))
        uri = metadata.get('uri', '')
        
        if prediction == 'normal':
            status = "🟢"
            color_end = ""
        elif prediction == 'sqli':
            status = "🔴 SQLi"
        elif prediction == 'xss':
            status = "🟠 XSS"
        elif prediction == 'directory_traversal':
            status = "🟡 Trav"
        else:
            status = f"🔴 {prediction.upper()}"

        print(f"{status} Sent: {uri[:60]}...")

    def follow_file(self, filename):
        """Generator that mimics 'tail -f'."""
        print(f"📂 Monitoring {filename}...")
        while not os.path.exists(filename):
            print(f"Waiting for {filename}...")
            time.sleep(1)

        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            # Go to end of file
            f.seek(0, os.SEEK_END)
            
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                yield line

    def process_bulk(self, file_pattern):
        """Process multiple files matching a pattern."""
        files = glob.glob(file_pattern)
        if not files:
            print(f"❌ No files found matching: {file_pattern}")
            return

        print(f"📚 Bulk processing {len(files)} files...")
        for filepath in files:
            print(f"📄 Processing {filepath}...")
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    self.process_log_line(line)

    def close(self):
        self.socket.close()
        self.context.term()

def main():
    parser = argparse.ArgumentParser(description="ModSec-IDS Log Producer")
    parser.add_argument('--input', type=str, default=DEFAULT_LOG_FILE, help='Input log file or pattern')
    parser.add_argument('--continuous', action='store_true', help='Monitor file in real-time (tail -f)')
    parser.add_argument('--bulk-process', action='store_true', help='Process all matching files and exit')
    
    args = parser.parse_args()
    
    producer = LogProducer()

    try:
        if args.bulk_process:
            producer.process_bulk(args.input)
            print("✅ Bulk processing complete.")
        elif args.continuous:
            # Tail mode
            for line in producer.follow_file(args.input):
                producer.process_log_line(line)
        else:
            # Default: Read file from start, then tail (Simulated "Continuous" from original script)
            # Or just follow behavior of original script which read from start?
            # The original script read from start and then tailed. 
            # Let's implement "Read existing then tail" as the default behavior if no flags
            if os.path.exists(args.input):
                print(f"📄 Reading existing logs from {args.input}...")
                with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        producer.process_log_line(line)
                
                # Then continue tailing
                print("🔄 Switching to real-time monitoring...")
                for line in producer.follow_file(args.input):
                    producer.process_log_line(line)
            else:
                 for line in producer.follow_file(args.input):
                    producer.process_log_line(line)

    except KeyboardInterrupt:
        print("\n🛑 Stopping producer...")
    finally:
        producer.close()

if __name__ == "__main__":
    main()
