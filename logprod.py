import time
import zmq
import os
import sys
import json
import argparse
import glob
import numpy as np
import urllib.request
import urllib.error
from pathlib import Path

# Import the new TensorFlow inference engine
try:
    from detectors.tensorflow_semantic_inference import TensorFlowSemanticInference
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow inference module not found or failed to load.")
    TENSORFLOW_AVAILABLE = False

# Import existing feature extractor for metadata


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

import re
import urllib.parse
# Regex to parse standard Combined Log Format
LOG_PATTERN = re.compile(
    r'(?P<ip>[\d\.]+) - - \[(?P<timestamp>.*?)\] "(?P<method>\w+) (?P<uri>.*?) (?P<protocol>HTTP\/[\d\.]+)" (?P<status>\d+) (?P<size>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
)

def _parse_raw_log_line(log_line):
    """
    Parses a raw log line using regex and extracts relevant components.
    Returns a dictionary of extracted data or None if parsing fails.
    """
    match = LOG_PATTERN.match(log_line)
    if not match:
        return None
    data = match.groupdict()
    # Ensure URI is unquoted and lowercased for consistent feature extraction
    data['decoded_uri'] = urllib.parse.unquote(data['uri']).lower()
    return data

class LogProducer:
    def __init__(self, port=ZMQ_PORT, api_url=None):
        self.port = port
        self.api_url = api_url
        self.context = zmq.Context()
        
        # Only bind ZMQ if not using a remote API for inference
        if not self.api_url:
            self.socket = self.context.socket(zmq.PUB)
            try:
                self.socket.bind(f"tcp://*:{self.port}")
                print(f"🚀 Producer started on port {self.port}")
            except zmq.error.ZMQError:
                print(f"❌ Error: Port {self.port} is in use. Is the producer already running?")
                sys.exit(1)
        else:
            self.socket = None # No local ZMQ binding


        
        if self.api_url:
            print(f"🌐 Using Remote API for inference: {self.api_url}")
            self.tf_model = None
        elif TENSORFLOW_AVAILABLE:
            print("🧠 Initializing TensorFlow Semantic Model...")
            self.tf_model = TensorFlowSemanticInference()
        else:
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
            print(f"⚠️ API Error: {e}")
            return 'error', 0.0, {}

    def process_log_line(self, line):
        if not line.strip():
            return

        try:
            parsed_data = _parse_raw_log_line(line)
            if not parsed_data:
                print(f"⚠️  Could not parse log line: {line.strip()}")
                return

            # Extract data from parsed log
            ip = parsed_data['ip']
            timestamp = parsed_data['timestamp']
            method = parsed_data['method']
            uri = parsed_data['uri']
            status = parsed_data['status']
            user_agent = parsed_data.get('user_agent', '')
            
            # Semantic text for the TF model
            semantic_text = f"{method} {uri} {user_agent}"

            semantic_result = {}
            if self.api_url:
                # Remote Inference
                # Use the original log line for remote API if it processes the full log
                # Otherwise, send the semantic_text if the API expects pre-extracted text
                # For consistency with local processing, send semantic_text.
                label, conf, probs = self._call_remote_api(semantic_text, parsed_data) 
                if label != 'error':
                    semantic_result = {
                        'semantic_prediction': label,
                        'semantic_confidence': float(conf),
                        'semantic_probs': probs
                    }
            elif self.tf_model:
                # Local Inference
                label, conf, probs = self.tf_model.predict(semantic_text)
                semantic_result = {
                    'semantic_prediction': label,
                    'semantic_confidence': float(conf),
                    'semantic_probs': probs
                }
            elif self.tf_model:
                # Local Inference
                label, conf, probs = self.tf_model.predict(semantic_text)
                semantic_result = {
                    'semantic_prediction': label,
                    'semantic_confidence': float(conf),
                    'semantic_probs': probs
                }

            # Construct metadata for dashboard
            metadata = {
                'ip': ip,
                'timestamp': timestamp,
                'method': method,
                'uri': uri,
                'status': status,
                'user_agent': user_agent,
                'raw': line.strip()
            }
            
            if semantic_result:
                metadata.update(semantic_result)
                metadata['is_attack'] = semantic_result.get('semantic_prediction', 'normal') != 'normal'
            else:
                # Default if no semantic result (e.g., model not loaded)
                metadata['semantic_prediction'] = 'unknown'
                metadata['semantic_confidence'] = 0.0
                metadata['is_attack'] = False

            payload = {
                'metadata': metadata
            }
            
            payload = sanitize_for_json(payload)

            if not self.api_url:
                self.socket.send_string("logs", flags=zmq.SNDMORE)
                self.socket.send_json(payload)

            self._print_status(metadata)
            
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

        print(f"{status} Sent: {uri[:100]}...")

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
        if self.socket: # Only close if a socket was created
            self.socket.close()
        self.context.term()

def main():
    parser = argparse.ArgumentParser(description="ModSec-IDS Log Producer")
    parser.add_argument('--input', type=str, default=DEFAULT_LOG_FILE, help='Input log file or pattern')
    parser.add_argument('--continuous', action='store_true', help='Monitor file in real-time (tail -f)')
    parser.add_argument('--bulk-process', action='store_true', help='Process all matching files and exit')
    parser.add_argument('--api-url', type=str, help='URL of the API Log Producer (e.g., http://localhost:8000)')
    
    args = parser.parse_args()
    
    producer = LogProducer(api_url=args.api_url)

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
