import time
import zmq
import os
import random
import sys
from security_model import FeatureExtractor

# Configuration
LOG_FILE = 'data/raw/access.txt'
ZMQ_PORT = 5555

def follow_file(filename):
    """Generator that mimics 'tail -f'."""
    print(f"Checking for {filename}...")
    while not os.path.exists(filename):
        time.sleep(1)

    print(f"Found {filename}. Streaming logs...")
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        # NOTE: In a real scenario, you might want f.seek(0, os.SEEK_END)
        # to only show *new* logs. For this demo, we read from the start.
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line

def main():
    # 1. Setup ZeroMQ Publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    try:
        socket.bind(f"tcp://*:{ZMQ_PORT}")
    except zmq.error.ZMQError:
        print(f"Error: Port {ZMQ_PORT} is in use. Is the producer already running?")
        sys.exit(1)
    
    print(f"🚀 Producer started on port {ZMQ_PORT}")

    extractor = FeatureExtractor()

    try:
        for line in follow_file(LOG_FILE):
            if not line.strip():
                continue

            # 2. Extract Features
            features, metadata = extractor.parse_and_extract(line)
            
            if features:
                # 3. Pack data for transmission
                payload = {
                    'features': features,
                    'metadata': metadata
                }
                
                # 4. Send via ZMQ (Topic: 'logs')
                socket.send_string("logs", flags=zmq.SNDMORE)
                socket.send_json(payload)
                
                # Simulate a slight delay to make the dashboard readable
                time.sleep(0.2) 
                
                # Feedback in console
                status = "🔴" if "exec" in metadata['uri'] or "base64" in metadata['uri'] else "🟢"
                print(f"{status} Sent: {metadata['uri'][:50]}...")

    except KeyboardInterrupt:
        print("\nStopping producer...")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
