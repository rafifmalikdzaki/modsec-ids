#!/usr/bin/env python3
"""
API Log Producer & Analyzer

This tool starts a persistent HTTP server that:
1. Loads the heavy TensorFlow Semantic Model ONCE at startup.
2. Listens for HTTP POST requests containing log lines.
3. Analyzes the log instantly.
4. Publishes the result to the IDSDashboard (via ZeroMQ).
5. Returns the analysis result as JSON.

Usage:
    python tools/api_log_producer.py [--port 8000]

Example Client:
    curl -X POST -d "GET /admin.php?id=1 OR 1=1" http://localhost:8000/analyze
"""

import http.server
import socketserver
import argparse
import sys
import os
import json
import time
import zmq
import random
import numpy as np

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from detectors.tensorflow_semantic_inference import TensorFlowSemanticInference
    from detectors.security_model import FeatureExtractor
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Error importing detector: {e}")
    MODEL_AVAILABLE = False

# Configuration
ZMQ_PORT = 5555
HTTP_PORT = 8000

# Global objects (loaded once)
detector = None
feature_extractor = None
zmq_socket = None

class LogRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle simple status checks."""
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            status = {"status": "running", "model_loaded": detector is not None}
            self.wfile.write(json.dumps(status).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle log analysis requests."""
        if self.path == '/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            # Check if input is JSON or raw text
            log_line = post_data
            metadata = {}
            try:
                json_data = json.loads(post_data)
                if 'log' in json_data:
                    log_line = json_data['log']
                if 'metadata' in json_data:
                    metadata = json_data['metadata']
            except json.JSONDecodeError:
                pass # Treat as raw text

            # 1. Analyze
            result = self.analyze_log(log_line)
            
            # 2. Respond to Client
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*') # CORS
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            
            # 3. Publish to Dashboard (ZeroMQ)
            self.publish_to_dashboard(log_line, result, metadata)
            
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass

    def analyze_log(self, log_line):
        """Run inference on the log line."""
        if not detector:
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        label, conf, probs = detector.predict(log_line)
        inference_time = (time.time() - start_time) * 1000

        print(f"⚡ Analyzed: {log_line[:50]}... -> {label} ({conf:.2%})")

        return {
            "log": log_line,
            "prediction": label,
            "confidence": float(conf),
            "is_attack": label != "normal",
            "probabilities": probs,
            "inference_time_ms": float(f"{inference_time:.2f}")
        }

    def publish_to_dashboard(self, log_line, result, client_metadata=None):
        """Send the result to the ZeroMQ bus for the dashboard."""
        if not zmq_socket:
            return

        # Create a realistic metadata packet
        # We use the feature extractor to get basic IP/Method info if possible
        # Otherwise we generate placeholders
        
        # Try to extract basic features (method, ip, etc) if it looks like a log line
        # If it's just a payload string, we'll have to fake the rest
        features_arr = np.zeros(15) # Dummy features
        
        # Default metadata
        metadata = {
            'ip': '127.0.0.1', # Remote submitter
            'port': '0',
            'method': 'API',
            'uri': log_line[:100], # Use the input as URI/Payload
            'status': '200'
        }
        
        # Override with client provided metadata
        if client_metadata:
            metadata.update(client_metadata)

        # Add inference results
        metadata.update({
            'semantic_prediction': result['prediction'],
            'semantic_confidence': result['confidence'],
            'semantic_probs': result['probabilities'],
            'is_attack': result['is_attack']
        })

        # Construct full payload
        payload = {
            'features': features_arr.tolist(),
            'metadata': metadata
        }

        try:
            zmq_socket.send_string("logs", flags=zmq.SNDMORE)
            zmq_socket.send_json(payload)
        except Exception as e:
            print(f"⚠️ Failed to publish to ZMQ: {e}")

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def run_server(port=HTTP_PORT):
    global detector, zmq_socket
    
    # 1. Initialize ZeroMQ
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUB)
    try:
        zmq_socket.bind(f"tcp://*:{ZMQ_PORT}")
        print(f"📡 ZMQ Publisher bound to port {ZMQ_PORT}")
    except zmq.error.ZMQError:
        print(f"⚠️  Port {ZMQ_PORT} is in use. Dashboard publishing might fail if another producer is running.")
        # We allow continuing, maybe the user just wants the HTTP API response

    # 2. Load Model
    print("🔄 Loading TensorFlow Semantic Model... (This takes a few seconds)")
    if MODEL_AVAILABLE:
        detector = TensorFlowSemanticInference()
        print("✅ Model Loaded Successfully!")
    else:
        print("❌ Model import failed. Service will run but cannot predict.")

    # 3. Start Server
    server_address = ('', port)
    try:
        httpd = ReusableTCPServer(server_address, LogRequestHandler)
        print(f"\n🚀 API Producer listening on http://localhost:{port}")
        print(f"   POST to /analyze to test samples.")
        print(f"   Example: curl -X POST -d 'GET /evil.php' http://localhost:{port}/analyze\n")
        
        httpd.serve_forever()
    except OSError as e:
        print(f"❌ Failed to bind to port {port}: {e}")
        print("   Try killing the process using this port or use --port <new_port>")
        if zmq_socket: zmq_socket.close()
        if context: context.term()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        if 'httpd' in locals(): httpd.server_close()
        if zmq_socket: zmq_socket.close()
        if context: context.term()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Log Producer")
    parser.add_argument('--port', type=int, default=HTTP_PORT, help='HTTP Port to listen on')
    args = parser.parse_args()
    
    run_server(args.port)
