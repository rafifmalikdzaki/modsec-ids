import sys
import os
import http.server
import socketserver
import json
import zmq
import numpy as np
import argparse
import time

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ⭐ CHANGE: Import Hybrid LSTM Detector
try:
    from detectors.hybrid_lstm_detector import HybridLSTMDetector
    MODEL_AVAILABLE = True
    print("✅ Hybrid LSTM Model module loaded successfully")
except ImportError as e:
    print(f"⚠️  Failed to import Hybrid LSTM model: {e}")
    MODEL_AVAILABLE = False

# Configuration
ZMQ_PORT = 5555
HTTP_PORT = 8000

# Global objects
detector = None
zmq_socket = None

class LogRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_status = {
                'status': 'healthy',
                'model_loaded': detector is not None,
                'zmq_active': zmq_socket is not None,
                'model_type': 'HybridLSTM'
            }
            
            self.wfile.write(json.dumps(health_status).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/analyze':
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            log_line = post_data.decode('utf-8').strip()
            
            # Metadata from headers (optional)
            client_metadata = {
                'timestamp': time.time(),
                'client_ip': self.client_address[0]
            }
            
            # 1. Analyze log
            result = self.analyze_log(log_line)
            
            # 2. Send HTTP response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            
            # 3. Publish to Dashboard (ZeroMQ)
            self.publish_to_dashboard(log_line, result, client_metadata)
            
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logging"""
        pass

    def analyze_log(self, log_line):
        """
        Run inference on the log line
        
        Returns:
            {
                'log': str,
                'prediction': str,
                'confidence': float,
                'is_attack': bool,
                'probabilities': Dict[str, float],
                'inference_time_ms': float
            }
        """
        if not detector:
            return {"error": "Model not loaded"}
        
        try:
            start_time = time.time()
            
            # ⭐ CHANGE: Use new detector API
            label, conf, probs = detector.predict(log_line)
            
            inference_time = (time.time() - start_time) * 1000

            print(f"⚡ Analyzed: {log_line[:60]}... -> {label} ({conf:.2%})")

            return {
                "log": log_line,
                "prediction": label,
                "confidence": float(conf),
                "is_attack": label.lower() not in ['normal', 'clean', 'benign'],
                "probabilities": probs,
                "inference_time_ms": float(f"{inference_time:.2f}")
            }
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return {
                "error": str(e),
                "log": log_line
            }

    def publish_to_dashboard(self, log_line, result, client_metadata=None):
        """
        Send result to ZeroMQ bus for dashboard
        
        Args:
            log_line: Raw log input
            result: Inference result dict
            client_metadata: Additional metadata
        """
        if not zmq_socket:
            return

        # Create metadata packet
        metadata = {
            'ip': client_metadata.get('client_ip', '127.0.0.1'),
            'port': '0',
            'method': 'API',
            'uri': log_line[:100],
            'status': '200',
            'timestamp': client_metadata.get('timestamp', time.time())
        }
        
        # Add inference results
        if 'error' not in result:
            metadata.update({
                'semantic_prediction': result['prediction'],
                'semantic_confidence': result['confidence'],
                'semantic_probs': result['probabilities'],
                'is_attack': result['is_attack']
            })
        
        # Construct payload
        payload = {
            'features': np.zeros(15).tolist(),  # Dummy features for compatibility
            'metadata': metadata
        }

        try:
            zmq_socket.send_string("logs", flags=zmq.SNDMORE)
            zmq_socket.send_json(payload)
        except Exception as e:
            print(f"⚠️  Failed to publish to ZMQ: {e}")


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def run_server(port=HTTP_PORT):
    """
    Main server initialization and run loop
    """
    global detector, zmq_socket
    
    # 1. Initialize ZeroMQ
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUB)
    try:
        zmq_socket.bind(f"tcp://*:{ZMQ_PORT}")
        print(f"📡 ZMQ Publisher bound to port {ZMQ_PORT}")
    except zmq.error.ZMQError as e:
        print(f"⚠️  Port {ZMQ_PORT} is in use: {e}")
        print("   Dashboard publishing might fail if another producer is running.")

    # 2. Load Hybrid LSTM Model
    print("\n🔄 Loading Hybrid LSTM Model...")
    if MODEL_AVAILABLE:
        try:
            detector = HybridLSTMDetector(models_dir='models')
            print("✅ Hybrid LSTM Model Loaded Successfully!")
            print(f"   Classes: {detector.label_classes}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("   Service will run but cannot predict.")
            detector = None
    else:
        print("❌ Model import failed. Service will run but cannot predict.")

    # 3. Start HTTP Server
    server_address = ('', port)
    try:
        httpd = ReusableTCPServer(server_address, LogRequestHandler)
        
        print(f"\n{'='*70}")
        print(f"🚀 API Producer listening on http://localhost:{port}")
        print(f"{'='*70}")
        print(f"\n📖 Usage:")
        print(f"   Health Check: curl http://localhost:{port}/health")
        print(f"   Analyze Log:  curl -X POST -d 'GET /evil.php' http://localhost:{port}/analyze")
        print(f"\n💡 Examples:")
        print(f"   SQL Injection: curl -X POST -d \"GET /login?id=1' OR '1'='1\" http://localhost:{port}/analyze")
        print(f"   XSS:          curl -X POST -d 'GET /search?q=<script>alert(1)</script>' http://localhost:{port}/analyze")
        print(f"\n{'='*70}\n")
        
        httpd.serve_forever()
        
    except OSError as e:
        print(f"❌ Failed to bind to port {port}: {e}")
        print("   Try killing the process using this port or use --port <new_port>")
        if zmq_socket: 
            zmq_socket.close()
        if context: 
            context.term()
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        if 'httpd' in locals(): 
            httpd.server_close()
        if zmq_socket: 
            zmq_socket.close()
        if context: 
            context.term()
        print("✅ Server stopped gracefully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Log Producer with Hybrid LSTM")
    parser.add_argument('--port', type=int, default=HTTP_PORT, 
                       help='HTTP Port to listen on (default: 8000)')
    args = parser.parse_args()
    
    run_server(args.port)