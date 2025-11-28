import sys
import zmq
import torch
import datetime
import joblib
import numpy as np
from threading import Thread

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, DataTable, Log, Static, Sparkline, Digits
from textual.message import Message
from textual import work
from rich.text import Text

from detectors.security_model import AttackClassifier
from detectors.multiclass_security_model import MultiClassAttackClassifier

# ZMQ Configuration
ZMQ_HOST = "localhost"
ZMQ_PORT = 5555

class NewLogEntry(Message):
    """Message sent when a new log arrives."""
    def __init__(self, data, prediction_result):
        self.data = data
        self.prediction_result = prediction_result
        self.is_attack = prediction_result['is_attack']
        self.confidence = prediction_result['confidence']
        self.predicted_class = prediction_result['predicted_class']
        super().__init__()

class IdsDashboard(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #stats-row {
        height: 10;
        dock: top;
        margin-bottom: 1;
    }

    .stat-box {
        border: solid green;
        background: $surface;
        height: 100%;
        width: 1fr;
        padding: 1;
    }
    
    #attack-box {
        border: solid red;
    }

    Sparkline {
        width: 100%;
        height: 3;
        margin-top: 1;
        color: $accent;
    }
    
    .attack-spark {
        color: red;
    }

    #main-logs {
        height: 60%;
        border: solid $accent;
    }

    #alerts-log {
        height: 1fr;
        border: solid red;
        background: $surface-darken-1;
    }
    """

    TITLE = "ModSec-IDS: Real-Time Attack Monitor"
    SUB_TITLE = "Powered by PyTorch & Textual"

    def __init__(self, use_multiclass=False, use_semantic=False):
        super().__init__()
        # Initialize Model (semantic > multi-class > binary priority)
        self.use_multiclass = use_multiclass
        self.use_semantic = use_semantic

        # Try semantic model first
        if use_semantic:
            self.model = AttackClassifier(use_semantic=True)
            if (hasattr(self.model, 'use_semantic') and self.model.use_semantic) or \
               (hasattr(self.model, 'semantic_model') and self.model.semantic_model):
                print("Using LSTM semantic classifier")
                self.use_semantic = True
            else:
                print("Semantic model not available, falling back to multi-class...")
                self.use_semantic = False

        # Try multi-class model if semantic failed
        if not self.use_semantic and use_multiclass:
            self.model = MultiClassAttackClassifier()
            if not self.model.is_loaded:
                print("Falling back to binary classifier...")
                self.model = AttackClassifier()
                self.use_multiclass = False
            else:
                self.use_multiclass = True

        # Fallback to simple binary model
        if not self.use_semantic and not self.use_multiclass:
            self.model = AttackClassifier()

        if hasattr(self.model, 'model'):
            self.model.model.eval() # Set to evaluation mode

        # Stats
        self.total_requests = 0
        self.total_attacks = 0
        self.attack_history = [0] * 60 # Last 60 updates

    def compose(self) -> ComposeResult:
        yield Header()
        
        with Horizontal(id="stats-row"):
            with Vertical(classes="stat-box"):
                yield Static("TOTAL TRAFFIC", classes="label")
                yield Digits("0", id="counter-total")
                yield Static("Traffic Volume", classes="label")
                yield Sparkline(self.attack_history, summary_function="mean", id="spark-traffic")
            
            with Vertical(classes="stat-box", id="attack-box"):
                yield Static("ATTACKS DETECTED", classes="label")
                yield Digits("0", id="counter-attacks")
                yield Static("Attack Density (Last 60 events)", classes="label")
                yield Sparkline(self.attack_history, summary_function="max", classes="attack-spark", id="spark-attacks")

        yield DataTable(id="main-logs", zebra_stripes=True)
        yield Log(id="alerts-log", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        table = self.query_one(DataTable)
        if self.use_semantic:
            table.add_columns("Time", "IP", "Port", "Method", "Status", "URI", "Semantic", "Confidence")
        elif self.use_multiclass:
            table.add_columns("Time", "IP", "Port", "Method", "Status", "URI", "Attack Type", "Confidence")
        else:
            table.add_columns("Time", "IP", "Port", "Method", "Status", "URI", "Prediction", "Conf")
        table.fixed_columns = 1

        # Start ZMQ Listener in a background thread
        self.start_zmq_listener()

    @work(exclusive=True, thread=True)
    def start_zmq_listener(self):
        """Listens to ZeroMQ and posts messages to the UI thread."""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
        socket.subscribe("logs")

        while True:
            try:
                # Non-blocking check or standard recv
                topic = socket.recv_string()
                payload = socket.recv_json()
                
                features = payload['features']
                meta = payload['metadata']

                # Debug output
                print(f"📥 Received: IP={meta.get('ip', 'unknown')}, Port={meta.get('port', 'unknown')}, Features={len(features)}")

                # Inference
                try:
                    # PRIORITY 1: Pre-calculated Semantic Prediction (from logprod.py / TensorFlow)
                    if self.use_semantic and 'semantic_prediction' in meta:
                        pred_class = meta['semantic_prediction']
                        # Use the producer's is_attack flag if present, otherwise derive from class
                        is_attack = meta.get('is_attack', pred_class != 'normal')
                        conf = float(meta.get('semantic_confidence', 1.0))
                        
                        prediction_result = {
                            'is_attack': is_attack,
                            'confidence': conf,
                            'predicted_class': pred_class,
                            'all_probabilities': meta.get('semantic_probs', {})
                        }
                        # Debug output
                        print(f"🧠 Semantic result (Remote): {pred_class} (conf: {conf:.2f})")

                    # PRIORITY 2: Local Multi-class Model
                    elif self.use_multiclass:
                        # Multi-class prediction
                        prediction_result = self.model.predict(features)
                        if not prediction_result:  # Model failed to load
                            print("❌ Multi-class model prediction failed, falling back to binary")
                            # Binary prediction fallback
                            with torch.no_grad():
                                input_tensor = torch.tensor([features])
                                output = self.model(input_tensor)
                                probabilities = output.tolist()[0] # [Prob_Safe, Prob_Attack]

                                is_attack = probabilities[1] > 0.5
                                confidence = probabilities[1] if is_attack else probabilities[0]
                                prediction_result = {
                                    'is_attack': is_attack,
                                    'confidence': confidence,
                                    'predicted_class': 'attack' if is_attack else 'normal',
                                    'all_probabilities': {'safe': probabilities[0], 'attack': probabilities[1]}
                                }
                        # Debug output
                        print(f"🔍 Multi-class result: {prediction_result.get('predicted_class', 'unknown')} (conf: {prediction_result.get('confidence', 0):.2f})")

                    # PRIORITY 3: Local Binary Model
                    else:
                        # Binary prediction
                        with torch.no_grad():
                            input_tensor = torch.tensor([features])
                            output = self.model(input_tensor)
                            probabilities = output.tolist()[0] # [Prob_Safe, Prob_Attack]

                            is_attack = probabilities[1] > 0.5
                            confidence = probabilities[1] if is_attack else probabilities[0]
                            prediction_result = {
                                'is_attack': is_attack,
                                'confidence': confidence,
                                'predicted_class': 'attack' if is_attack else 'normal',
                                'all_probabilities': {'safe': probabilities[0], 'attack': probabilities[1]}
                            }
                        # Debug output
                        print(f"🔍 Binary result: {prediction_result.get('predicted_class', 'unknown')} (conf: {prediction_result.get('confidence', 0):.2f})")

                    # Update UI (Must be done via post_message to be thread-safe)
                    self.post_message(NewLogEntry(meta, prediction_result))

                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    # Log error to internal log but don't crash
                    continue

            except Exception as e:
                # Log error to internal log but don't crash
                continue

    def on_new_log_entry(self, message: NewLogEntry) -> None:
        """Handler for new log messages."""
        data = message.data
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        self.total_requests += 1
        self.query_one("#counter-total", Digits).update(str(self.total_requests))

        # Update Traffic Sparkline (just generic activity)
        spark_traffic = self.query_one("#spark-traffic", Sparkline)
        self.attack_history.append(1) # Just activity
        if len(self.attack_history) > 60: self.attack_history.pop(0)
        spark_traffic.data = self.attack_history

        # Colorize Row based on prediction
        if message.is_attack:
            self.total_attacks += 1
            self.query_one("#counter-attacks", Digits).update(str(self.total_attacks))

            # Update Attack Sparkline (High value for attack)
            spark_attack = self.query_one("#spark-attacks", Sparkline)
            spark_data = spark_attack.data
            spark_data.append(10) # Spike
            if len(spark_data) > 60: spark_data.pop(0)
            spark_attack.data = spark_data

            # Handle semantic vs multi-class vs binary display
            if self.use_semantic:
                semantic_prediction = data.get('semantic_prediction', 'attack').upper()
                color = 'red' if 'ATTACK' in semantic_prediction else 'orange'
                pred_text = Text(semantic_prediction, style=f"bold {color}")

                # Add to Alerts Log (Bottom Panel) with semantic prediction
                log_widget = self.query_one("#alerts-log", Log)
                semantic_conf = data.get('semantic_confidence', 0.0)
                log_widget.write_line(f"[{timestamp}] ⚠️  SEMANTIC {semantic_prediction} (conf: {semantic_conf:.2f}) from {data.get('ip', 'unknown')} | {data.get('uri', 'unknown')}")
            elif self.use_multiclass:
                attack_colors = self.model.get_attack_colors()
                color = attack_colors.get(message.predicted_class, 'red')
                pred_text = Text(message.predicted_class.upper(), style=f"bold {color}")

                # Add to Alerts Log (Bottom Panel) with specific attack type
                log_widget = self.query_one("#alerts-log", Log)
                log_widget.write_line(f"[{timestamp}] ⚠️  {message.predicted_class.upper()} DETECTED from {data.get('ip', 'unknown')}:{data.get('port', 'unknown')} | {data.get('uri', 'unknown')}")
            else:
                pred_text = Text("ATTACK", style="bold red")

                # Add to Alerts Log (Bottom Panel)
                log_widget = self.query_one("#alerts-log", Log)
                log_widget.write_line(f"[{timestamp}] ⚠️  ATTACK DETECTED from {data.get('ip', 'unknown')}:{data.get('port', 'unknown')} | {data.get('uri', 'unknown')}")
        else:
            if self.use_semantic:
                semantic_prediction = data.get('semantic_prediction', 'safe').upper()
                pred_text = Text(semantic_prediction, style="bold green")
            elif self.use_multiclass:
                pred_text = Text(message.predicted_class.upper(), style="bold green")
            else:
                pred_text = Text("SAFE", style="green")

            # Decay attack sparkline
            spark_attack = self.query_one("#spark-attacks", Sparkline)
            spark_data = spark_attack.data
            spark_data.append(0)
            if len(spark_data) > 60: spark_data.pop(0)
            spark_attack.data = spark_data

        # Add to Main Table
        table = self.query_one(DataTable)

        # Truncate URI for display
        uri_display = (data.get('uri', '')[:40] + '..') if len(data.get('uri', '')) > 40 else data.get('uri', '')

        # Add row with appropriate columns for semantic vs multi-class vs binary
        if self.use_semantic:
            table.add_row(
                timestamp,
                data.get('ip', 'unknown'),           # IP column
                str(data.get('port', 'unknown')),   # Port column
                data.get('method', 'unknown'),
                str(data.get('status', 'unknown')),
                uri_display,
                pred_text,  # Semantic column
                f"{data.get('semantic_confidence', message.confidence):.2f}"  # Semantic confidence column
            )
        elif self.use_multiclass:
            table.add_row(
                timestamp,
                data.get('ip', 'unknown'),           # IP column
                str(data.get('port', 'unknown')),   # Port column
                data.get('method', 'unknown'),
                str(data.get('status', 'unknown')),
                uri_display,
                pred_text,  # Attack Type column
                f"{message.confidence:.2f}"  # Confidence column
            )
        else:
            table.add_row(
                timestamp,
                data.get('ip', 'unknown'),           # IP column
                str(data.get('port', 'unknown')),   # Port column
                data.get('method', 'unknown'),
                str(data.get('status', 'unknown')),
                uri_display,
                pred_text,  # Prediction column
                f"{message.confidence:.2f}"
            )

        # Auto-scroll table
        table.scroll_end(animate=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ModSecurity IDS Dashboard")
    parser.add_argument("--multiclass", action="store_true", help="Use multi-class attack classifier")
    parser.add_argument("--semantic", action="store_true", help="Use LSTM semantic model for text analysis")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced LSTM model (deprecated, use --multiclass)")

    args = parser.parse_args()

    # Priority: semantic > multiclass > enhanced > simple
    use_semantic = args.semantic
    use_multiclass = args.multiclass or args.enhanced

    app = IdsDashboard(use_multiclass=use_multiclass, use_semantic=use_semantic)
    app.run()
