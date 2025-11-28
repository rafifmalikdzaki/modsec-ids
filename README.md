# ModSecurity IDS: Real-Time Semantic Attack Detection

A machine learning-based Intrusion Detection System (IDS) that detects and classifies web attacks in real-time using ModSecurity logs. This system employs a hybrid approach, utilizing both Feature-Based Feed-Forward Neural Networks (FFNN) and Semantic LSTM (Long Short-Term Memory) models to identify 8 distinct types of web attacks.

## 🚀 Key Features

*   **Real-Time Detection**: Streams and analyzes log data instantly using ZeroMQ.
*   **Hybrid AI Models**:
    *   **Semantic LSTM**: Analyzes the *meaning* of payloads (SQLi, XSS, etc.) using Natural Language Processing (NLP).
    *   **Feature-Based FFNN**: Analyzes statistical features (length, special characters, etc.).
*   **Multi-Class Classification**: Detects 8 specific classes:
    *   `Normal` (Safe Traffic)
    *   `SQL Injection (SQLi)`
    *   `Cross-Site Scripting (XSS)`
    *   `Local File Inclusion (LFI)`
    *   `Remote File Inclusion (RFI)`
    *   `Remote Code Execution (RCE)`
    *   `Directory Traversal`
    *   `Command Injection`
    *   `Brute Force`
*   **Interactive Dashboard**: A terminal-based UI (TUI) for monitoring traffic, visualizing attack trends, and viewing detailed alerts.

## 📦 Architecture

The system consists of three main components:

1.  **Log Producer (`logprod.py`)**: Reads web server logs (e.g., Apache/Nginx), performs initial feature extraction, and publishes data via ZeroMQ. It also runs the TensorFlow Semantic Model for deep payload analysis.
2.  **Dashboard (`idsdashboard.py`)**: Subscribes to the log stream, aggregates statistics, and displays real-time alerts in a rich TUI. It can also run a secondary PyTorch model for verification.
3.  **Training Pipeline**: A suite of scripts to preprocess data and train the models.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Recommended for dependency management)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd modsec-ids
    ```
2.  **Install dependencies:**
    ```bash
    uv sync
    source .venv/bin/activate
    ```

## 🚦 Running the System

To run the full system, you need two terminal windows:

### Terminal 1: Start the Log Producer
This script simulates a live web server log stream.
```bash
# Run with comprehensive test data simulation
uv run python test_log_producer.py --full-test --delay 0.2
```
*Alternatively, to monitor a real file:*
```bash
uv run python logprod.py --input data/raw/access.txt --continuous
```

### Terminal 2: Start the Dashboard
This launches the monitoring interface.
```bash
uv run python idsdashboard.py --semantic
```
*   `--semantic`: Enables the advanced LSTM-based text analysis display.
*   `--multiclass`: Uses the statistical feature-based classifier.

## 🧠 Model Training

If you need to retrain the models on new data:

### 1. Semantic LSTM Model (TensorFlow)
The most accurate model for payload analysis.
```bash
# Train the semantic model
uv run python train_tensorflow_working.py --input data/raw/Modsec-WP.csv --epochs 20
```

### 2. Feature-Based Model (PyTorch)
A lightweight backup model.
```bash
# Preprocess data and train
uv run python train_enhanced_ids.py --preprocess --train
```

## 📂 Project Structure

*   `idsdashboard.py`: Main dashboard application (TUI).
*   `logprod.py`: Log streaming service (Real-time producer).
*   `test_log_producer.py`: Test tool for generating synthetic/replay traffic.
*   `train_tensorflow_working.py`: Training script for the Semantic LSTM model.
*   `train_enhanced_ids.py`: Training script for the Feature-based model.
*   `tensorflow_semantic_inference.py`: Inference engine for the trained TensorFlow model.
*   `data/`: Contains raw logs and processed datasets.
*   `models/` & `results/`: Stores trained model artifacts (`.h5`, `.keras`, `.pkl`).

## 🤝 Contributors
- **DzakirM** - *Initial Work*
