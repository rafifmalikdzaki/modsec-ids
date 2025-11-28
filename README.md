# ModSecurity IDS: Real-Time Semantic Attack Detection

A machine learning-based Intrusion Detection System (IDS) that detects and classifies web attacks in real-time using ModSecurity logs. This system employs a hybrid approach, utilizing both Feature-Based Feed-Forward Neural Networks (FFNN) and Semantic LSTM (Long Short-Term Memory) models to identify 8 distinct types of web attacks.

## 🚀 Key Features

*   **Real-Time Detection**: Streams and analyzes log data instantly using ZeroMQ.
*   **Hybrid AI Models**:
    *   **Semantic LSTM**: Analyzes the *meaning* of payloads (SQLi, XSS, etc.) using Natural Language Processing (NLP) with specialized tokenization for web attacks.
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
*   **Atomic Testing**: Tools to instantly test individual attack payloads against the model.

## 📦 Architecture

The system consists of these main components:

1.  **API Producer (`tools/api_log_producer.py`)**: The core inference engine. It loads the heavy AI model *once* and provides an HTTP API for analysis. It publishes results to the Dashboard.
2.  **Dashboard (`idsdashboard.py`)**: Subscribes to the inference stream, aggregates statistics, and displays real-time alerts in a rich TUI.
3.  **Log Producers**: Clients that read logs and send them to the API Producer.
    *   `logprod.py`: Reads real log files (e.g., `access.txt`).
    *   `test_log_producer.py`: Generates synthetic test traffic.
4.  **Training Pipeline**: Scripts in `training/` to preprocess data and train the models with improved special character handling.

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

To run the full system efficiently, use **three terminal windows**:

### Terminal 1: Start the Inference Engine
This loads the model and listens for requests.
```bash
uv run python tools/api_log_producer.py
```

### Terminal 2: Start the Dashboard
This launches the monitoring interface.
```bash
uv run python idsdashboard.py --semantic
```

### Terminal 3: Feed Data (Logs or Tests)
Send traffic to the engine.

*   **Option A: Monitor Real Logs**
    ```bash
    uv run python logprod.py --input data/raw/access.txt --continuous --api-url http://localhost:8000
    ```

*   **Option B: Run Full Test Suite**
    ```bash
    uv run python test_log_producer.py --full-test --api-url http://localhost:8000 --delay 0.1
    ```

*   **Option C: Atomic Testing (Single Attacks)**
    ```bash
    uv run python tools/test_sample.py "GET /login.php?user=' OR 1=1"
    ```
    *(See `ATOMIC_TESTING_GUIDE.md` for more)*

## 🧠 Model Training

If you need to retrain the models on new data:

### Semantic LSTM Model (TensorFlow)
The most accurate model for payload analysis.
```bash
# Train the semantic model
uv run python training/train_tensorflow_working.py --input data/raw/Modsec-WP.csv --epochs 20
```

## 📂 Project Structure

*   `detectors/`: Core detection logic (`tensorflow_semantic_inference.py`, `security_model.py`)
*   `training/`: Model training scripts.
*   `tools/`: Utility scripts.
    *   `api_log_producer.py`: HTTP Inference Server.
    *   `test_sample.py`: Atomic testing tool.
*   `data/`: Contains raw logs and processed datasets.
*   `models/` & `results/`: Stores trained model artifacts (`.h5`, `.keras`, `.pkl`).
*   `idsdashboard.py`: Main dashboard application (TUI).
*   `logprod.py`: Log streaming service.
*   `test_log_producer.py`: Test data generator.

## 🤝 Contributors
- **DzakirM** - *Initial Work*