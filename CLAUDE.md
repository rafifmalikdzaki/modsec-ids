# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ModSecurity Intrusion Detection System (IDS) that performs real-time web attack detection using machine learning. The system consists of:

- **ML-based Attack Detection**: PyTorch neural network classifier (`security_model.py`) that analyzes HTTP request features
- **Real-time Dashboard**: Textual-based TUI (`idsdashboard.py`) for monitoring attacks live
- **Log Producer**: ZeroMQ-based log streaming (`logprod.py`) that processes web server logs
- **Data Pipeline**: Log transformation utilities (`pipeline/transform.py`) for preprocessing

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running the System

The system requires two separate processes running simultaneously:

1. **Start the log producer** (Terminal 1):
```bash
python logprod.py
```
- Reads from `data/raw/access.txt`
- Publishes parsed log data via ZeroMQ on port 5555
- Waits for log file to exist if not found

2. **Start the dashboard** (Terminal 2):
```bash
python idsdashboard.py
```
- Subscribes to ZeroMQ on port 5555
- Performs real-time inference using the attack classifier
- Displays traffic and attack statistics in a TUI interface

### Model Training and Development

3. **Train different model variants**:
```bash
# Simple FFNN classifier
python train_simple.py

# Enhanced LSTM classifier
python train_enhanced_ids.py

# LSTM semantic classifier with multi-class attack types
python train_semantic_model.py [--input data/dataset.csv] [--epochs 10] [--batch-size 32]
```

4. **Data processing and utilities**:
```bash
# Transform raw logs to dataset format
python pipeline/transform.py --input data/raw/access.txt --output pipeline/modsec_wp_dataset.csv

# Streaming mode (read from stdin, write to stdout)
cat access.txt | python pipeline/transform.py --stream

# Convert Excel data to CSV
python convert_excel_to_csv.py

# Create synthetic attack samples for testing
python create_attack_samples.py
```

### Testing
```bash
# Run main application entry point
python main.py

# Test log producer independently
python test_log_producer.py
```

### Model Selection Priority
The system automatically selects models in this priority order:
1. **LSTM Semantic Model** (`enhanced_security_model.LSTMMultiClassSemanticAttackClassifier`) - 8-class classifier
2. **Enhanced LSTM Model** (`enhanced_security_model.EnhancedAttackClassifier`) - Binary classifier
3. **Simple FFNN Model** (`security_model.AttackClassifier`) - Basic 10-feature classifier

## Architecture

### Data Flow
1. **Log Producer** (`logprod.py`) → ZeroMQ Publisher
2. **Dashboard** (`idsdashboard.py`) → ZeroMQ Subscriber + ML Inference + TUI
3. **Feature Extraction** → Multiple model types (Simple FFNN, Enhanced LSTM, LSTM Semantic)
4. **Neural Networks** → Binary (Safe/Attack) and Multi-class (8 attack types) classifiers

### Model Architecture Hierarchy

The system supports three progressively sophisticated model types:

#### 1. Simple FFNN Classifier (`security_model.py`)
- **Features**: 10 numerical features extracted from HTTP requests
- **Architecture**: Input(10) → Hidden(16, ReLU) → Output(2, softmax)
- **Output**: Binary classification (Safe/Attack probabilities)

#### 2. Enhanced LSTM Classifier (`enhanced_security_model.py`)
- **Features**: Sequential analysis with attention mechanism
- **Architecture**: LSTM layers with attention over request sequences
- **Output**: Binary classification with confidence scores

#### 3. LSTM Semantic Classifier (`enhanced_security_model.py`)
- **Features**: Full semantic understanding of HTTP headers, payloads, and responses
- **Architecture**: Multi-layer LSTM with embedding and dropout
- **Output**: 8-class classification (Normal + 7 attack types: SQLi, Brute Force, LFI, XSS, RCE, Directory Traversal, Command Injection)
- **Classes**: `{'normal': 0, 'sqli': 1, 'bruteforce': 2, 'lfi': 3, 'xss': 4, 'rce': 5, 'directory_traversal': 6, 'command_injection': 7}`

### Key Components

#### Feature Engineering (`security_model.py:FeatureExtractor`)
Supports three extraction modes:
- **Basic**: 10 numerical features from HTTP requests
- **Enhanced**: Sequential features with temporal context
- **Semantic**: Full text analysis of headers, payloads, and responses

Basic features extracted:
- Admin area indicators (`wp-admin`, `admin-ajax`)
- Request method (POST vs others)
- URI length (normalized)
- Special characters (XSS indicators: `<`, `>`, `'`, `"`)
- Attack keywords (`base64`, `exec`, `union`)
- HTTP status codes (200 OK, 4xx/5xx errors)

#### Model Training Infrastructure
- **Simple FFNN**: `train_simple.py` - Basic binary classifier
- **Enhanced LSTM**: `train_enhanced_ids.py` - Sequential binary classifier
- **LSTM Semantic**: `train_semantic_model.py` - Multi-class semantic analyzer
- **Data Preprocessing**: `data_preprocessor.py` - Shared preprocessing utilities

#### Real-time Communication
- **Protocol**: ZeroMQ PUB/SUB pattern
- **Host**: localhost
- **Port**: 5555
- **Topic**: "logs"
- **Message Format**: JSON with `features` (list) and `metadata` (dict)

#### Dashboard UI (`idsdashboard.py`)
- **Multi-class Support**: Handles both binary and multi-class predictions
- **Stats row**: Total traffic counter, attack counter, sparklines
- **Main table**: Real-time log entries with predictions and attack type labels
- **Alerts log**: Attack-specific notifications with color coding by attack type
- **Threading**: ZeroMQ listener runs in background thread, UI updates via message posting

### Data Sources
- **Raw logs**: Expected in `data/raw/access.txt` (Apache/Nginx combined log format)
- **Processed dataset**: `pipeline/modsec_wp_dataset.csv` with 26-field ModSecurity schema
- **Attack patterns**: Auto-labeled based on signatures in `ATTACK_SIGNATURES`
- **Excel data**: Convertible via `convert_excel_to_csv.py`
- **Synthetic data**: Generated via `create_attack_samples.py` for testing

## Configuration

### Model Configuration

#### Simple FFNN (`security_model.py`)
- `INPUT_DIM = 10`: Basic feature vector size
- `HIDDEN_DIM = 16`: Hidden layer size
- `OUTPUT_DIM = 2`: Binary classification output

#### Enhanced LSTM (`enhanced_security_model.py`)
- `LSTM_MODEL_PATH = "models/lstm_attack_classifier.pth"`
- `PREPROCESSOR_PATH = "data/processed/modsec_processed_preprocessor.pkl"`

#### LSTM Semantic (`enhanced_security_model.py`)
- `LSTM_SEMANTIC_MODEL_PATH = "models/lstm_semantic_classifier.pth"`
- `SEMANTIC_PREPROCESSOR_PATH = "data/processed/lstm_semantic_preprocessor.pkl"`
- `VOCAB_SIZE = 10000`: Token vocabulary size
- `EMBEDDING_DIM = 128`: Word embedding dimensions
- `HIDDEN_DIM = 256`: LSTM hidden state size
- `OUTPUT_DIM = 8`: Multi-class output (7 attack types + normal)
- `MAX_SEQ_LENGTH = 500`: Maximum sequence length for semantic analysis

### ZeroMQ Configuration
- Host: `localhost` (hardcoded in both producer and dashboard)
- Port: `5555` (configurable via `ZMQ_PORT` constants)
- Topic: `"logs"` for message filtering

### File Locations
- **Log source**: `data/raw/access.txt`
- **Dataset output**: `pipeline/modsec_wp_dataset.csv`
- **Model storage**: `models/` directory for trained classifiers
- **Preprocessors**: `data/processed/` for fitted preprocessing objects
- **Virtual environment**: `.venv/`

## Development Notes

- **Dependency Management**: Uses `uv` for fast dependency resolution (configured in `pyproject.toml`)
- **Python Version**: Requires Python 3.12+ (specified in `.python-version` and `pyproject.toml`)
- **GPU Acceleration**: PyTorch uses CUDA 128 binaries for GPU acceleration where available
- **Thread Safety**: Dashboard handles thread-safe UI updates via Textual's `post_message` system
- **Error Handling**: Log parsing is tolerant of encoding errors and malformed lines
- **Performance**: Producer includes 0.2s delay between messages for dashboard readability
- **Model Fallback**: Automatic model selection with graceful degradation from semantic → enhanced → simple models
- **Attack Classification**: Supports both binary detection and multi-class attack type identification with color-coded visualization