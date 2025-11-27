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

3. **Transform raw logs to dataset format**:
```bash
# File mode (convert entire log file)
python pipeline/transform.py --input data/raw/access.txt --output pipeline/modsec_wp_dataset.csv

# Streaming mode (read from stdin, write to stdout)
cat access.txt | python pipeline/transform.py --stream
```

### Testing
```bash
# Run main application entry point
python main.py

# The system doesn't have automated tests currently
```

## Architecture

### Data Flow
1. **Log Producer** (`logprod.py`) → ZeroMQ Publisher
2. **Dashboard** (`idsdashboard.py`) → ZeroMQ Subscriber + ML Inference + TUI
3. **Feature Extraction** (`security_model.py`) → Converts raw logs to 10-dimensional features
4. **Neural Network** → 2-class classifier (Safe/Attack) with 16 hidden units

### Key Components

#### Feature Engineering (`security_model.py:FeatureExtractor`)
Extracts 10 numerical features from each HTTP request:
- Admin area indicators (`wp-admin`, `admin-ajax`)
- Request method (POST vs others)
- URI length (normalized)
- Special characters (XSS indicators: `<`, `>`, `'`, `"`)
- Attack keywords (`base64`, `exec`, `union`)
- HTTP status codes (200 OK, 4xx/5xx errors)

#### Neural Network Architecture (`security_model.py:AttackClassifier`)
- Input: 10 features
- Hidden layer: 16 units with ReLU activation
- Output: 2 units with softmax (Safe/Attack probabilities)
- Training: Binary classification on labeled attack data

#### Real-time Communication
- **Protocol**: ZeroMQ PUB/SUB pattern
- **Host**: localhost
- **Port**: 5555
- **Topic**: "logs"
- **Message Format**: JSON with `features` (list) and `metadata` (dict)

#### Dashboard UI (`idsdashboard.py`)
- **Stats row**: Total traffic counter, attack counter, sparklines
- **Main table**: Real-time log entries with predictions
- **Alerts log**: Attack-specific notifications
- **Threading**: ZeroMQ listener runs in background thread, UI updates via message posting

### Data Sources
- **Raw logs**: Expected in `data/raw/access.txt` (Apache/Nginx combined log format)
- **Processed dataset**: `pipeline/modsec_wp_dataset.csv` with 26-field ModSecurity schema
- **Attack patterns**: Auto-labeled based on signatures in `ATTACK_SIGNATURES`

## Configuration

### Model Configuration (`security_model.py`)
- `INPUT_DIM = 10`: Feature vector size
- `HIDDEN_DIM = 16`: Hidden layer size
- `OUTPUT_DIM = 2`: Binary classification output

### ZeroMQ Configuration
- Host: `localhost` (hardcoded in both producer and dashboard)
- Port: `5555` (configurable via `ZMQ_PORT` constants)
- Topic: `"logs"` for message filtering

### File Locations
- Log source: `data/raw/access.txt`
- Dataset output: `pipeline/modsec_wp_dataset.csv`
- Virtual environment: `.venv/`

## Development Notes

- The system uses `uv` for dependency management (configured in `pyproject.toml`)
- Python 3.12+ is required (specified in `.python-version` and `pyproject.toml`)
- PyTorch uses CUDA 128 binaries for GPU acceleration where available
- The dashboard handles thread-safe UI updates via Textual's `post_message` system
- Log parsing is tolerant of encoding errors and malformed lines
- The producer includes a 0.2s delay between messages for dashboard readability