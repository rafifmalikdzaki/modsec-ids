# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ModSecurity Intrusion Detection System (IDS) that performs real-time web attack detection using machine learning. The system consists of:

- **ML-based Attack Detection**: Semantic LSTM (`detectors.tensorflow_semantic_inference`) and PyTorch classifiers (`detectors.security_model`)
- **Real-time Dashboard**: Textual-based TUI (`idsdashboard.py`) for monitoring attacks live
- **Log Producer**: ZeroMQ-based log streaming (`logprod.py`) that processes web server logs
- **Training Pipeline**: Scripts in `training/` for model development

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running the System

The system requires three separate processes running simultaneously:

1.  **Start the Inference Engine (API Producer)** (Terminal 1):
    ```bash
    uv run python tools/api_log_producer.py
    ```
    *   Loads model once, serves predictions via HTTP.

2.  **Start the Dashboard** (Terminal 2):
    ```bash
    uv run python idsdashboard.py --semantic
    ```

3.  **Start Data Feed** (Terminal 3):
    ```bash
    # For testing/demo (simulated traffic)
    uv run python test_log_producer.py --full-test --api-url http://localhost:8000 --delay 0.1

    # For real log monitoring
    uv run python logprod.py --input data/raw/access.txt --continuous --api-url http://localhost:8000
    ```

### Model Training and Development

3. **Train the Semantic LSTM Model (Primary)**:
```bash
uv run python training/train_tensorflow_working.py --input data/raw/Modsec-WP.csv --epochs 20
```

4. **Data processing and utilities**:
```bash
# Transform raw logs to dataset format
uv run python pipeline/transform.py --input data/raw/access.txt --output pipeline/modsec_wp_dataset.csv

# Create synthetic attack samples for testing
uv run python tools/create_attack_samples.py
```

### Testing
```bash
# Run main application entry point
uv run python main.py

# Atomic testing
uv run python tools/test_sample.py --interactive
```

### Model Selection Priority
The system automatically selects models in this priority order:
1. **Semantic LSTM Model** (TensorFlow) - 8-class semantic analysis of payloads.
2. **Multi-class LSTM Model** (PyTorch) - 8-class classification based on features.
3. **Binary Classifier** (PyTorch) - Simple safe/attack classification.

## Architecture

### Directory Structure
- `detectors/`: Core detection logic (`tensorflow_semantic_inference.py`, `security_model.py`)
- `training/`: Model training scripts (`train_tensorflow_working.py`)
- `tools/`: Utility scripts (`create_attack_samples.py`, `api_log_producer.py`)
- `idsdashboard.py`: TUI Dashboard
- `logprod.py`: Real-time log producer
- `test_log_producer.py`: Test data generator

### Data Flow
1. **Log Producer** (`logprod.py`) → HTTP Request → **API Producer** (`tools/api_log_producer.py`)
2. **API Producer** (Inference) → ZeroMQ Publisher
3. **Dashboard** (`idsdashboard.py`) → ZeroMQ Subscriber + TUI Visualization

### Key Components

#### Feature Engineering (`detectors.security_model`)
Supports extraction of numerical features and semantic text preprocessing.

#### Real-time Communication
- **Protocol**: ZeroMQ PUB/SUB pattern
- **Host**: localhost
- **Port**: 5555
- **Topic**: "logs"

#### Dashboard UI (`idsdashboard.py`)
- **Thread Safety**: Dashboard handles thread-safe UI updates via Textual's `post_message` system
- **Visuals**: Sparklines for traffic/attacks, detailed log table, alert panel.

### Data Sources
- **Raw logs**: Expected in `data/raw/access.txt`
- **Processed dataset**: `data/processed/`
- **Model storage**: `models/` and `results/`

## Configuration

### File Locations
- **Log source**: `data/raw/access.txt`
- **Model storage**: `results/final_model.keras` (TensorFlow), `models/*.pth` (PyTorch)
- **Preprocessors**: `results/tokenizer.pkl`, `results/label_encoder.pkl`

## Development Notes

- **Dependency Management**: Uses `uv` (configured in `pyproject.toml`)
- **Python Version**: Requires Python 3.12+
- **GPU Acceleration**: TensorFlow/PyTorch use GPU where available
- **Error Handling**: Robust error handling in producers to prevent crash on malformed logs