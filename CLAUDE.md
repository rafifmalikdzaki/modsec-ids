# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ModSecurity Intrusion Detection System (IDS) utilizing a **Character-Level LSTM** for semantic attack detection.

- **Inference Engine**: `tools/api_log_producer.py` (Loads TensorFlow model, serves HTTP API).
- **Dashboard**: `idsdashboard.py` (TUI for visualization).
- **Training**: `training/train_tensorflow_working.py` (Handles balancing, char-level tokenization, and training).

## Development Commands

### Environment Setup
```bash
uv sync
source .venv/bin/activate
```

### Running the System

1.  **Start Inference Engine (Terminal 1)**:
    ```bash
    uv run python tools/api_log_producer.py
    ```

2.  **Start Dashboard (Terminal 2)**:
    ```bash
    uv run python idsdashboard.py --semantic
    ```

3.  **Feed Data (Terminal 3)**:
    ```bash
    # Full Test Suite
    uv run python test_log_producer.py --full-test --api-url http://localhost:8000 --delay 0.1

    # Atomic Test
    uv run python tools/test_sample.py "GET /login.php?user=' OR 1=1"
    ```

### Model Training

**Train Semantic LSTM (Character-Level)**:
```bash
uv run python training/train_tensorflow_working.py --input data/raw/Modsec-WP.csv --epochs 20
```
*   This script automatically handles class imbalance (oversampling/undersampling).
*   It uses character-level tokenization (robust to obfuscation).

## Architecture & File Structure

### Core Logic (`detectors/`)
- `tensorflow_semantic_inference.py`: Loads `.keras` model and performs inference. Contains `_preprocess_text` logic (must match training).
- `security_model.py`: Feature extraction logic.

### Training (`training/`)
- `train_tensorflow_working.py`: The definitive training script. Defines model architecture (Embedding -> Conv1D -> LSTM).

### Tools (`tools/`)
- `api_log_producer.py`: Persistent inference server.
- `test_sample.py`: CLI tool for quick tests.

### Data Flow
1.  Log source (`logprod.py` / `test_log_producer.py`) sends raw log line via HTTP POST to `api_log_producer.py`.
2.  `api_log_producer.py` preprocesses (URL decode -> Lowercase), tokenizes (Char-level), and runs inference.
3.  Result is published via ZeroMQ to `idsdashboard.py` and returned via HTTP to the caller.

## Configuration

- **Model Storage**: `results/final_model.keras` (or `best_model.keras`), `results/tokenizer.pkl`.
- **Max Sequence Length**: 1000 (Character level).
- **Vocab Size**: ~200 (ASCII characters).
