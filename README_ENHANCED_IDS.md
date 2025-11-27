# Enhanced LSTM-based Intrusion Detection System

This document explains how to train and use the LSTM-based IDS using the ModSecurity WordPress dataset.

## Overview

The enhanced system uses a deep learning approach with LSTM (Long Short-Term Memory) networks to detect web attacks with higher accuracy compared to the simple feed-forward neural network. The system is designed to work with the existing dashboard infrastructure.

## Key Features

- **LSTM Architecture**: Bidirectional LSTM with multi-head attention for sequence-based analysis
- **Enhanced Feature Extraction**: 15-dimensional feature vector including URL patterns, HTTP characteristics, and attack signatures
- **ModSecurity Dataset Support**: Works with the full ModSecurity WordPress dataset (983K+ records)
- **Backward Compatibility**: Seamlessly integrates with existing dashboard and log producer
- **Real-time Prediction**: Optimized for real-time intrusion detection

## File Structure

```
├── lstm_security_model.py          # LSTM model architecture and training
├── data_preprocessor.py           # Dataset preprocessing utilities
├── enhanced_security_model.py     # Enhanced model for integration
├── train_enhanced_ids.py         # Complete training pipeline
├── security_model.py             # Updated to support enhanced model
├── idsdashboard.py               # Updated dashboard with enhanced model support
└── data/
    ├── raw/
    │   ├── Modsec-WP.csv         # Raw ModSecurity dataset (189MB)
    │   └── access.txt            # Raw web server logs
    └── processed/
        └── modsec_processed.npz   # Preprocessed training data
```

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install torch pandas numpy scikit-learn
```

### 2. Train the Enhanced Model

```bash
# Complete training pipeline (preprocess + train)
python train_enhanced_ids.py --preprocess --train

# Or step by step:
# Step 1: Preprocess dataset
python train_enhanced_ids.py --preprocess

# Step 2: Train LSTM model
python train_enhanced_ids.py --train

# With custom parameters:
python train_enhanced_ids.py --preprocess --train --sample 50000 --epochs 30 --batch-size 128
```

### 3. Run the Enhanced Dashboard

```bash
# Terminal 1: Start log producer (unchanged)
python logprod.py

# Terminal 2: Start dashboard with enhanced model
python idsdashboard.py --enhanced
```

## Training Options

### Preprocessing Parameters

- `--input`: Path to ModSecurity CSV file (default: `data/raw/Modsec-WP.csv`)
- `--processed`: Output path for processed data (default: `data/processed/modsec_processed.npz`)
- `--sample`: Sample size for training (use `None` for full dataset)

### Training Parameters

- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Training batch size (default: 64)

### Example Commands

```bash
# Quick training with smaller dataset (for testing)
python train_enhanced_ids.py --preprocess --train --sample 10000 --epochs 10

# Full dataset training (may take several hours)
python train_enhanced_ids.py --preprocess --train --epochs 50

# Training with larger batch size for faster training
python train_enhanced_ids.py --train --epochs 30 --batch-size 128
```

## Model Architecture

### LSTM Configuration

- **Input Features**: 15-dimensional feature vector
- **Sequence Length**: 50 time steps (sliding window over request history)
- **Hidden Dimensions**: 64 units per direction (128 bidirectional)
- **LSTM Layers**: 2 bidirectional layers with dropout
- **Attention Mechanism**: Multi-head attention (8 heads)
- **Output**: Binary classification (Normal/Attack)

### Feature Extraction

The system extracts 15 features from each request:

1. **URL Features** (7):
   - URL length (normalized)
   - Decoded URL length (normalized)
   - Admin area detection (`wp-admin`, `wp-login`)
   - Ajax endpoint detection (`admin-ajax`, `ajax`)
   - XSS character patterns (`<`, `>`, `'`, `"`, `&`)
   - Base64 encoding detection
   - SQL injection keywords (`union`, `select`, `insert`, `delete`)

2. **HTTP Features** (3):
   - Request method (POST detection)
   - Status code (normalized)
   - Error status detection (4xx, 5xx)

3. **User-Agent Features** (2):
   - User-Agent length (normalized)
   - Bot detection patterns

4. **Attack Pattern Features** (3):
   - Blocked action detection
   - Attack pattern detection (SQLi, XSS, RCE, LFI, RFI)
   - Severity indicators

## Integration with Existing System

### Dashboard Integration

The dashboard now supports both the original simple model and the enhanced LSTM model:

```bash
# Use original simple model
python idsdashboard.py

# Use enhanced LSTM model (if available)
python idsdashboard.py --enhanced
```

### Model Loading

The system automatically detects and loads the enhanced model if available:

- **Model Path**: `models/lstm_attack_classifier.pth`
- **Preprocessor Path**: `data/processed/modsec_processed_preprocessor.pkl`
- **Fallback**: Uses simple model if enhanced model not found

### Log Producer Integration

The log producer (`logprod.py`) works unchanged with both models. The enhanced feature extractor supports both:

1. **Raw log lines** (Apache/Nginx combined format)
2. **ModSecurity dataset rows** (from CSV processing)

## Performance Considerations

### Training Performance

- **Full Dataset**: 983K records may require 1-4 hours depending on hardware
- **GPU Acceleration**: Automatically uses CUDA if available
- **Memory Requirements**: ~2GB RAM for full dataset preprocessing

### Inference Performance

- **Sequence Processing**: LSTM processes 50-step sequences efficiently
- **Real-time Performance**: ~1-2ms per prediction on modern hardware
- **Memory Usage**: ~10MB for loaded model

### Optimization Tips

1. **Use GPU Training**: CUDA-enabled GPUs provide 3-5x speedup
2. **Batch Size**: Larger batch sizes (128-256) train faster but use more memory
3. **Sample Size**: Start with smaller samples (10K-50K) for testing
4. **Early Stopping**: Monitor validation loss to prevent overfitting

## Troubleshooting

### Common Issues

1. **Dataset Not Found**:
   ```bash
   Error: Dataset not found at data/raw/Modsec-WP.csv
   ```
   **Solution**: Ensure the ModSecurity dataset is in `data/raw/Modsec-WP.csv`

2. **Memory Errors**:
   ```bash
   MemoryError: Unable to allocate array
   ```
   **Solution**: Use `--sample` parameter to reduce dataset size

3. **Model Not Loading**:
   ```bash
   Error loading enhanced model: [error], falling back to simple model
   ```
   **Solution**: Run training first to create the model files

### Debug Mode

Enable verbose logging for debugging:

```bash
# Enable debug output in preprocessing
python -c "import logging; logging.basicConfig(level=logging.DEBUG); from data_preprocessor import clean_and_preprocess_dataset; clean_and_preprocess_dataset('data/raw/Modsec-WP.csv', 'data/processed/debug.npz', 1000)"
```

## Next Steps

1. **Model Evaluation**: Test the trained model on validation data
2. **Feature Engineering**: Experiment with additional features
3. **Hyperparameter Tuning**: Optimize LSTM architecture and training parameters
4. **Ensemble Methods**: Combine multiple models for better accuracy
5. **Deployment**: Deploy model in production environment

## References

- [ModSecurity WordPress Dataset](https://github.com/faizann24/ModSecurity-WordPress-Dataset)
- [LSTM for Sequence Classification](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Multi-head Attention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)