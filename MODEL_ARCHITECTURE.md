# 🧠 Model Architecture: Character-Aware CNN-LSTM with BPE

This document details the architecture of the deep learning model used in the ModSecurity IDS, which has been designed for robust semantic analysis of web attack payloads.

## 🚀 Overview

The model is a **Character-Aware Convolutional Neural Network (CNN) followed by a Bidirectional Long Short-Short Term Memory (Bi-LSTM) network**, using **Byte-Pair Encoding (BPE)** for tokenization. This hybrid approach combines the strengths of subword tokenization, local feature extraction (CNN), and sequential context understanding (LSTM) to detect complex attack patterns.

## 🧱 Architecture Details

The model is built using TensorFlow/Keras and consists of the following layers:

1.  **Input Layer**:
    *   Takes sequences of BPE token IDs.
    *   `input_length`: `MAX_SEQ_LENGTH` (currently `256` tokens).

2.  **Embedding Layer**:
    *   `tf.keras.layers.Embedding(vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH)`
    *   Converts each BPE token ID into a dense vector representation.
    *   `vocab_size`: `10000` (learned BPE vocabulary).
    *   `EMBEDDING_DIM`: `64` (dimensionality of the token embeddings).

3.  **Convolutional Layer (CNN for local pattern extraction)**:
    *   `tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')`
    *   Applies 1D convolutions over the embedded sequences. This layer is crucial for identifying local patterns and n-grams of subwords/characters, which are highly indicative of attack signatures (e.g., `union`, `select`, `<script>`).
    *   `filters=64`: Number of feature maps (patterns to detect).
    *   `kernel_size=5`: Each filter looks at 5 tokens at a time.

4.  **Max Pooling Layer**:
    *   `tf.keras.layers.MaxPooling1D(pool_size=2)`
    *   Downsamples the output of the convolutional layer, reducing dimensionality and making the model more robust to minor shifts in pattern location.

5.  **Bidirectional LSTM Layer (Bi-LSTM for sequential context)**:
    *   `tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HIDDEN_DIM, return_sequences=False, dropout=DROPOUT))`
    *   Processes the sequence in both forward and backward directions, capturing long-range dependencies and contextual information from the locally extracted CNN features.
    *   `HIDDEN_DIM`: `256` (number of units in the LSTM cells).
    *   `dropout=DROPOUT`: `0.4` (regularization to prevent overfitting).
    *   `return_sequences=False`: Only returns the output from the last timestep, suitable for classification.

6.  **Dense Layer (Feature Combination)**:
    *   `tf.keras.layers.Dense(64, activation='relu')`
    *   A standard fully connected layer to combine features learned by the LSTM.

7.  **Dropout Layer**:
    *   `tf.keras.layers.Dropout(DROPOUT)`
    *   Further regularization to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

8.  **Output Layer**:
    *   `tf.keras.layers.Dense(num_classes, activation='softmax')`
    *   Produces probability scores for each attack class.
    *   `num_classes`: The number of attack types (currently 8-9, dynamically adjusted).
    *   `softmax`: Ensures output probabilities sum to 1.

## 💡 Key Design Principles

*   **Subword Tokenization (BPE)**: Breaks text into subword units, effectively handling out-of-vocabulary words and common obfuscation techniques (e.g., `select` becomes `sel`, `ect` if `select` isn't in vocabulary, but `sel` and `ect` might be). It also learns specific subwords like `http://` or `union`.
*   **Hybrid CNN-LSTM**: The CNN layer excels at finding local patterns (like specific character sequences in a URL or payload), while the LSTM layer captures the broader context and dependencies across the entire (subword) sequence. This combination is highly effective for security contexts where both local and global patterns matter.
*   **Class Imbalance Handling**: The training pipeline uses dataset balancing (oversampling minorities, undersampling majorities) and class weights during training to ensure the model learns effectively from rare attack types.

## ⚙️ Configuration (Current)

*   `VOCAB_SIZE`: `20000` (BPE vocabulary size).
*   `EMBEDDING_DIM`: `128`.
*   `HIDDEN_DIM`: `512` (LSTM units).
*   `MAX_SEQ_LENGTH`: `256` (maximum sequence length in BPE tokens).
*   `DROPOUT`: `0.4`.
*   `BATCH_SIZE`: `32`.
*   `OPTIMIZER`: Adam.
*   `LOSS`: Categorical Crossentropy.
*   `CLASSES`: Dynamically determined during training, includes 'normal', 'sqli', 'xss', 'lfi', 'rfi', 'rce', 'bruteforce', 'directory_traversal'.

This architecture provides a robust and intelligent Intrusion Detection System capable of identifying a wide range of web attacks with high accuracy and resilience.
