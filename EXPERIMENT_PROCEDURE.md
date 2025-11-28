# 🧪 Prosedur Percobaan & Pengujian Sistem

Dokumen ini berisi panduan langkah demi langkah untuk melakukan pengujian dan validasi sistem ModSec-IDS. Sistem ini menggunakan **Character-Level LSTM** untuk mendeteksi serangan web secara semantik.

## 📋 Prasyarat

Pastikan Anda berada di direktori root proyek dan virtual environment telah aktif:

```bash
cd /home/dzakirm/MLproject/modsec-ids
# Jika menggunakan uv
uv sync
source .venv/bin/activate
```

---

## 1. Simulasi Sistem Penuh (End-to-End) - Mode Cepat (Recommended)

Skenario ini mensimulasikan lingkungan produksi. Kita menggunakan **API Log Producer** sebagai otak AI terpusat.

### Langkah 1: Jalankan Inference Engine (Terminal 1)
Ini memuat model TensorFlow dan menunggu permintaan.
```bash
uv run python tools/api_log_producer.py
```
*Status:* Tunggu hingga muncul "🚀 API Producer listening on http://localhost:8000".

### Langkah 2: Jalankan Dashboard (Terminal 2)
Dashboard bertindak sebagai "monitor" yang menerima data dari engine.
```bash
uv run python idsdashboard.py --semantic
```
*Status:* Dashboard akan muncul dengan status "Waiting for logs...".

### Langkah 3: Jalankan Simulasi Serangan (Terminal 3)
Kita akan mengirimkan seluruh dataset tes (`--full-test`) ke engine API (`--api-url`).

```bash
uv run python test_log_producer.py --full-test --api-url http://localhost:8000 --delay 0.1
```

**Hasil yang Diharapkan:**
1.  **Traffic Real-time:** Dashboard akan menampilkan lonjakan trafik dan serangan.
2.  **Klasifikasi Akurat:** Serangan SQLi akan terdeteksi sebagai SQLi, XSS sebagai XSS, dll.
3.  **Kecepatan:** Sistem harus responsif karena model hanya dimuat satu kali.

---

## 2. Skenario Pengujian Spesifik (Atomic Testing)

Berguna untuk demo atau validasi serangan spesifik.

### A. Uji SQL Injection (SQLi)
```bash
uv run python tools/test_sample.py "GET /login.php?user=admin' OR '1'='1"
```
*Expected Result:* 🔴 ATTACK (SQLI)

### B. Uji Cross-Site Scripting (XSS)
```bash
uv run python tools/test_sample.py "GET /search?q=<script>alert(1)</script>"
```
*Expected Result:* 🔴 ATTACK (XSS)

### C. Uji Directory Traversal / LFI
```bash
uv run python tools/test_sample.py "GET /index.php?page=../../../../etc/passwd"
```
*Expected Result:* 🔴 ATTACK (DIRECTORY_TRAVERSAL) atau (LFI)

### D. Uji Normal Traffic
```bash
uv run python tools/test_sample.py "GET /wp-content/themes/style.css"
```
*Expected Result:* 🟢 SAFE (NORMAL)

---

## 3. Pelatihan Ulang Model (Opsional)

Jika Anda ingin melatih ulang model dengan data baru atau parameter baru.

1.  **Dataset:** Pastikan `data/raw/Modsec-WP.csv` tersedia.
2.  **Jalankan Training:**
    ```bash
    uv run python training/train_tensorflow_working.py --input data/raw/Modsec-WP.csv --epochs 20
    ```
    *Script ini akan otomatis menyeimbangkan dataset (oversampling/undersampling) dan melatih model Character-Level.*

3.  **Restart Sistem:** Matikan dan nyalakan kembali `tools/api_log_producer.py` untuk memuat model baru.

## 📂 Lokasi Preprocessing & Logika

Jika Anda perlu memodifikasi logika:

*   **Preprocessing (Training):** `training/train_tensorflow_working.py` (fungsi `preprocess_text` dan `prepare_data`).
*   **Preprocessing (Inference):** `detectors/tensorflow_semantic_inference.py` (metode `_preprocess_text`).
*   **Model Architecture:** `training/train_tensorflow_working.py` (fungsi `create_model`).
