# 🧪 Prosedur Percobaan & Pengujian Sistem

Dokumen ini berisi panduan langkah demi langkah untuk melakukan pengujian dan validasi sistem ModSec-IDS. Panduan ini mencakup persiapan, pengujian komponen individual, dan simulasi serangan penuh.

## 📋 Prasyarat

Pastikan Anda berada di direktori root proyek dan virtual environment telah aktif:

```bash
cd /home/dzakirm/MLproject/modsec-ids
# Jika menggunakan uv
uv sync
source .venv/bin/activate
```

---

## 1. Pengujian Unit Komponen

Sebelum menjalankan sistem secara utuh, verifikasi bahwa setiap komponen berfungsi dengan baik.

### A. Tes Inferensi Model Semantik
Pastikan model TensorFlow dapat memuat dan memprediksi serangan dari string teks mentah.

1.  Buat skrip tes sederhana (atau gunakan `tools/test_sample.py`):
    ```bash
    uv run python tools/test_sample.py "GET /etc/passwd"
    ```
2.  **Hasil yang diharapkan:** Prediksi harus mendeteksi serangan (misalnya `lfi` atau `directory_traversal`) dengan confidence tinggi.

---

## 2. Simulasi Sistem Penuh (End-to-End) - Mode Cepat (Recommended)

Skenario ini mensimulasikan lingkungan produksi di mana `logprod.py` (atau `test_log_producer.py`) mengirimkan data log ke `idsdashboard.py`.

Kita menggunakan **API Log Producer** untuk memuat model AI yang berat hanya sekali, sehingga producer klien (terminal 3) sangat ringan dan cepat.

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

**Apa yang akan terjadi:**
1.  **Terminal 3 (Producer):** Akan mengirimkan log ke API dan mencetak status.
    *   `🟢 Sent sample #1: /wp-content/themes/style.css...`
    *   `🔴 SQLI Sent sample #2: /login.php?user=' OR '1'='1...`
2.  **Terminal 1 (API Engine):** Akan memproses request dan mencetak hasil inferensi.
    *   `⚡ Analyzed: GET /login.php... -> sqli (99.8%)`
3.  **Terminal 2 (Dashboard):**
    *   **Traffic Sparkline:** Grafik aktivitas akan mulai bergerak.
    *   **Attacks Sparkline:** Grafik serangan akan melonjak merah saat serangan dikirim.
    *   **Alerts Log (Panel Bawah):** Akan muncul pesan peringatan real-time.
    *   **Main Table:** Tabel utama akan terisi dengan detail request (IP asli dataset, bukan localhost).

---

## 3. Skenario Pengujian Spesifik

### A. Uji Serangan Spesifik (Misal: SQL Injection & XSS)
Jika Anda ingin mempresentasikan kemampuan deteksi spesifik:

```bash
# Di Terminal 3
uv run python test_log_producer.py --attack-type sqli --attack-type xss --api-url http://localhost:8000 --delay 0.5
```
Ini akan mengirimkan campuran trafik normal, SQL Injection, dan XSS saja.

### B. Atomic Testing (Uji Manual Satu per Satu)
Sangat berguna untuk demo langsung. Anda bisa mengetik serangan sendiri.

```bash
# Di Terminal 3
uv run python tools/test_sample.py --interactive
```
*   Lalu ketik: `GET /login.php?user=admin' OR 1=1` (Tekan Enter)
*   Lihat hasilnya di layar, dan juga lihat Dashboard (Terminal 2) bereaksi.

---

## 4. Pelatihan Ulang Model (Opsional)

Jika Anda ingin memperbarui model dengan data baru:

1.  **Siapkan Dataset:** Pastikan file CSV ada di `data/raw/Modsec-WP.csv`.
2.  **Jalankan Training:**
    ```bash
    uv run python training/train_tensorflow_working.py --epochs 10 --batch-size 64
    ```
3.  **Verifikasi Output:** Pastikan file baru terbentuk di folder `results/`:
    *   `final_model.keras` (atau `best_model.keras`)
    *   `tokenizer.pkl`
    *   `label_encoder.pkl`
4.  **Restart Sistem:** Matikan dan nyalakan kembali `tools/api_log_producer.py` untuk memuat model baru.

## 🐛 Pemecahan Masalah Umum

*   **Error "Address already in use":**
    *   Port 8000 (HTTP) atau 5555 (ZMQ) sedang dipakai. Matikan proses python lain (`killall python` jika perlu).
*   **Dashboard tidak menampilkan "SEMANTIC":**
    *   Pastikan Anda menjalankan dashboard dengan flag `--semantic`.
*   **IP Address selalu 127.0.0.1 di Dashboard:**
    *   Pastikan Anda menggunakan versi terbaru `test_log_producer.py` dan `api_log_producer.py` yang mendukung metadata forwarding.
    *   Restart `api_log_producer.py`.