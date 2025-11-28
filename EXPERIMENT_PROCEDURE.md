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

1.  Buat skrip tes sederhana (atau gunakan `test_inference.py` jika ada):
    ```python
    from tensorflow_semantic_inference import TensorFlowSemanticInference
    
    detector = TensorFlowSemanticInference()
    test_log = "GET /wp-admin/admin-ajax.php?action=revslider_show_image&img=../wp-config.php HTTP/1.1"
    label, conf, _ = detector.predict(test_log)
    
    print(f"Input: {test_log}")
    print(f"Prediksi: {label} (Conf: {conf:.2%})")
    ```
2.  Jalankan skrip tersebut. **Hasil yang diharapkan:** Prediksi harus mendeteksi serangan (misalnya `lfi` atau `directory_traversal`) dengan confidence tinggi.

### B. Tes Log Producer (Simulasi Data)
Verifikasi bahwa producer dapat membaca dataset tes dan mengirimkannya via ZeroMQ.

```bash
uv run python test_log_producer.py --stats-only
```
**Hasil yang diharapkan:** Menampilkan statistik dataset tes, termasuk distribusi kelas serangan.

---

## 2. Simulasi Sistem Penuh (End-to-End)

Skenario ini mensimulasikan lingkungan produksi di mana `logprod.py` (atau `test_log_producer.py`) mengirimkan data log ke `idsdashboard.py`.

### Langkah 1: Jalankan Dashboard (Terminal 1)
Dashboard bertindak sebagai "monitor" yang menerima data. Kita gunakan mode `--semantic` untuk visualisasi terbaik.

```bash
uv run python idsdashboard.py --semantic
```
*Status:* Dashboard akan muncul dengan status "Waiting for logs..." atau grafik kosong.

### Langkah 2: Jalankan Simulasi Serangan (Terminal 2)
Kita akan mengirimkan seluruh dataset tes (`--full-test`) untuk melihat bagaimana sistem menangani berbagai jenis serangan secara berurutan.

```bash
uv run python test_log_producer.py --full-test --delay 0.1
```

**Apa yang akan terjadi:**
1.  **Terminal 2 (Producer):** Akan mulai mencetak log yang dikirim, misalnya:
    *   `🟢 Sent sample #1: /wp-content/themes/style.css...`
    *   `🔴 SQLI Sent sample #2: /login.php?user=' OR '1'='1...`
2.  **Terminal 1 (Dashboard):**
    *   **Traffic Sparkline:** Grafik aktivitas akan mulai bergerak.
    *   **Attacks Sparkline:** Grafik serangan akan melonjak merah saat serangan dikirim.
    *   **Alerts Log (Panel Bawah):** Akan muncul pesan peringatan real-time:
        *   `[Jam] ⚠️ SEMANTIC SQLI (conf: 0.99) from 192.168.x.x...`
        *   `[Jam] ⚠️ SEMANTIC XSS (conf: 0.98) from 192.168.x.x...`
    *   **Main Table:** Tabel utama akan terisi dengan detail request (IP, Method, URI, Semantic Prediction).

---

## 3. Skenario Pengujian Spesifik

### A. Uji Serangan Spesifik (Misal: SQL Injection & XSS)
Jika Anda ingin mempresentasikan kemampuan deteksi spesifik:

```bash
# Di Terminal 2
uv run python test_log_producer.py --attack-type sqli --attack-type xss --delay 0.5
```
Ini akan mengirimkan campuran trafik normal, SQL Injection, dan XSS saja. Memudahkan untuk melihat perbedaan warna dan label di dashboard.

### B. Benchmark Performa (Stress Test)
Menguji seberapa cepat sistem dapat memproses pesan.

```bash
# Di Terminal 2
uv run python test_log_producer.py --benchmark --iterations 5000
```
Perhatikan apakah dashboard tetap responsif atau mengalami lag. Ini menguji efisiensi ZeroMQ dan UI rendering.

---

## 4. Pelatihan Ulang Model (Opsional)

Jika Anda ingin memperbarui model dengan data baru:

1.  **Siapkan Dataset:** Pastikan file CSV ada di `data/raw/Modsec-WP.csv`.
2.  **Jalankan Training:**
    ```bash
    uv run python train_tensorflow_working.py --epochs 10 --batch-size 64
    ```
3.  **Verifikasi Output:** Pastikan file baru terbentuk di folder `results/`:
    *   `final_model.keras`
    *   `tokenizer.pkl`
    *   `label_encoder.pkl`
4.  **Restart Sistem:** Matikan dan nyalakan kembali `idsdashboard.py` dan producer untuk memuat model baru.

## 🐛 Pemecahan Masalah Umum

*   **Error "Port 5555 in use":**
    *   Ada proses producer lain yang masih berjalan. Matikan terminal lain atau cari proses dengan `lsof -i :5555` dan matikan (`kill -9 <PID>`).
*   **Dashboard tidak menampilkan "SEMANTIC":**
    *   Pastikan Anda menjalankan dashboard dengan flag `--semantic`.
    *   Pastikan producer juga mengirimkan prediksi semantik (default pada `test_log_producer.py` dan `logprod.py` terbaru).
*   **Label "Unknown" atau Angka di Dashboard:**
    *   Ini menandakan ketidakcocokan antara `label_encoder.pkl` dan kode inferensi. (Sudah diperbaiki di versi terakhir, pastikan Anda menggunakan kode terbaru).
