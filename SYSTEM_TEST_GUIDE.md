# 🧪 ModSec-IDS System Test Guide

This guide provides step-by-step instructions to validate the Real-time Log Processing, Dashboard Visualization, and Attack Simulation features.

## 📋 Prerequisites

Ensure you are in the project root and your virtual environment is active:

```bash
cd /home/dzakirm/MLproject/modsec-ids
# If using uv
uv sync
source .venv/bin/activate
```

---

## 🖥️ Terminal 1: Launch the Dashboard

The dashboard is the central monitoring interface. Start it first so it's ready to receive data.

**Option A: Semantic Mode (Recommended)**
Prioritizes the TensorFlow LSTM model for high-accuracy text analysis.
```bash
uv run python idsdashboard.py --semantic
```

**Option B: Multi-class Mode**
Uses the PyTorch model for 8-class detection based on numerical features.
```bash
uv run python idsdashboard.py --multiclass
```

**Expected Output:**
- A TUI (Text User Interface) should appear.
- "Waiting for logs..." or empty charts.
- Top stats bar: "TOTAL TRAFFIC" and "ATTACKS DETECTED".

---

## 🔧 Terminal 2: Start Log Processing (Real Data)

This component monitors actual log files and runs the inference engine.

**Scenario 1: Continuous Monitoring (Real-time)**
Simulate monitoring a live server log (using the provided sample dataset).
```bash
uv run python logprod.py --input data/raw/access.txt --continuous
```

**Scenario 2: Bulk Processing**
Process a batch of historical logs and exit.
```bash
uv run python logprod.py --input "data/raw/*.txt" --bulk-process
```

**Expected Output (Terminal 2):**
- `🚀 Producer started on port 5555`
- `🧠 Initializing TensorFlow Semantic Model...`
- `🟢 Sent: /normal/path...`
- `🔴 SQLI Sent: /vulnerable.php?id=1' OR 1=1...`

**Expected Output (Dashboard):**
- The table should populate with live entries.
- Attack counters should increment.
- "Semantic" column should show specific predictions (e.g., `SQLI`, `XSS`).

---

## 🎮 Terminal 3: Run Test Scenarios (Simulation)

Use the enhanced test producer to generate specific attack patterns and verify dashboard visualization.

**Scenario 1: Comprehensive System Test**
Cycles through all 8 attack types + normal traffic to verify dashboard handling.
```bash
uv run python test_log_producer.py --comprehensive-test --delay 0.5
```

**Scenario 2: Specific Attack Simulation**
Test how the system handles a burst of specific attacks (e.g., SQL Injection and XSS).
```bash
uv run python test_log_producer.py --attack-type sqli --attack-type xss --delay 0.2
```

**Scenario 3: Performance Benchmark**
Test the throughput of the messaging pipeline.
```bash
uv run python test_log_producer.py --benchmark --iterations 5000
```

**Expected Output (Dashboard):**
- **Scenario 1:** You should see a colorful variety of alerts in the "Alerts" panel (Red for RCE/SQLi, Orange for XSS, Yellow for Traversal).
- **Scenario 2:** Only SQLi and XSS entries should appear.
- **Scenario 3:** The "Total Traffic" counter should spin up rapidly.

---

## 🐛 Troubleshooting

- **Dashboard not showing data?**
  - Ensure `logprod.py` or `test_log_producer.py` is running in a separate terminal.
  - Check that all scripts are using port `5555`.

- **"Model not found" error?**
  - Ensure `results/final_model.keras` exists. If not, run:
    ```bash
    uv run python train_tensorflow_working.py --epochs 1
    ```

- **ImportError: No module named 'tensorflow'?**
  - Ensure you are running with `uv run` or that your virtual environment has `tensorflow` installed.
