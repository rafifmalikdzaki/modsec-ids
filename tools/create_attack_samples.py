#!/usr/bin/env python3
"""
Create attack samples in the ModSecurity format for better model training
"""

import pandas as pd
import random
import numpy as np

def create_attack_samples(csv_path: str, num_samples: int = 1000):
    """Add synthetic attack samples to balance the dataset"""
    print(f"Adding {num_samples} synthetic attack samples to {csv_path}")

    # Load existing data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} existing rows")

    # Create synthetic attack samples
    attack_samples = []

    # SQL Injection samples
    sql_injection_patterns = [
        "' OR '1'=1",
        "' UNION SELECT * FROM users--",
        "'; DROP TABLE users--",
        "admin'/*",
        "' AND '1'='1",
        "1' OR '1'='1",
        "(SELECT * FROM information_schema.tables)",
        "xp_cmdshell('whoami')",
        "'; exec('ping 127.0.0.1')",
    ]

    # XSS samples
    xss_patterns = [
        "<script>alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "'; var x = document.cookie;",
        "<svg onload=alert('XSS')>",
    ]

    # RCE samples
    rce_patterns = [
        ";system('whoami')",
        ";eval('echo hello')",
        ";exec('rm -rf /')",
        "&cmd.exe /c net user",
        "|python -c 'import socket,subprocess,os; socket.connect((ip,port)); subprocess.Popen(cmd)';",
    ]

    # File Inclusion samples
    lfi_patterns = [
        "../../../etc/passwd",
        "../../../etc/shadow",
        "....//....//etc/passwd",
        "wp-config.php",
        "config.php.bak",
        "..\\..\\..\\..\\..\\..\\..\\..\\..\\..\\..",
    ]

    # Create attack samples
    for i in range(num_samples):
        attack_type = random.choice(['sql_injection', 'xss', 'rce', 'lfi'])

        if attack_type == 'sql_injection':
            pattern = random.choice(sql_injection_patterns)
            url = f"/wp-admin/admin-ajax.php?action=test&action={pattern}&id={i+1}"
            method = "POST"
        elif attack_type == 'xss':
            pattern = random.choice(xss_patterns)
            url = f"/search?q=<script>alert('{i}')</script>&search={i+1}"
            method = "GET"
        elif attack_type == 'rce':
            pattern = random.choice(rce_patterns)
            url = f"/admin.php?cmd={pattern}"
            method = "POST"
        elif attack_type == 'lfi':
            pattern = random.choice(lfi_patterns)
            url = f"/wp-config.php?file={pattern}"
            method = "POST"
        else:  # Normal sample
            url = f"/wp-content/plugins/bug-library/stylesheet.css?ver={i+1}"
            method = "GET"

        # Create row
        attack_row = {
            'transaction_id': f"ATK{i+10:08}UQ{random.randint(10, 99):Q{random.randint(10, 99)}{random.randint(10, 99)}",
            'event_time': f"2025-11-{random.randint(1, 28):02:00:00 +0000",
            'remote_address': f"192.168.1.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            'request_host': 'ekosakti.ub.ac.id',
            'request_useragent': 'Mozilla/5.0 (compatible; Attack Scanner; +https://example.com)',
            'request_line': f"{method} {url} HTTP/1.1",
            'request_line_method': method,
            'request_line_url': url,
            'request_line_protocol': 'HTTP/1.1',
            'response_protocol': 'HTTP/1.1',
            'response_status': '200',
            'action': 'blocked' if attack_type != 'normal' else 'allowed',
            'action_phase': '2',
            'action_message': f"Warning. Pattern match '{pattern}' at REQUEST_HEADERS:Host" if attack_type != 'normal' else '-',
            'message_type': 'outbound' if attack_type != 'normal' else '-',
            'message_description': f"Attempted {attack_type} attack" if attack_type != 'normal' else 'Normal request',
            'message_rule_id': '999001' if attack_type != 'normal' else '-',
            'message_rule_file': 'REQUEST-920-PROTOCOL-ENFORCEMENT.conf' if attack_type != 'normal' else '-',
            'message_msg': f"Detected {attack_type} attempt" if attack_type != 'normal' else 'Normal request',
            'message_severity': 'CRITICAL' if attack_type != 'normal' else 'WARNING',
            'message_accuracy': '10' if attack_type != 'normal' else '0',
            'message_maturity': '9' if attack_type != 'normal' else '0',
            'full_message_line': f'{method} {url} HTTP/1.1" 200 -',
            'request_body': f"action={action}&id={i+1}" if method == 'POST' else '',
            'response_body': '',
            'label': 'malicious' if attack_type != 'normal' else 'normal'
        }

        attack_samples.append(attack_row)

    # Create DataFrame from attack samples
    attack_df = pd.DataFrame(attack_samples)

    # Combine with original data
    combined_df = pd.concat([df, attack_df], ignore_index=True)

    # Save combined dataset
    combined_df.to_csv(csv_path, index=False)

    print(f"✅ Added {len(attack_samples)} attack samples")
    print(f"Total dataset size: {len(combined_df)}")
    print(f"New label distribution: {combined_df['label'].value_counts()}")

    return len(attack_samples)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create attack samples for ModSecurity dataset")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--samples", type=int, default=1000, help="Number of attack samples to add")

    args = parser.parse_args()

    create_attack_samples(args.input, args.samples)