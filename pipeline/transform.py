import re
import csv
import urllib.parse
import uuid
import random
import string

# Configuration
INPUT_LOG_FILE = '../data/raw/access.txt'
OUTPUT_CSV_FILE = 'modsec_wp_dataset.csv'

# Regex to parse standard Combined Log Format
# Format: IP - - [Date] "Method URI Protocol" Status Bytes "Referer" "UserAgent"
LOG_PATTERN = re.compile(
    r'(?P<ip>[\d\.]+) - - \[(?P<timestamp>.*?)\] "(?P<method>\w+) (?P<uri>.*?) (?P<protocol>HTTP\/[\d\.]+)" (?P<status>\d+) (?P<size>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
)

# Attack Signatures to auto-label the dataset
ATTACK_SIGNATURES = [
    'upg_datatable', 'exec:', 'base64', 'chmod', 'wget', 'curl', 'cd /var/www'
]

def generate_transaction_id():
    """Generates a random ID similar to ModSec format (e.g., Zo1Eg5ZuOSGaV0fnQWE8gwAAAAA)"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=27))

def extract_host(referer):
    """Attempts to extract hostname from referer, defaults to the known host."""
    if referer and referer != "-":
        try:
            parsed = urllib.parse.urlparse(referer)
            if parsed.netloc:
                return parsed.netloc
        except:
            pass
    return "ekosakti.ub.ac.id" # Fallback based on log content

def transform_to_modsec_schema(log_line):
    match = LOG_PATTERN.match(log_line)
    if not match:
        return None
    
    data = match.groupdict()
    raw_uri = data['uri']
    decoded_uri = urllib.parse.unquote(raw_uri)
    
    # Determine Label (Simulated Detection)
    is_malicious = any(sig in decoded_uri for sig in ATTACK_SIGNATURES)
    label = "malicious" if is_malicious else "normal"
    
    # Message/Alert Simulation (Only for malicious rows to mimic WAF logs)
    if is_malicious:
        action_msg = "Warning. Pattern match found for RCE attempt"
        rule_id = "999001" # Simulated Rule ID for Custom RCE
        msg_severity = "CRITICAL"
        msg = "Detected Shell Command Execution attempt"
    else:
        action_msg = "-"
        rule_id = "-"
        msg_severity = "-"
        msg = "-"

    # Map to Target Schema
    row = {
        'transaction_id': generate_transaction_id(),
        'event_time': data['timestamp'],
        'remote_address': data['ip'],
        'request_host': extract_host(data['referer']),
        'request_useragent': data['user_agent'],
        'request_line': f"{data['method']} {data['uri']} {data['protocol']}",
        'request_line_method': data['method'],
        'request_line_url': data['uri'],
        'request_line_protocol': data['protocol'],
        'response_protocol': data['protocol'], # Usually matches request
        'response_status': data['status'],
        
        # ModSecurity specific fields (Empty/Placeholder since source is access.txt)
        'action': 'Blocked' if is_malicious else '-', # Assuming WAF would block attacks
        'action_phase': '2' if is_malicious else '-', # Phase 2 is usually Request Body/URI
        'action_message': action_msg,
        'message_type': 'outbound' if is_malicious else '-',
        'message_description': '-',
        'message_rule_id': rule_id,
        'message_rule_file': 'custom_rules.conf' if is_malicious else '-',
        'message_msg': msg,
        'message_severity': msg_severity,
        'message_accuracy': '10' if is_malicious else '-',
        'message_maturity': '9' if is_malicious else '-',
        'full_message_line': '-', # Complex log line omitted
        'request_body': '-', # Not present in access logs
        'response_body': '-', # Not present in access logs
        'label': label
    }
    
    return row

def main():
    print(f"Reading from {INPUT_LOG_FILE}...")
    extracted_rows = []
    
    try:
        with open(INPUT_LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    row = transform_to_modsec_schema(line)
                    if row:
                        extracted_rows.append(row)
    except FileNotFoundError:
        print(f"Error: {INPUT_LOG_FILE} not found. Please upload the file.")
        return

    if not extracted_rows:
        print("No valid log lines found.")
        return

    # Define the exact column order from the reference dataset
    fieldnames = [
        'transaction_id', 'event_time', 'remote_address', 'request_host', 
        'request_useragent', 'request_line', 'request_line_method', 
        'request_line_url', 'request_line_protocol', 'response_protocol', 
        'response_status', 'action', 'action_phase', 'action_message', 
        'message_type', 'message_description', 'message_rule_id', 
        'message_rule_file', 'message_msg', 'message_severity', 
        'message_accuracy', 'message_maturity', 'full_message_line', 
        'request_body', 'response_body', 'label'
    ]

    # Write to CSV
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_rows)
        
    print(f"Successfully converted {len(extracted_rows)} logs to ModSec-WP dataset format.")
    print(f"Output saved to: {OUTPUT_CSV_FILE}")
    print("\nSummary of Labels:")
    malicious_count = sum(1 for r in extracted_rows if r['label'] == 'malicious')
    print(f"Normal: {len(extracted_rows) - malicious_count}")
    print(f"Malicious: {malicious_count}")

if __name__ == "__main__":
    main()
