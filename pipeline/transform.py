import re
import csv
import urllib.parse
import random
import string
import sys
import argparse
import signal

# Configuration
DEFAULT_INPUT_FILE = '../data/raw/access.txt'
DEFAULT_OUTPUT_FILE = 'modsec_wp_dataset.csv'

# Regex to parse standard Combined Log Format
LOG_PATTERN = re.compile(
    r'(?P<ip>[\d\.]+) - - \[(?P<timestamp>.*?)\] "(?P<method>\w+) (?P<uri>.*?) (?P<protocol>HTTP\/[\d\.]+)" (?P<status>\d+) (?P<size>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
)

# Attack Signatures to auto-label the dataset
ATTACK_SIGNATURES = [
    'upg_datatable', 'exec:', 'base64', 'chmod', 'wget', 'curl', 'cd /var/www'
]

# The exact 26 fields required by your dataset schema
FIELDNAMES = [
    'transaction_id', 'event_time', 'remote_address', 'request_host', 
    'request_useragent', 'request_line', 'request_line_method', 
    'request_line_url', 'request_line_protocol', 'response_protocol', 
    'response_status', 'action', 'action_phase', 'action_message', 
    'message_type', 'message_description', 'message_rule_id', 
    'message_rule_file', 'message_msg', 'message_severity', 
    'message_accuracy', 'message_maturity', 'full_message_line', 
    'request_body', 'response_body', 'label'
]

def handle_sigint(signum, frame):
    """Handles Ctrl+C gracefully during streaming."""
    sys.exit(0)

def generate_transaction_id():
    """Generates a random ID similar to ModSec format."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=27))

def extract_host(referer):
    """Attempts to extract hostname from referer."""
    if referer and referer != "-":
        try:
            parsed = urllib.parse.urlparse(referer)
            if parsed.netloc:
                return parsed.netloc
        except:
            pass
    return "ekosakti.ub.ac.id" # Fallback

def transform_line(log_line):
    """Parses a raw log line and maps it to the 26-column schema."""
    match = LOG_PATTERN.match(log_line)
    if not match:
        return None
    
    data = match.groupdict()
    raw_uri = data['uri']
    decoded_uri = urllib.parse.unquote(raw_uri)
    
    # Determine Label
    is_malicious = any(sig in decoded_uri for sig in ATTACK_SIGNATURES)
    label = "malicious" if is_malicious else "normal"
    
    # Simulate WAF Alert Data for malicious entries
    if is_malicious:
        action_msg = "Warning. Pattern match found for RCE attempt"
        rule_id = "999001"
        msg_severity = "CRITICAL"
        msg = "Detected Shell Command Execution attempt"
        action = "Blocked"
        phase = "2"
        type_ = "outbound"
    else:
        action_msg = "-"
        rule_id = "-"
        msg_severity = "-"
        msg = "-"
        action = "-"
        phase = "-"
        type_ = "-"

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
        'response_protocol': data['protocol'],
        'response_status': data['status'],
        'action': action,
        'action_phase': phase,
        'action_message': action_msg,
        'message_type': type_,
        'message_description': '-',
        'message_rule_id': rule_id,
        'message_rule_file': 'custom_rules.conf' if is_malicious else '-',
        'message_msg': msg,
        'message_severity': msg_severity,
        'message_accuracy': '10' if is_malicious else '-',
        'message_maturity': '9' if is_malicious else '-',
        'full_message_line': '-', 
        'request_body': '-', 
        'response_body': '-', 
        'label': label
    }
    return row

def main():
    parser = argparse.ArgumentParser(description="Log to ModSec-WP Dataset Converter")
    parser.add_argument('--stream', action='store_true', help="Read from Stdin and write to Stdout (Realtime mode)")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE, help="Input log file (if not streaming)")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE, help="Output CSV file (if not streaming)")
    args = parser.parse_args()

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, handle_sigint)

    if args.stream:
        # --- STREAMING MODE ---
        # Reads from pipe, writes to pipe. No headers usually needed for streaming unless requested,
        # but standard CSV streams often include headers once or rely on external schema.
        # Here we write the header once at startup.
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
        writer.writeheader()
        
        try:
            for line in sys.stdin:
                if line.strip():
                    row = transform_line(line)
                    if row:
                        writer.writerow(row)
                        sys.stdout.flush() # Ensure immediate output
        except BrokenPipeError:
            # Handle case where output pipe is closed (e.g. piping to `head`)
            sys.stderr.close()
            sys.exit(0)

    else:
        # --- FILE MODE ---
        print(f"Reading from {args.input}...")
        try:
            with open(args.input, 'r', encoding='utf-8', errors='ignore') as f_in, \
                 open(args.output, 'w', newline='', encoding='utf-8') as f_out:
                
                writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
                writer.writeheader()
                
                count = 0
                for line in f_in:
                    if line.strip():
                        row = transform_line(line)
                        if row:
                            writer.writerow(row)
                            count += 1
            
            print(f"Success! Converted {count} logs.")
            print(f"Output saved to: {args.output}")
            
        except FileNotFoundError:
            print(f"Error: {args.input} not found.")

if __name__ == "__main__":
    main()
