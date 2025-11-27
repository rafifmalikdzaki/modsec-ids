#!/usr/bin/env python3
"""
Clean Excel file and convert to clean CSV format
"""

import pandas as pd
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_excel_to_csv(excel_path: str, csv_path: str) -> bool:
    """Clean problematic Excel file and convert to clean CSV"""
    try:
        logger.info(f"Loading Excel file: {excel_path}")

        # Read Excel file
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} rows from Excel file")

        # Check available columns
        logger.info(f"Excel columns: {list(df.columns)}")

        # Focus on essential ModSecurity columns
        required_columns = [
            'transaction_id', 'event_time', 'remote_address', 'request_host',
            'request_useragent', 'request_line', 'request_line_method', 'request_line_url',
            'request_line_protocol', 'response_protocol', 'response_status', 'action', 'action_phase',
            'action_message', 'message_type', 'message_description', 'message_rule_id',
            'message_rule_file', 'message_msg', 'message_severity', 'message_accuracy',
            'message_maturity', 'full_message_line', 'request_body', 'response_body', 'label'
        ]

        # Filter to required columns only
        available_columns = [col for col in required_columns if col in df.columns]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {available_columns}")
            logger.info("Skipping rows with missing data...")
        else:
            logger.info("All required columns present")

        # Create clean dataframe with only required columns
        clean_df = df[available_columns].copy()

        # Clean URL field - remove HTML and problematic characters
        if 'request_line_url' in clean_df.columns:
            logger.info("Cleaning request_line_url field...")

            # Convert to string if needed
            clean_df['request_line_url'] = clean_df['request_line_url'].astype(str)

            # Remove rows with HTML/JavaScript indicators
            html_indicators = [
                '<!DOCTYPE', '<html', '<script', '<javascript:', '<function(', 'alert(',
                '<iframe', '<object', '<embed', '<applet', '<meta', '<link', '<style'
            ]

            # Count rows before cleaning
            before_count = len(clean_df)

            # Filter out HTML content
            mask = clean_df['request_line_url'].str.contains('|'.join(html_indicators), case=False)
            clean_df = clean_df[~mask]  # Inverse of mask

            # Count after cleaning
            after_count = len(clean_df)

            logger.info(f"URL field cleaning: {before_count} -> {after_count} rows ({before_count - after_count} removed)")

            # Remove rows with empty critical fields
            for col in ['request_line_url', 'request_line_method', 'response_status']:
                clean_df = clean_df[clean_df[col] != '']

        # Clean label column - focus on binary classification
        if 'label' in clean_df.columns:
            logger.info("Processing label field...")
            # Convert labels to standardized format
            clean_df['label'] = clean_df['label'].astype(str).str.lower().str.strip()

            # Map various label formats to binary classification
            attack_indicators = [
                'malicious', 'attack', 'blocked', 'deny', 'critical', 'warning',
                'sql injection', 'sqli', 'xss', 'cross-site scripting', 'directory traversal',
                'lfi', 'rfi', 'rce', 'remote code execution'
            ]

            def normalize_label(label):
                label_str = str(label).lower().strip()
                return 1 if any(indicator in label_str for indicator in attack_indicators) else 0

            clean_df['binary_label'] = clean_df['label'].apply(normalize_label)

            # Count label distribution
            label_counts = clean_df['binary_label'].value_counts()
            logger.info(f"Label distribution: {dict(label_counts)}")

        # Log basic statistics
        logger.info(f"Cleaned dataset shape: {clean_df.shape}")
        logger.info(f"Target columns: {list(clean_df.columns)}")

        # Save cleaned CSV
        clean_df.to_csv(csv_path, index=False)
        logger.info(f"Cleaned dataset saved to: {csv_path}")

        return True

    except Exception as e:
        logger.error(f"Error cleaning Excel file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python clean_excel_and_convert.py <input_excel_path> <output_csv_path>")
        sys.exit(1)

    excel_path = sys.argv[1]
    csv_path = sys.argv[2]

    print(f"Converting {excel_path} -> {csv_path}")

    success = clean_excel_to_csv(excel_path, csv_path)

    if success:
        print("✅ Conversion completed successfully!")
        print(f"📊 Cleaned CSV: {csv_path}")
    else:
        print("❌ Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()