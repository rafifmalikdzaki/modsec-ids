#!/usr/bin/env python3
"""
Convert ModSecurity WordPress Excel file to properly encoded CSV
Handles encoding issues and provides clean CSV for training
"""

import argparse
import pandas as pd
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_excel_to_csv(excel_path: str, csv_path: str, encoding: str = 'utf-8'):
    """Convert Excel file to CSV with proper encoding handling."""

    logger.info(f"Converting Excel file: {excel_path}")
    logger.info(f"Output CSV path: {csv_path}")

    try:
        # Read Excel file (handles encoding better)
        logger.info("Reading Excel file...")
        df = pd.read_excel(excel_path)

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")

        # Clean the data
        logger.info("Cleaning data...")

        # Remove completely empty rows
        df = df.dropna(how='all')

        # Fill missing values with appropriate defaults
        df_cleaned = df.copy()

        # Critical fields that should not be empty
        critical_fields = ['request_line_url', 'response_status', 'request_line_method']

        for field in critical_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = df_cleaned[field].fillna('').astype(str)
                df_cleaned[field] = df_cleaned[field].replace('nan', '')
                df_cleaned[field] = df_cleaned[field].replace('None', '')

        # Handle other text fields
        text_fields = ['request_useragent', 'action', 'action_message',
                      'message_type', 'message_description', 'message_msg']

        for field in text_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = df_cleaned[field].fillna('').astype(str)

        # Save to CSV with proper encoding
        logger.info(f"Saving to CSV with {encoding} encoding...")
        df_cleaned.to_csv(csv_path, index=False, encoding=encoding)

        logger.info(f"Successfully converted to {csv_path}")
        logger.info(f"Final dataset shape: {df_cleaned.shape}")

        # Show sample of data
        logger.info("\nSample of converted data:")
        logger.info(df_cleaned.head(3).to_string())

        # Show label distribution if available
        if 'label' in df_cleaned.columns:
            label_counts = df_cleaned['label'].value_counts()
            logger.info("\nLabel distribution:")
            for label, count in label_counts.items():
                logger.info(f"  {label}: {count} ({count/len(df_cleaned)*100:.1f}%)")

        return True

    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {e}")
        return False

def detect_excel_file(directory: str) -> str:
    """Detect the Excel file in the given directory."""
    excel_files = list(Path(directory).glob("*.xlsx")) + list(Path(directory).glob("*.xls"))

    if len(excel_files) == 0:
        logger.error(f"No Excel files found in {directory}")
        return None

    if len(excel_files) > 1:
        logger.warning(f"Multiple Excel files found, using: {excel_files[0]}")

        print("\nAvailable Excel files:")
        for i, file in enumerate(excel_files):
            print(f"  {i+1}: {file.name}")

        try:
            choice = int(input(f"Select file (1-{len(excel_files)}): ")) - 1
            if 0 <= choice < len(excel_files):
                return str(excel_files[choice])
        except (ValueError, IndexError):
            pass

        logger.warning(f"Using first file: {excel_files[0]}")

    return str(excel_files[0])

def main():
    parser = argparse.ArgumentParser(description="Convert ModSecurity Excel file to CSV")
    parser.add_argument("--input", type=str, default="data/raw",
                       help="Input directory containing Excel files")
    parser.add_argument("--output", type=str, default="data/raw/Modsec-WP-from-excel.csv",
                       help="Output CSV file path")
    parser.add_argument("--encoding", type=str, default="utf-8",
                       choices=["utf-8", "latin-1", "cp1252", "iso-8859-1"],
                       help="Output encoding for CSV file")

    args = parser.parse_args()

    # Find Excel file
    excel_path = detect_excel_file(args.input)
    if not excel_path:
        return 1

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Convert Excel to CSV
    success = convert_excel_to_csv(excel_path, args.output, args.encoding)

    if success:
        print(f"\n✅ Conversion successful!")
        print(f"📁 Excel file: {excel_path}")
        print(f"📄 CSV file: {args.output}")
        print(f"🔤 Encoding: {args.encoding}")
        print(f"\nYou can now use this CSV for training:")
        print(f"python train_enhanced_ids.py --input '{args.output}' --preprocess --train")
        return 0
    else:
        print("\n❌ Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())