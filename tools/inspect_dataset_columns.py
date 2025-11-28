import pandas as pd
import argparse
import os

def inspect(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📖 Reading {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for attacks
    if 'label' in df.columns:
        print("Filtering for RFI...")
        df = df[df['label'].str.lower().str.strip() == 'rfi']
    
    cols = ['request_line_method', 'request_line_url', 'request_useragent', 'action_message', 'request_body']
    print("\n🔍 Sample Data (First 5 rows):")
    for col in cols:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(df[col].head().to_string(index=False))
        else:
            print(f"\n❌ Column '{col}' missing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/Modsec-WP.csv')
    args = parser.parse_args()
    inspect(args.input)
