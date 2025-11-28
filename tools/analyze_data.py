import pandas as pd
import argparse
import os

def analyze(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📊 Loading {file_path}...")
    # Handling potential mixed types or parsing errors
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print(f"✅ Loaded {len(df)} rows.")
    
    if 'label' not in df.columns:
        print("❌ Column 'label' not found!")
        print(f"Columns: {df.columns.tolist()}")
        return

    # Normalize labels
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    
    counts = df['label'].value_counts()
    print("\n📈 Class Distribution:")
    print("-" * 30)
    for label, count in counts.items():
        percent = (count / len(df)) * 100
        print(f"{label:<20} {count:>6} ({percent:.2f}%)")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/Modsec-WP.csv')
    args = parser.parse_args()
    analyze(args.input)
