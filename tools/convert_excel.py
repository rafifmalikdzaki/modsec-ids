import pandas as pd
import os

def convert():
    excel_path = 'data/raw/Modsec-WP.xlsx'
    csv_path = 'data/raw/Modsec-WP.csv'
    
    if not os.path.exists(excel_path):
        print(f"❌ {excel_path} not found.")
        return

    print(f"📖 Reading {excel_path}...")
    try:
        df = pd.read_excel(excel_path)
        print(f"✅ Loaded {len(df)} rows.")
        
        print(f"💾 Saving to {csv_path}...")
        df.to_csv(csv_path, index=False)
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    convert()
