import duckdb
import pandas as pd
import os
import glob
from config import CACHE_DIR

def seed_historical_index():
    db_path = os.path.join(CACHE_DIR, "trading_cache.db")
    conn = duckdb.connect(db_path)
    
    # Ensure the table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_index_map (
            index_name VARCHAR,
            symbol VARCHAR,
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            UNIQUE(index_name, symbol, start_date)
        )
    """)
    
    # Find all historical seed CSVs in the root directory
    seed_files = glob.glob("*_historical_seed.csv")
    
    if not seed_files:
        print("[WARNING] No historical seed CSV files (*_historical_seed.csv) found in the root directory.")
        conn.close()
        return

    total_records = 0
    for csv_path in seed_files:
        print(f"[INFO] Processing seed file: {os.path.basename(csv_path)}")
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = {"index_name", "symbol", "start_date", "end_date"}
            if not required_cols.issubset(df.columns):
                print(f"  [ERROR] Skipping {csv_path}: Missing one or more required columns {required_cols}")
                continue
                
            # Date conversions
            df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')
            df['end_date'] = pd.to_datetime(df['end_date'], dayfirst=True, errors='coerce')
            
            # Extract unique index names in this file to clear their old data selectively
            unique_indices = df['index_name'].dropna().unique().tolist()
            
            for index_name in unique_indices:
                index_clean = index_name.lower().replace(" ", "").replace("-", "")
                conn.execute("DELETE FROM historical_index_map WHERE index_name = ?", [index_clean])
            
            # Normalize index names in the DataFrame to match lowercased, space-stripped standard
            df['index_name'] = df['index_name'].str.lower().str.replace(" ", "").str.replace("-", "")
            
            # Register and insert
            conn.register("seed_df", df)
            conn.execute("INSERT INTO historical_index_map SELECT * FROM seed_df")
            
            print(f"  [SUCCESS] Successfully seeded {len(df)} historical records for indices: {unique_indices}")
            total_records += len(df)
        except Exception as e:
            print(f"  [ERROR] Error processing {csv_path}: {e}")
            
    print(f"[INFO] Seeding complete. Total of {total_records} historical records loaded into DuckDB.")
    conn.close()

if __name__ == "__main__":
    seed_historical_index()