import duckdb
import pandas as pd
import os
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
    
    # Load the CSV
    csv_path = "nifty50_historical_seed.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # --- UPDATED DATE CONVERSIONS ---
        # Tell pandas that the day comes first, or use format='mixed' to handle blank fields cleanly
        df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')
        df['end_date'] = pd.to_datetime(df['end_date'], dayfirst=True, errors='coerce')
        
        # Register and insert, overwriting if necessary
        conn.register("seed_df", df)
        conn.execute("DELETE FROM historical_index_map WHERE index_name='nifty50'")
        conn.execute("INSERT INTO historical_index_map SELECT * FROM seed_df")
        
        print(f"✅ Successfully seeded {len(df)} historical records into DuckDB.")
    else:
        print(f"❌ Seed file {csv_path} not found.")
        
    conn.close()

if __name__ == "__main__":
    seed_historical_index()