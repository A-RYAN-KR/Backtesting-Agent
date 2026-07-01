"""
Point-in-Time (PIT) Gating Demonstration Script
───────────────────────────────────────────────
This script demonstrates the mechanics of index resolution and point-in-time
universe gating. It simulates trading signals for YESBANK around its official
removal date from the Nifty Bank index (March 27, 2020) and shows how:
1. Entries are blocked after removal.
2. Forced liquidation is triggered on the exact day of removal.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root to path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_connectivity import DuckDBCache
from seed_db import seed_historical_index

def main():
    print("=== Point-in-Time (PIT) Gating Demonstration ===")
    
    # 1. Force seeding the database to ensure we have the correct CSV-based historical intervals
    print("[INFO] Seeding database from CSV files to ensure correct intervals...")
    seed_historical_index()
    cache = DuckDBCache()



    # 2. Query constituents of Nifty Bank around the March 2020 rebalance
    index_name = "niftybank"
    print(f"\n1. Resolving historical constituents for index: '{index_name}'")
    
    # Let's query all constituents that were active in March 2020
    target_date = pd.Timestamp("2020-03-27")
    active_constituents = cache.conn.execute(
        "SELECT symbol, start_date, end_date FROM historical_index_map "
        "WHERE index_name = ? AND start_date <= ? AND end_date >= ? "
        "ORDER BY symbol",
        [index_name, target_date, target_date]
    ).df()
    
    print(f"Active constituents on {target_date.date()}:")
    print(active_constituents.to_string(index=False))

    # 3. Generate a daily timeline around the rebalance date
    # Yes Bank was removed on March 27, 2020.
    # We create a daily timeline from March 24 to March 31, 2020 (4 trading days).
    timeline = pd.DatetimeIndex([
        "2020-03-24", # Tuesday (In index)
        "2020-03-25", # Wednesday (In index)
        "2020-03-26", # Thursday (In index)
        "2020-03-27", # Friday (Removed today)
        "2020-03-30", # Monday (Out of index)
        "2020-03-31"  # Tuesday (Out of index)
    ])
    
    print(f"\n2. Generating daily Universe Mask Matrix for timeline...")
    mask_matrix = cache.get_historical_universe_mask(index_name, timeline)
    
    # Display the mask values for YESBANK and BANDHANBNK side-by-side
    demo_mask = mask_matrix[["YESBANK", "BANDHANBNK"]].copy()
    print(demo_mask)

    # 4. Simulate strategy signals for YESBANK
    print(f"\n3. Simulating strategy signals and applying Point-in-Time Gating for YESBANK...")
    
    # Suppose our strategy generates the following raw signals:
    # - Buy (entry) signals on March 25 and March 30
    # - Sell (exit) signals on March 31
    raw_entries = pd.Series([False, True, False, True, True, False], index=timeline, dtype=bool)
    raw_exits = pd.Series([False, False, False, False, False, True], index=timeline, dtype=bool)
    
    is_constituent = mask_matrix["YESBANK"].reindex(timeline).fillna(False).astype(bool)
    
    # Formula A: Entry Gating (Bitwise AND)
    # Blocks buy signals on days the stock is not in the index
    gated_entries = raw_entries & is_constituent
    
    # Formula B: Transition Detection
    # Detects the exact day of removal by checking: (Was in index yesterday) AND (Is not in index today)
    was_removed_today = (is_constituent.shift(1) == True) & (is_constituent == False)
    
    # Formula C: Forced Liquidation (Bitwise OR)
    # Automatically triggers a sell signal on the removal day
    gated_exits = raw_exits | was_removed_today

    # 5. Display the step-by-step masking math in a clean table
    results_df = pd.DataFrame({
        "Is In Index": is_constituent,
        "Raw Entry": raw_entries,
        "Gated Entry": gated_entries,
        "Removed Today": was_removed_today,
        "Raw Exit": raw_exits,
        "Gated Exit": gated_exits
    }, index=timeline)
    
    print("\n=== SIGNAL GATING RESOLUTION TABLE (YESBANK) ===")
    print(results_df.to_string())
    
    print("\n=== Key Observations ===")
    print("- March 25 (In Index): Raw Entry is True. Gated Entry remains True.")
    print("- March 27 (Removal Day): 'Removed Today' becomes True. This forces Gated Exit to be True, liquidating the position.")
    print("- March 30 (Out of Index): Raw Entry is True. Gated Entry is masked to False, blocking look-ahead trading.")

if __name__ == "__main__":
    main()
