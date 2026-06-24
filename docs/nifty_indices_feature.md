# Nifty Multi-Index Support & Point-in-Time Gating Feature Guide

This document describes the design, architecture, and implementation of the **Generalized Multi-Index Support** and **Point-in-Time (PIT) Universe Gating** system. This feature solves the critical challenges of **survivorship bias** and **look-ahead bias** in quantitative index backtesting.

---

## 1. Feature Overview

In historical backtesting, testing on an index using only its current members introduces **survivorship bias** (ignoring companies that were delisted or removed) and **look-ahead bias** (trading companies before they were officially added or keeping them after they were removed). 

Our system solves this by dynamically reconstructing the exact constituents of any supported index for **every single trading day** in the backtest window, and applying a strict daily gate on trading signals.

### Key Capabilities:
* **Dynamic Index Recognition**: The Natural Language Processing (NLP) layer automatically maps natural language queries (e.g., "Bank Nifty", "Nifty Next 50") to standardized index macros.
* **Dynamic Data Connectivity Routing**: Intercepts index macros, queries the DuckDB database cache to find all active historical constituents during the backtest window, downloads/caches their pricing data, and stamps metadata context.
* **Point-in-Time (PIT) Universe Gating**:
  * **Entry Gating**: Blocks strategy entries for stocks on days they were not in the index.
  * **Exit Gating (Forced Liquidation)**: Forces an immediate liquidation of any open position on the exact day a stock is officially removed from the index.
* **High-Accuracy Seeding**: Supports dynamic, index-specific loading of constituent timelines via modular CSV seed files.

---

## 2. Supported Indices

The system supports a curated list of liquid Nifty indices. Their live sources, fallbacks, and historical rebalancing details are configured as follows:

| Index Name | Index Ticker | Internal Macro | Live NSE CSV Source URL | Historical Depth & Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Nifty 50** | `^NSEI` | `nifty50` | `.../ind_nifty50list.csv` | Full historical rebalancings seeded from **2010 to present** (highly accurate). |
| **Nifty Bank** | `^NSEBANK` | `niftybank` | `.../ind_niftybanklist.csv` | High-accuracy seed tracking major rebalancings (Yes Bank exit, Bandhan Bank entry, AU Bank, PNB, Canara Bank, etc.). |
| **Nifty IT** | `^CNXIT` | `niftyit` | `.../ind_niftyitlist.csv` | High-accuracy seed tracking major rebalancings (LTIM, Coforge, Mphasis, Persistent, LTTS additions; Mindtree, Hexaware removals). |
| **Nifty Next 50** | `^NSEJR` | `niftynext50` | `.../ind_niftynext50list.csv` | Current constituents seeded as active from **2010 to 2099** (accurate for recent periods). |

---

## 3. System Architecture & Pipeline Flow

The following diagram traces the lifecycle of an index strategy through the 5-module pipeline:

```mermaid
graph TD
    A[User Query: 'Buy Nifty Bank when...'] -->|Step 1: NLP Parse| B[QueryInterpreter]
    B -->|Resolves 'Nifty Bank' to macro| C[Tickers: 'niftybank']
    
    C -->|Step 3: Data Connectivity| D[DataRouter]
    D -->|Fast DB Check: SELECT EXISTS| E[DuckDB Cache: historical_index_map]
    
    E -->|True: Registered Index| F[Dynamic Constituent Query]
    F -->|Retrieves active tickers in range| G[Resolved Ticker List]
    
    G -->|Fetches price data| H[Yahoo Finance / Cache]
    H -->|Stamps metadata: index_context = 'niftybank'| I[Pricing DataFrames]
    
    I -->|Step 4: Execution Engine| J[BacktestEngine]
    J -->|Queries cache for mask| K[get_historical_universe_mask]
    K -->|Generates daily boolean matrix| L[PIT Universe Mask Matrix]
    
    L -->|Gates entry signals| M[entries = entries & is_constituent]
    L -->|Forces liquidation on removal| N[exits = exits | was_removed_today]
    
    M & N -->|Executes trades at T+1 Open| O[VectorBT Portfolio Run]
    O -->|Step 5: Analytics & Audit| P[ReportGenerator & QuantStats]
```

---

## 4. In-Depth Technical Working

### Phase 1: NLP Macro Resolution (`modules/nlp_orchestration.py`)
The system prompt in the `QueryInterpreter` instructs the Gemini LLM to detect when a user wants to trade an entire index. Rather than expanding the symbols statically, the LLM maps the index name to its standardized macro string:
```python
# System prompt instruction:
"- INDEX RECOGNITION (CRITICAL): If the user requests to trade an entire index (e.g., 'nifty bank'), pass the exact index macro name in the tickers array (e.g., ['niftybank'])."
```

### Phase 2: Dynamic Index Routing (`modules/data_connectivity.py`)
When the `DataRouter` receives the ticker array in `fetch_multiple`, it checks if the symbol is a registered index in the DuckDB database:
```python
# Query DuckDB to check if the symbol is a registered index
is_index = self.cache.conn.execute(
    "SELECT EXISTS(SELECT 1 FROM historical_index_map WHERE index_name = ?)", 
    [sym_clean]
).fetchone()[0]
```
If `is_index` is true, the router dynamically queries the database for all tickers that were active members of that index during the backtest timeframe (plus 100 days of warmup padding). It downloads their prices and stamps `df.attrs["index_context"] = sym_clean` on each DataFrame.

### Phase 3: Point-in-Time Universe Gating (`modules/execution_engine.py`)
During strategy execution, if `index_context` is present, the engine queries the cache to generate a daily boolean mask matrix (`universe_mask_matrix`):
```python
# Get daily boolean membership matrix
universe_mask_matrix = cache.get_historical_universe_mask(index_context, pricing_index)
```
This matrix is aligned day-by-day with the strategy's signal vectors:
1. **Entry Gating**: Restricts entries to days where the stock is an active member:
   ```python
   entries = entries & is_constituent
   ```
2. **Forced Liquidation**: Detects the day the stock was removed and forces an immediate exit signal:
   ```python
   was_removed_today = (is_constituent.shift(1) == True) & (is_constituent == False)
   exits = exits | was_removed_today
   ```

---

## 5. Database Seeding Mechanics (`seed_db.py`)

The DuckDB database acts as the fast query layer for historical constituency. The seeder script (`seed_db.py`) is designed to be fully modular and generic:
1. It automatically scans the project root for files matching `*_historical_seed.csv`.
2. It parses the `index_name` column of each CSV.
3. To prevent cross-index data contamination, it selectively deletes existing records *only* for the indices found in that CSV:
   ```sql
   DELETE FROM historical_index_map WHERE index_name = ?
   ```
4. It normalizes the index names and bulk-inserts the new constituency intervals.

---

## 6. How to Run & Seed

### Step 1: Seeding the Database
To load or update the historical constituency records in the database, close any active backtesting sessions and run:
```powershell
python seed_db.py
```

### Step 2: Running a Backtest
Start the pipeline in interactive mode:
```powershell
python main.py
```
Or run in single-shot mode:
```powershell
python main.py --query "Buy Nifty Bank when RSI < 30, sell when RSI > 70 for the last 6 months"
```

---

## 7. Sample Queries & Expected Log Outputs

### Sample Query A: Nifty Bank (RSI Strategy)
* **Query**: `"Buy Nifty Bank when RSI < 30, sell when RSI > 70 for the last 6 months"`
* **Expected Log Trace**:
  * **NLP**: Recognizes the index and outputs `Tickers: ['niftybank']`, `Duration: 6mo`.
  * **Router**: Prints `[INFO] Resolved 'niftybank' macro to 12 historical constituents.` and downloads/caches the 12 banks.
  * **PIT Gating**: Prints `📌 Generated Point-in-Time universe mask matrix for index: niftybank`.
  * **Execution**: Runs the strategy on each bank stock, gating entries and exits day-by-day.

### Sample Query B: Nifty IT (Moving Average Crossover)
* **Query**: `"Buy Nifty IT when SMA 10 crosses above SMA 30, sell when SMA 10 crosses below SMA 30 for the last 1 year"`
* **Expected Log Trace**:
  * **NLP**: Outputs `Tickers: ['niftyit']`, `Duration: 1y`.
  * **Router**: Prints `[INFO] Resolved 'niftyit' macro to 10 historical constituents.`, downloading and caching `TCS`, `INFY`, `HCLTECH`, `WIPRO`, `TECHM`, `LTIM`, `PERSISTENT`, `COFORGE`, `MPHASIS`, and `LTTS`.
  * **PIT Gating**: Prints `📌 Generated Point-in-Time universe mask matrix for index: niftyit`.
  * **Execution**: Runs the strategy across the IT portfolio, strictly gating trades based on historical membership dates.
