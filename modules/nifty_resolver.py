"""
Generalized Nifty Index Constituent Resolver Module
────────────────────────────────────────────────────
Retrieves current constituents dynamically for any supported Nifty index
and applies historical rebalancing changes in reverse to determine the
index composition at any past date.
"""

import pandas as pd
from datetime import datetime
import urllib.request
import json
import os
import functools

# Configuration for all supported indices
SUPPORTED_INDICES = {
    "nifty50": {
        "url": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "fallback": [
            "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
            "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
            "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
            "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
            "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY",
            "JSWSTEEL", "KOTAKBANK", "LT", "LTIM", "M&M",
            "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
            "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", "TATACONSUMER",
            "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
            "ULTRACEMCO", "WIPRO", "TRENT", "ZOMATO", "JIOFIN"
        ],
        "rebalancing_log": [
            {
                "date": "2025-09-30",
                "added": ["INDIGO", "MAXHEALTH"],
                "removed": ["HEROMOTOCO", "INDUSINDBK"]
            },
            {
                "date": "2025-03-28",
                "added": ["JIOFIN", "ZOMATO"],
                "removed": ["BPCL", "BRITANNIA"]
            },
            {
                "date": "2024-09-30",
                "added": ["TRENT", "BEL"],
                "removed": ["DIVISLAB", "LTIM"]
            },
            {
                "date": "2024-03-28",
                "added": ["SHRIRAMFIN"],
                "removed": ["UPL"]
            },
            {
                "date": "2023-07-13",
                "added": ["LTIM"],
                "removed": ["HDFC"]
            },
            {
                "date": "2022-09-30",
                "added": ["ADANIENT"],
                "removed": ["SHREECEM"]
            },
            {
                "date": "2022-03-31",
                "added": ["APOLLOHOSP"],
                "removed": ["IOC"]
            },
            {
                "date": "2021-03-31",
                "added": ["TATACONSUMER"],
                "removed": ["GAIL"]
            },
            {
                "date": "2020-09-25",
                "added": ["DIVISLAB", "SBILIFE"],
                "removed": ["INFRATEL", "ZEEL"]
            },
            {
                "date": "2020-03-19",
                "added": ["SHREECEM"],
                "removed": ["YESBANK"]
            },
            {
                "date": "2019-09-27",
                "added": ["NESTLEIND"],
                "removed": ["IBULHSGFIN"]
            },
            {
                "date": "2019-03-29",
                "added": ["BRITANNIA"],
                "removed": ["HPCL"]
            },
            {
                "date": "2018-09-28",
                "added": ["JSWSTEEL"],
                "removed": ["LUPIN"]
            }
        ]
    },
    "niftybank": {
        "url": "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",
        "fallback": [
            "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "INDUSINDBK",
            "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB", "AUBANK", "BANDHANBNK"
        ],
        "rebalancing_log": []
    },
    "niftyit": {
        "url": "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",
        "fallback": [
            "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM",
            "PERSISTENT", "COFORGE", "MPHASIS", "LTTS"
        ],
        "rebalancing_log": []
    },
    "niftynext50": {
        "url": "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
        "fallback": [
            "ABB", "AMBUJACEM", "APOLLOHOSP", "BAJAJHLDNG", "BANKBARODA", "BEL", "BOSCHLTD",
            "CANBK", "CHOLAFIN", "COALINDIA", "DLF", "GAIL", "GICRE", "GODREJCP", "GRASIM",
            "HAL", "HAVELLS", "HEROMOTOCO", "ICICIGI", "ICICIPRULI", "IOC", "IRCTC", "IRFC",
            "JSWENERGY", "JSWINFRA", "LICI", "LUPIN", "MRF", "MUTHOOTFIN", "NMDC", "OBEROIRLTY",
            "PIDILITIND", "PNB", "RECLTD", "SBICARD", "SHREECEM", "SIEMENS", "SRF", "TATAELXSI",
            "TATAPOWER", "TVSMOTOR", "UNITDSPR", "VBL", "ZOMATO", "ZYDUSLIFE", "JIOFIN", "TRENT"
        ],
        "rebalancing_log": []
    }
}


@functools.lru_cache(maxsize=16)
def fetch_current_constituents(index_name: str) -> list[str]:
    """Fetches the current constituents dynamically for a given index from the NSE server."""
    index_clean = index_name.lower().replace(" ", "").replace("-", "")
    
    if index_clean not in SUPPORTED_INDICES:
        print(f"  ⚠️  Index '{index_name}' not officially supported in resolver. No URL available.")
        return []

    idx_config = SUPPORTED_INDICES[index_clean]
    url = idx_config["url"]
    fallback = idx_config["fallback"]

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            df = pd.read_csv(response)
            if "Symbol" in df.columns:
                symbols = df["Symbol"].dropna().str.strip().tolist()
                return [s for s in symbols if s]
    except Exception as e:
        print(f"  ⚠️  Failed to fetch current {index_name} constituents from NSE: {e}. Using local fallback.")
    return list(fallback)


@functools.lru_cache(maxsize=4096)
def get_constituents(index_name: str, target_date: datetime | str) -> list[str]:
    """
    Reconstructs the index constituents active on a given target date.
    
    Args:
        index_name: Name of the index (e.g., 'nifty50', 'niftybank').
        target_date: Date to resolve constituents for (datetime object or 'YYYY-MM-DD' string).
        
    Returns:
        List of uppercase ticker symbols active on that date.
    """
    index_clean = index_name.lower().replace(" ", "").replace("-", "")
    if index_clean not in SUPPORTED_INDICES:
        raise ValueError(f"Unsupported index: {index_name}")

    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = pd.to_datetime(target_date)
    
    # Standardize tz-naive comparison
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_localize(None)

    # 1. Start with the live current list
    constituents = set(fetch_current_constituents(index_clean))
    
    # 2. Iterate backward in time through rebalancing events
    rebal_log = SUPPORTED_INDICES[index_clean].get("rebalancing_log", [])
    for event in rebal_log:
        event_dt = pd.to_datetime(event["date"])
        
        # If the event occurred strictly AFTER our target date, we must UNDO it
        if event_dt > target_dt:
            # Tickers added in this event must be REMOVED
            for ticker in event["added"]:
                constituents.discard(ticker)
            # Tickers removed in this event must be ADDED back
            for ticker in event["removed"]:
                constituents.add(ticker)
        else:
            break
            
    return sorted(list(constituents))


def get_constituents_in_range(index_name: str, start_date: datetime | str, end_date: datetime | str = None) -> list[str]:
    """
    Finds the union of all index constituents that were active at any point 
    between start_date and end_date.
    
    Args:
        index_name: Name of the index.
        start_date: Start of the date range.
        end_date: End of the date range (defaults to current date).
        
    Returns:
        List of all ticker symbols that were active constituents in this period.
    """
    index_clean = index_name.lower().replace(" ", "").replace("-", "")
    if index_clean not in SUPPORTED_INDICES:
        raise ValueError(f"Unsupported index: {index_name}")

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
    
    # Standardize tz-naive
    if start_dt.tzinfo is not None:
        start_dt = start_dt.tz_localize(None)
    if end_dt.tzinfo is not None:
        end_dt = end_dt.tz_localize(None)
        
    # Get constituents at the start_date
    constituents = set(get_constituents(index_clean, start_dt))
    
    # Add any ticker that was added or removed in the log during this period
    rebal_log = SUPPORTED_INDICES[index_clean].get("rebalancing_log", [])
    for event in rebal_log:
        event_dt = pd.to_datetime(event["date"])
        if start_dt <= event_dt <= end_dt:
            for t in event["added"]:
                constituents.add(t)
            for t in event["removed"]:
                constituents.add(t)
                
    return sorted(list(constituents))


# ═══════════════════════════════════════════════════════════
# Backward-Compatible Wrappers for Nifty 50
# ═══════════════════════════════════════════════════════════

def get_nifty50_constituents(target_date: datetime | str) -> list[str]:
    """Reconstructs the Nifty 50 constituents active on a given target date (for backward compatibility)."""
    return get_constituents("nifty50", target_date)


def get_nifty50_constituents_in_range(start_date: datetime | str, end_date: datetime | str = None) -> list[str]:
    """Finds the union of all Nifty 50 constituents active in a date range (for backward compatibility)."""
    return get_constituents_in_range("nifty50", start_date, end_date)
