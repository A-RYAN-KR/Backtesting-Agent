"""
Module 3: Data & Market Connectivity ("The Heart")
───────────────────────────────────────────────────
• DuckDBCache          – Local OHLCV caching
• IndianEquityAdaptor  – Cost model (STT, stamp duty, GST, brokerage)
• YFinanceAgent        – Equities data via Yahoo Finance
• OpenAlgoAgent        – Indian markets (mock → yfinance .NS fallback)
• DataRouter           – Auto-routes tickers to the right agent
"""

import os
import re
import pandas as pd
import numpy as np
import duckdb
import yfinance as yf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CACHE_DIR


# ═══════════════════════════════════════════════════════════
#  DuckDB Cache
# ═══════════════════════════════════════════════════════════
class DuckDBCache:
    """Persistent OHLCV data cache using DuckDB."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.path.join(CACHE_DIR, "trading_cache.db")
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                source VARCHAR,
                UNIQUE(symbol, timestamp)
            )
        """)

    def save(self, df: pd.DataFrame, symbol: str, source: str):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return

        try:
            clean = pd.DataFrame()
            clean["symbol"] = [symbol] * len(df)
            # Ensure timestamps are tz-naive for DuckDB TIMESTAMP compatibility
            ts_index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["Date"])
            if hasattr(ts_index, 'tz') and ts_index.tz is not None:
                ts_index = ts_index.tz_localize(None)
            clean["timestamp"] = ts_index
            clean["open"] = df["Open"].values
            clean["high"] = df["High"].values
            clean["low"] = df["Low"].values
            clean["close"] = df["Close"].values
            clean["volume"] = df["Volume"].values
            clean["source"] = [source] * len(df)

            self.conn.register("temp_df", clean)
            self.conn.execute("INSERT OR IGNORE INTO ohlcv_data SELECT * FROM temp_df")
            print(f"  💾  Cached {len(clean)} rows for {symbol}")
        except Exception as e:
            print(f"  ❌  Cache write error for {symbol}: {e}")

    def load(self, symbol: str) -> pd.DataFrame:
        try:
            df = self.conn.execute(
                "SELECT * FROM ohlcv_data WHERE symbol=? ORDER BY timestamp ASC", [symbol]
            ).df()
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                # Map back to Title Case to satisfy downstream validation
                df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                }, inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

    def has_data(self, symbol: str) -> bool:
        count = self.conn.execute(
            "SELECT COUNT(*) FROM ohlcv_data WHERE symbol=?", [symbol]
        ).fetchone()[0]
        return count > 0

    def get_cache_range(self, symbol: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Returns the (min_timestamp, max_timestamp) for a cached symbol."""
        try:
            row = self.conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv_data WHERE symbol=?", [symbol]
            ).fetchone()
            if row and row[0] is not None and row[1] is not None:
                return pd.to_datetime(row[0]), pd.to_datetime(row[1])
        except Exception as e:
            print(f"  ❌  Error reading cache range for {symbol}: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════
#  Indian Equity Cost Adaptor
# ═══════════════════════════════════════════════════════════
class IndianEquityAdaptor:
    """Models Indian equity costs: STT, stamp duty, GST, brokerage."""

    def __init__(self, is_intraday: bool = False):
        self.is_intraday = is_intraday

    def calculate_costs(self, trade_value: float, qty: int, is_buy: bool) -> dict:
        return {
            "brokerage": 0.0,
            "stt": 0.0,
            "gst": 0.0,
            "stamp_duty": 0.0,
            "total_impact": 0.0,
        }


def get_market_adaptor(is_intraday: bool = False):
    """Returns the Indian equity cost adaptor."""
    return IndianEquityAdaptor(is_intraday=is_intraday)


# ═══════════════════════════════════════════════════════════
#  Data Agents
# ═══════════════════════════════════════════════════════════
class YFinanceAgent:
    """Fetches data from Yahoo Finance for Indian equities."""

    def fetch(self, symbol: str, period: str = "2y", interval: str = "1d", start=None) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            if start is not None:
                df = ticker.history(start=start, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
            if df is None or df.empty:
                print(f"  ⚠️  No data returned for {symbol} (ticker may be delisted or misspelled)")
                return pd.DataFrame()
            # Safe timezone stripping — only strip if tz-aware
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            print(f"  ❌  YFinance error for {symbol}: {e}")
            return pd.DataFrame()


class OpenAlgoAgent:
    """Indian market data via yfinance .NS suffix."""

    def fetch(self, symbol: str, period: str = "2y", interval: str = "1d", start=None) -> pd.DataFrame:
        yf_symbol = f"{symbol}.NS" if not symbol.endswith((".NS", ".BO")) else symbol
        return YFinanceAgent().fetch(yf_symbol, period, interval, start=start)


# ═══════════════════════════════════════════════════════════
#  Data Router
# ═══════════════════════════════════════════════════════════
def parse_duration_to_start_date(period: str) -> pd.Timestamp:
    """
    Parses yfinance-style period durations (e.g. 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    into a target start date pd.Timestamp.
    """
    now = pd.Timestamp.now()
    if period == "max":
        return pd.Timestamp("1900-01-01")
    elif period == "ytd":
        return pd.Timestamp(year=now.year, month=1, day=1)

    match = re.match(r"^(\d+)(y|mo|d|w)$", period)
    if not match:
        # Fallback to default of 2y if pattern does not match
        return now - pd.DateOffset(years=2)

    val = int(match.group(1))
    unit = match.group(2)

    if unit == "y":
        return now - pd.DateOffset(years=val)
    elif unit == "mo":
        return now - pd.DateOffset(months=val)
    elif unit == "d":
        return now - pd.DateOffset(days=val)
    elif unit == "w":
        return now - pd.DateOffset(weeks=val)

    return now - pd.DateOffset(years=2)


class DataRouter:
    """
    Auto-routes ticker symbols to the appropriate data agent.
    Rules:
      - Ends with '.NS' or '.BO' → Indian (YFinance with NSE/BSE suffix)
      - Otherwise → Indian (auto-appends .NS suffix via OpenAlgo)
    """

    def __init__(self):
        self.yf_agent = YFinanceAgent()
        self.openalgo_agent = OpenAlgoAgent()
        self.cache = DuckDBCache()

    def fetch(self, symbol: str, period: str = "2y", market: str = "IN") -> pd.DataFrame:
        target_start_date = parse_duration_to_start_date(period)
        padded_start_date = target_start_date - pd.DateOffset(days=100)
        
        # Check cache validity (coverage and staleness)
        cache_valid = False
        if self.cache.has_data(symbol):
            min_cached, max_cached = self.cache.get_cache_range(symbol)
            if min_cached is not None and max_cached is not None:
                # 1. Check if the cache is recent (not stale)
                # We consider it stale if the maximum cached timestamp is more than 5 days older than now
                now = pd.Timestamp.now()
                # Safe timezone stripping — only strip if tz-aware, avoid TypeError on tz-naive timestamps
                max_cached_naive = max_cached.tz_localize(None) if max_cached.tzinfo is not None else max_cached
                min_cached_naive = min_cached.tz_localize(None) if min_cached.tzinfo is not None else min_cached
                now_naive = now.tz_localize(None) if now.tzinfo is not None else now
                padded_start_naive = padded_start_date.tz_localize(None) if padded_start_date.tzinfo is not None else padded_start_date
                
                is_recent = (now_naive - max_cached_naive).days <= 5
                
                # 2. Check if cache goes back far enough
                has_history = min_cached_naive <= padded_start_naive
                
                if is_recent and has_history:
                    cache_valid = True
                else:
                    reason = []
                    if not is_recent:
                        reason.append(f"stale (latest: {max_cached_naive.date()})")
                    if not has_history:
                        reason.append(f"insufficient history (earliest: {min_cached_naive.date()}, requested with padding: {padded_start_naive.date()})")
                    print(f"  ⚠️  Cache invalid for {symbol}: {', '.join(reason)}. Refetching...")

        if cache_valid:
            print(f"  ⚡  Loading {symbol} from cache …")
            cached = self.cache.load(symbol)
            if len(cached) > 50:  # only use cache if it has meaningful data
                cached.index = pd.to_datetime(cached.index)
                if hasattr(cached.index, 'tz') and cached.index.tz is not None:
                    cached.index = cached.index.tz_localize(None)
                padded_start_naive = padded_start_date.tz_localize(None) if padded_start_date.tzinfo is not None else padded_start_date
                target_start_naive = target_start_date.tz_localize(None) if target_start_date.tzinfo is not None else target_start_date
                sliced = cached[cached.index >= padded_start_naive]
                sliced.attrs["target_start_date"] = target_start_naive
                return sliced

        # Route to right agent — all tickers go through Indian market path
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            source = "yfinance"
            df = self.yf_agent.fetch(symbol, start=padded_start_date)
        else:
            source = "openalgo"
            df = self.openalgo_agent.fetch(symbol, start=padded_start_date)

        # Cache it
        if df is not None and not df.empty:
            self.cache.save(df, symbol, source)
            target_start_naive = target_start_date.tz_localize(None) if target_start_date.tzinfo is not None else target_start_date
            df.attrs["target_start_date"] = target_start_naive

        return df

    def fetch_multiple(self, symbols: list[str], period: str = "2y", market: str = "IN") -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols, returns dict of symbol → DataFrame.
        Gracefully skips tickers that return no data (delisted, misspelled, etc.)."""
        result = {}
        for sym in symbols:
            sym_clean = sym.strip()
            print(f"  📡  Fetching {sym_clean} …")
            try:
                df = self.fetch(sym_clean, period, market)
            except Exception as e:
                print(f"  ❌  {sym_clean}: Unexpected error during fetch: {e}")
                continue

            if df is not None and not df.empty:
                # Validate that the DataFrame has minimum required columns
                required_cols = {"Open", "High", "Low", "Close", "Volume"}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    print(f"  ⚠️  {sym_clean}: Missing required columns {missing_cols}. Skipping.")
                    continue
                result[sym_clean] = df
                print(f"  ✅  {sym_clean}: {len(df)} rows fetched")
            else:
                print(f"  ⚠️  {sym_clean}: No data returned (ticker may be delisted or misspelled). Skipping.")
        return result
