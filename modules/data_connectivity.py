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
            clean["timestamp"] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["Date"])
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
            return df
        except Exception:
            return pd.DataFrame()

    def has_data(self, symbol: str) -> bool:
        count = self.conn.execute(
            "SELECT COUNT(*) FROM ohlcv_data WHERE symbol=?", [symbol]
        ).fetchone()[0]
        return count > 0


# ═══════════════════════════════════════════════════════════
#  Indian Equity Cost Adaptor
# ═══════════════════════════════════════════════════════════
class IndianEquityAdaptor:
    """Models Indian equity costs: STT, stamp duty, GST, brokerage."""

    def __init__(self, is_intraday: bool = False):
        self.is_intraday = is_intraday

    def calculate_costs(self, trade_value: float, qty: int, is_buy: bool) -> dict:
        brokerage = min(20.0, trade_value * 0.0003)
        stt = (trade_value * 0.00025 if not is_buy else 0.0) if self.is_intraday else trade_value * 0.001
        txn_charge = trade_value * 0.0000322
        gst = (brokerage + txn_charge) * 0.18
        stamp_duty = (trade_value * (0.00003 if self.is_intraday else 0.00015)) if is_buy else 0.0
        sebi = trade_value * 0.000001

        total = brokerage + stt + txn_charge + gst + stamp_duty + sebi
        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "gst": round(gst, 2),
            "stamp_duty": round(stamp_duty, 2),
            "total_impact": round(total, 2),
        }


def get_market_adaptor(is_intraday: bool = False):
    """Returns the Indian equity cost adaptor."""
    return IndianEquityAdaptor(is_intraday=is_intraday)


# ═══════════════════════════════════════════════════════════
#  Data Agents
# ═══════════════════════════════════════════════════════════
class YFinanceAgent:
    """Fetches data from Yahoo Finance for Indian equities."""

    def fetch(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            print(f"  ❌  YFinance error for {symbol}: {e}")
            return pd.DataFrame()


class OpenAlgoAgent:
    """Indian market data via yfinance .NS suffix."""

    def fetch(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        yf_symbol = f"{symbol}.NS" if not symbol.endswith((".NS", ".BO")) else symbol
        return YFinanceAgent().fetch(yf_symbol, period, interval)


# ═══════════════════════════════════════════════════════════
#  Data Router
# ═══════════════════════════════════════════════════════════
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
        # Check cache first
        if self.cache.has_data(symbol):
            print(f"  ⚡  Loading {symbol} from cache …")
            cached = self.cache.load(symbol)
            if len(cached) > 50:  # only use cache if it has meaningful data
                return cached

        # Route to right agent — all tickers go through Indian market path
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            source = "yfinance"
            df = self.yf_agent.fetch(symbol, period)
        else:
            source = "openalgo"
            df = self.openalgo_agent.fetch(symbol, period)

        # Cache it
        if df is not None and not df.empty:
            self.cache.save(df, symbol, source)

        return df

    def fetch_multiple(self, symbols: list[str], period: str = "2y", market: str = "IN") -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols, returns dict of symbol → DataFrame."""
        result = {}
        for sym in symbols:
            print(f"  📡  Fetching {sym} …")
            df = self.fetch(sym.strip(), period, market)
            if df is not None and not df.empty:
                result[sym.strip()] = df
                print(f"  ✅  {sym}: {len(df)} rows fetched")
            else:
                print(f"  ⚠️  {sym}: No data returned")
        return result
