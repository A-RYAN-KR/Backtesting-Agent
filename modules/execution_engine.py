"""
Module 4: Execution & Portfolio Engine ("The Engine")
─────────────────────────────────────────────────────
• BacktestEngine       – Vectorized execution via VectorBT
• TransactionCostAgent – Dynamic slippage based on volatility
• RiskAgent            – VaR, CVaR, position constraints
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from scipy.stats import norm
import multiprocessing
import traceback
import queue

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_INIT_CASH

# ─── Timeout for exec() to prevent infinite loops from LLM code ────
EXEC_TIMEOUT_SECONDS = 10


def print(*args, sep=" ", end="\n", file=None, flush=False):
    """
    Safe print function for Windows terminals. Gracefully encodes Unicode
    characters by substituting unrecognized glyphs with placeholders instead of crashing.
    """
    target_file = file if file is not None else sys.stdout
    text = sep.join(map(str, args)) + end
    try:
        target_file.write(text)
        if flush:
            target_file.flush()
    except UnicodeEncodeError:
        encoding = getattr(target_file, "encoding", None) or "ascii"
        encoded = text.encode(encoding, errors="replace").decode(encoding)
        target_file.write(encoded)
        if flush:
            target_file.flush()


# ═══════════════════════════════════════════════════════════
#  Case-Insensitive DataFrame & pandas_ta Proxy
# ═══════════════════════════════════════════════════════════
class CaseInsensitiveDF(pd.DataFrame):
    """
    A DataFrame subclass that supports case-insensitive column access.
    Columns are stored lowercased; lookups try lowercase first.
    """

    def __getitem__(self, key):
        if isinstance(key, str):
            # Try exact match first, then lowercase
            if key in self.columns:
                return super().__getitem__(key)
            lower_key = key.lower()
            if lower_key in self.columns:
                return super().__getitem__(lower_key)
            # Fuzzy match: strip underscores/numbers pattern differences
            for col in self.columns:
                if col.lower() == lower_key:
                    return super().__getitem__(col)
        return super().__getitem__(key)


class PandasTAProxy:
    """
    Wraps the pandas_ta module so that every indicator function
    that returns a DataFrame has its column names normalized to
    lowercase.  This prevents KeyError from LLM-generated code
    that guesses wrong casing (e.g. 'MACDS_12_26_9' vs 'macds_12_26_9').
    """

    def __init__(self):
        import pandas_ta as _pta
        self._pta = _pta

    def __getattr__(self, name):
        attr = getattr(self._pta, name)
        if callable(attr):
            def _wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    result.columns = [c.lower() for c in result.columns]
                    result = CaseInsensitiveDF(result)
                elif isinstance(result, pd.Series):
                    if result.name and isinstance(result.name, str):
                        result.name = result.name.lower()
                return result
            return _wrapper
        return attr


# ═══════════════════════════════════════════════════════════
#  Transaction Cost Agent
# ═══════════════════════════════════════════════════════════
class TransactionCostAgent:
    """Estimates dynamic slippage based on historical volatility."""

    @staticmethod
    def calculate_dynamic_slippage(price_series: pd.Series, window: int = 14) -> pd.Series:
        return pd.Series(0.0, index=price_series.index)

    @staticmethod
    def estimate_fees(market: str = "IN") -> float:
        """Returns a flat fee percentage for Indian equity market."""
        return 0.0


# ═══════════════════════════════════════════════════════════
#  Risk Agent
# ═══════════════════════════════════════════════════════════
class RiskAgent:
    """Computes risk metrics and enforces position constraints."""

    @staticmethod
    def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Value-at-Risk."""
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        return float(np.percentile(clean, (1 - confidence) * 100))

    @staticmethod
    def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)."""
        var = RiskAgent.compute_var(returns, confidence)
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    @staticmethod
    def position_constraint(var: float, max_acceptable_var: float = -0.02) -> dict:
        """Returns sizing constraint based on VaR limit."""
        if var < max_acceptable_var:
            leverage = max_acceptable_var / var
            return {
                "approved": False,
                "leverage_factor": round(leverage, 4),
                "message": f"VaR ({var:.2%}) exceeds limit ({max_acceptable_var:.2%}). Scale to {leverage:.2%} of capital.",
            }
        return {
            "approved": True,
            "leverage_factor": 1.0,
            "message": f"VaR ({var:.2%}) within safe limits. Full position approved.",
        }


def _multiprocess_worker(code: str, price_series_dict: dict, has_pandas_ta: bool, result_queue):
    """
    Worker function to execute LLM-generated code in an isolated child process.
    Prevents infinite loops from locking up the main thread or process.
    """
    try:
        import pandas as pd
        import numpy as np

        _pta_proxy = None
        if has_pandas_ta:
            try:
                _pta_proxy = PandasTAProxy()
            except Exception:
                pass

        # Prepare namespace with all required variables and aliases
        namespace = {
            "pd": pd,
            "np": np,
            "close_prices": price_series_dict.get("close"),
            "open_prices": price_series_dict.get("open"),
            "high_prices": price_series_dict.get("high"),
            "low_prices": price_series_dict.get("low"),
            "volume": price_series_dict.get("volume"),
            "pandas_ta": _pta_proxy,
            "ta": _pta_proxy,
            "pta": _pta_proxy,
        }

        # Run the code
        exec(code, namespace)

        # Retrieve entries and exits
        entries = namespace.get("entries")
        exits = namespace.get("exits")

        # Collect indicator stats for debugging
        debug_indicators = {}
        for var_name in ["rsi", "rsi_values", "sma", "sma_fast", "sma_slow", "macd", "ema"]:
            if var_name in namespace and namespace[var_name] is not None:
                ind = namespace[var_name]
                if hasattr(ind, 'mean') and hasattr(ind, 'min') and hasattr(ind, 'max') and hasattr(ind, 'isna'):
                    try:
                        debug_indicators[var_name] = {
                            "mean": float(ind.mean()),
                            "min": float(ind.min()),
                            "max": float(ind.max()),
                            "nan_count": int(ind.isna().sum())
                        }
                    except Exception:
                        pass

        # Put results back
        result_queue.put((entries, exits, debug_indicators, None))
    except Exception as e:
        import traceback
        result_queue.put((None, None, None, (type(e).__name__, str(e), traceback.format_exc())))


# ═══════════════════════════════════════════════════════════
#  Backtest Engine
# ═══════════════════════════════════════════════════════════
class BacktestEngine:
    """
    Runs vectorized backtests via VectorBT.
    Supports:
      - Signal-based backtesting from generated entries/exits
      - Dynamic slippage + fees
      - Multi-ticker execution
    """

    EXEC_TIMEOUT = EXEC_TIMEOUT_SECONDS

    def __init__(self, init_cash: float = DEFAULT_INIT_CASH):
        self.init_cash = init_cash
        self.cost_agent = TransactionCostAgent()
        self.risk_agent = RiskAgent()

    @staticmethod
    def _exec_with_timeout(code: str, close_prices, open_prices, high_prices, low_prices, volume, has_pandas_ta: bool, timeout: int = 10):
        """
        Execute code with process isolation and a timeout to prevent infinite loops.
        """
        import multiprocessing
        import os
        import sys

        # FIX: Force the Windows child process to recognize the project root directory context
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if "PYTHONPATH" in os.environ:
            if project_root not in os.environ["PYTHONPATH"].split(os.pathsep):
                os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ["PYTHONPATH"]
        else:
            os.environ["PYTHONPATH"] = project_root

        price_series_dict = {
            "close": close_prices,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "volume": volume
        }

        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()

        p = ctx.Process(
            target=_multiprocess_worker,
            args=(code, price_series_dict, has_pandas_ta, result_queue)
        )
        p.start()

        try:
            # Read from the queue FIRST to clear the buffer
            entries, exits, debug_indicators, err = result_queue.get(timeout=timeout)
        except queue.Empty:
            # If the queue is empty after the timeout, it's a true infinite loop
            p.terminate()
            p.join()
            raise TimeoutError(f"Code execution exceeded {timeout}s timeout")

        # Now it is safe to cleanly join the process
        p.join(timeout=1)

        if err is not None:
            err_type, err_msg, tb_str = err
            raise RuntimeError(f"{err_type}: {err_msg}\n{tb_str}")

        return entries, exits, debug_indicators

    def execute_strategy_code(self, code: str, price_data: dict) -> dict:
        """
        Executes generated strategy code against price data.

        Args:
            code:       Python code string defining `entries` and `exits` as boolean Series
            price_data: dict of {ticker: DataFrame} with OHLCV data

        Returns:
            dict with portfolios, metrics, and risk analysis per ticker
        """
        results = {}

        for ticker, df in price_data.items():
            print(f"\n  ⚙️  Running backtest for {ticker} …")

            try:
                # Prepare data variables the generated code expects
                close_prices = df["Close"] if "Close" in df.columns else df["close"]
                open_prices = df["Open"] if "Open" in df.columns else df.get("open", close_prices)
                high_prices = df["High"] if "High" in df.columns else df.get("high", close_prices)
                low_prices = df["Low"] if "Low" in df.columns else df.get("low", close_prices)
                volume = df["Volume"] if "Volume" in df.columns else df.get("volume", pd.Series(0, index=df.index))

                # Ensure numeric types
                close_prices = pd.to_numeric(close_prices, errors="coerce").dropna()
                open_prices = pd.to_numeric(open_prices, errors="coerce").reindex(close_prices.index).fillna(close_prices)
                high_prices = pd.to_numeric(high_prices, errors="coerce").reindex(close_prices.index).fillna(close_prices)
                low_prices = pd.to_numeric(low_prices, errors="coerce").reindex(close_prices.index).fillna(close_prices)

                # ── DEBUG: Data shape validation ────────────────
                print(f"\n  ┌── DEBUG: DATA VALIDATION ({ticker}) ──────────────────────────")
                print(f"  │ close_prices: shape={close_prices.shape}, dtype={close_prices.dtype}")
                print(f"  │ date range: {close_prices.index[0]} → {close_prices.index[-1]}")
                print(f"  │ price range: ₹{close_prices.min():.2f} – ₹{close_prices.max():.2f}")
                print(f"  │ NaN count: {close_prices.isna().sum()}")
                print(f"  └────────────────────────────────────────────────────")

                # Execute the LLM-generated code in a sandboxed namespace
                # Use PandasTAProxy to normalize indicator column names to lowercase
                try:
                    _pta_proxy = PandasTAProxy()
                except ImportError:
                    _pta_proxy = None
                    print(f"  [WARNING] pandas_ta not installed — indicator functions unavailable")

                # Guard: If pandas_ta is not available and the code references it, fail fast
                if _pta_proxy is None and "pandas_ta" in code:
                    print(f"  [ERROR] Strategy code requires pandas_ta but it is not installed.")
                    results[ticker] = {"error": "pandas_ta is required but not installed."}
                    continue

                namespace = {
                    "pd": pd,
                    "np": np,
                    "close_prices": close_prices,
                    "open_prices": open_prices,
                    "high_prices": high_prices,
                    "low_prices": low_prices,
                    "volume": volume,
                    "pandas_ta": _pta_proxy,
                    "ta": _pta_proxy,
                    "pta": _pta_proxy,
                }

                # Rewrite 'import pandas_ta' so it binds our proxy instead (as a fallback)
                patched_code = code.replace("import pandas_ta", "pandas_ta = pandas_ta  # proxy already injected")

                # Execute with timeout to prevent infinite loops from LLM hallucinations
                try:
                    entries, exits, debug_indicators = self._exec_with_timeout(
                        patched_code,
                        close_prices,
                        open_prices,
                        high_prices,
                        low_prices,
                        volume,
                        _pta_proxy is not None,
                        timeout=self.EXEC_TIMEOUT
                    )
                except TimeoutError:
                    print(f"  ❌  Code execution timed out after {self.EXEC_TIMEOUT}s for {ticker} (possible infinite loop)")
                    results[ticker] = {"error": f"Code execution timed out after {self.EXEC_TIMEOUT}s (possible infinite loop)."}
                    continue
                except Exception as exec_err:
                    print(f"  ❌  Code execution failed for {ticker}: {exec_err}")
                    results[ticker] = {"error": f"Code execution error: {exec_err}"}
                    continue

                if entries is None or exits is None:
                    print(f"  ❌  Code did not produce 'entries' or 'exits' for {ticker}")
                    results[ticker] = {"error": "Signal generation failed."}
                    continue

                # Align indices
                entries = entries.reindex(close_prices.index).fillna(False).astype(bool)
                exits = exits.reindex(close_prices.index).fillna(False).astype(bool)

                # Trim warmup padding if present in metadata
                if "target_start_date" in df.attrs:
                    target_start = df.attrs["target_start_date"]
                    target_start = target_start.tz_localize(None) if target_start.tzinfo is not None else target_start
                    
                    close_prices = close_prices[close_prices.index >= target_start]
                    open_prices = open_prices[open_prices.index >= target_start]
                    high_prices = high_prices[high_prices.index >= target_start]
                    low_prices = low_prices[low_prices.index >= target_start]
                    volume = volume[volume.index >= target_start]
                    entries = entries[entries.index >= target_start]
                    exits = exits[exits.index >= target_start]

                # ══════════════════════════════════════════════════
                # DEBUG 1: SIGNAL GENERATION
                # ══════════════════════════════════════════════════
                print(f"\n  ┌── DEBUG: SIGNAL GENERATION ({ticker}) ─────────────────────────")
                print(f"  │ Total Entry Signals: {entries.sum()}")
                print(f"  │ Total Exit Signals:  {exits.sum()}")
                print(f"  │ Signal Series Length: {len(entries)}")
                print(f"  │ Entries dtype: {entries.dtype} | Exits dtype: {exits.dtype}")

                # Check for any indicator values generated in the namespace
                for var_name in ["rsi", "rsi_values", "sma", "sma_fast", "sma_slow", "macd", "ema"]:
                    if debug_indicators and var_name in debug_indicators:
                        stats = debug_indicators[var_name]
                        print(f"  │ Indicator '{var_name}': mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}, NaN={stats['nan_count']}")

                # ══════════════════════════════════════════════════
                # VECTORIZED SIGNAL FILTERING & PORTFOLIO RUN
                # ══════════════════════════════════════════════════
                raw_entries = entries.copy()
                raw_exits = exits.copy()

                # Dynamic slippage (Fixed to 0 per Indian market constraints)
                slippage = pd.Series(0.0, index=close_prices.index)
                fees = 0.0

                # ── DEBUG: Transaction costs ───────────────────────
                print(f"\n  ┌── DEBUG: TRANSACTION COSTS ({ticker}) ──────────────────────────")
                print(f"  │ Slippage: mean={slippage.mean():.6f}, max={slippage.max():.6f}")
                print(f"  │ Flat fee rate: {fees:.4f} ({fees*100:.2f}%)")
                print(f"  └────────────────────────────────────────────────────")

                # Run VectorBT portfolio directly on raw signals with native conflict resolution
                portfolio = vbt.Portfolio.from_signals(
                    close_prices,
                    entries=entries,
                    exits=exits,
                    freq="1D",
                    init_cash=self.init_cash,
                    fees=fees,
                    slippage=slippage,
                    upon_long_conflict='ignore',  # Ignores new BUY signals if already LONG
                    upon_short_conflict='ignore', # Ignores new SELL signals if already SHORT
                    upon_dir_conflict='ignore'    # If BUY and SELL happen on the exact same bar, ignore both
                )

                # Reconstruct resolved/executed signals from portfolio orders for tracing & diagnostics
                actual_entries = pd.Series(False, index=close_prices.index)
                actual_exits = pd.Series(False, index=close_prices.index)
                try:
                    orders_df = portfolio.orders.records_readable
                    if not orders_df.empty:
                        for _, row in orders_df.iterrows():
                            t = row['Timestamp']
                            side = row['Side']
                            if t in actual_entries.index:
                                if side == 'Buy':
                                    actual_entries.loc[t] = True
                                elif side == 'Sell':
                                    actual_exits.loc[t] = True
                except Exception as e:
                    print(f"  ⚠️  Could not extract actual orders from portfolio: {e}")

                entries = actual_entries
                exits = actual_exits

                # ── DEBUG: State-aware filtering results ────────
                raw_entry_count = int(raw_entries.sum())
                raw_exit_count = int(raw_exits.sum())
                filtered_entry_count = int(entries.sum())
                filtered_exit_count = int(exits.sum())

                print(f"\n  ┌── DEBUG: STATE-AWARE SIGNAL FILTERING ({ticker}) ──────────────")
                print(f"  │ Raw entry signals:      {raw_entry_count}")
                print(f"  │ Raw exit signals:       {raw_exit_count}")
                print(f"  │ Filtered entry signals: {filtered_entry_count}  (removed {raw_entry_count - filtered_entry_count} redundant)")
                print(f"  │ Filtered exit signals:  {filtered_exit_count}  (removed {raw_exit_count - filtered_exit_count} redundant)")
                if filtered_entry_count == 0:
                    print(f"  │ ⚠️  WARNING: No valid entry signals after state filtering!")
                if filtered_entry_count != filtered_exit_count and abs(filtered_entry_count - filtered_exit_count) > 1:
                    print(f"  │ ℹ️  Entry/Exit count mismatch by {abs(filtered_entry_count - filtered_exit_count)} (last position may still be open)")
                print(f"  └────────────────────────────────────────────────────")



                # ══════════════════════════════════════════════════
                # FAST TRADE EXECUTION LOGS
                # ══════════════════════════════════════════════════
                print(f"\n  ┌── DEBUG: TRADE EXECUTION LOG ({ticker}) ─────────────────")
                try:
                    trades_df = portfolio.trades.records_readable
                    if not trades_df.empty:
                        for _, row in trades_df.head(10).iterrows(): # Print first 10 trades safely
                            entry_date = str(row['Entry Timestamp'])[:10]
                            exit_date = str(row['Exit Timestamp'])[:10]
                            pnl = row['PnL']
                            print(f"  │ 🟢 BUY {entry_date} | 🔴 SELL {exit_date} | PnL: ₹{pnl:.2f}")
                        if len(trades_df) > 10:
                            print(f"  │ ... and {len(trades_df) - 10} more trades.")
                    else:
                        print(f"  │ ⚠️  NO TRADES were executed!")
                except Exception as e:
                    print(f"  │ ⚠️  Could not extract trades: {e}")
                print(f"  └────────────────────────────────────────────────────")

                # Risk analysis
                strat_returns = portfolio.returns()
                var_95 = self.risk_agent.compute_var(strat_returns)
                cvar_95 = self.risk_agent.compute_cvar(strat_returns)
                constraint = self.risk_agent.position_constraint(var_95)

                # ══════════════════════════════════════════════════
                # DEBUG 6: DAILY RETURNS DEBUG
                # ══════════════════════════════════════════════════
                print(f"\n  ┌── DEBUG: DAILY RETURNS ({ticker}) ─────────────────────────────")
                print(f"  │ Total return days: {len(strat_returns)}")
                print(f"  │ Zero-return days: {(strat_returns == 0).sum()}")
                print(f"  │ Positive days: {(strat_returns > 0).sum()}")
                print(f"  │ Negative days: {(strat_returns < 0).sum()}")
                print(f"  │ NaN days: {strat_returns.isna().sum()}")
                print(f"  │ Mean daily return: {strat_returns.mean():.6f}")
                print(f"  │ Std daily return:  {strat_returns.std():.6f}")
                print(f"  │ Min daily return:  {strat_returns.min():.6f}")
                print(f"  │ Max daily return:  {strat_returns.max():.6f}")

                # Warn if most days have zero return (suggests strategy never enters)
                zero_pct = (strat_returns == 0).sum() / len(strat_returns) * 100
                if zero_pct > 90:
                    print(f"  │ ⚠️  WARNING: {zero_pct:.1f}% of days have zero return — strategy may rarely be in a position!")
                print(f"  └────────────────────────────────────────────────────")

                # ── DEBUG: Risk analysis results ────────────────
                print(f"\n  ┌── DEBUG: RISK ANALYSIS ({ticker}) ──────────────────────────────")
                print(f"  │ VaR (95%):    {var_95:.4%}")
                print(f"  │ CVaR (95%):   {cvar_95:.4%}")
                print(f"  │ Constraint:   {constraint['message']}")
                print(f"  │ Approved:     {constraint['approved']}")
                print(f"  └────────────────────────────────────────────────────")

                # Collect fees
                total_fees = float(portfolio.orders.fees.sum()) if hasattr(portfolio.orders, 'fees') else 0.0

                results[ticker] = {
                    "portfolio": portfolio,
                    "total_return": float(portfolio.total_return()),
                    "total_fees": total_fees,
                    "trades_count": int(portfolio.trades.count()) if hasattr(portfolio.trades, 'count') else 0,
                    "win_rate": float(portfolio.trades.win_rate()) if hasattr(portfolio.trades, 'win_rate') else 0.0,
                    "risk": {
                        "var_95": var_95,
                        "cvar_95": cvar_95,
                        "constraint": constraint,
                    },
                    "entries_count": int(entries.sum()),
                    "exits_count": int(exits.sum()),
                }

                # ── DEBUG: Final result summary ─────────────────
                print(f"\n  ┌── DEBUG: FINAL RESULT SUMMARY ({ticker}) ────────────────────────")
                print(f"  │ Total Return:     {results[ticker]['total_return']:.4%}")
                print(f"  │ Total Fees:       ₹{total_fees:.2f}")
                print(f"  │ VBT Trades Count: {results[ticker]['trades_count']}")
                print(f"  │ VBT Win Rate:     {results[ticker]['win_rate']:.2%}")
                print(f"  │ Entry Signals:    {results[ticker]['entries_count']}")
                print(f"  │ Exit Signals:     {results[ticker]['exits_count']}")
                print(f"  │ Init Cash:        ₹{self.init_cash:,.2f}")
                print(f"  │ Final Equity:     ₹{float(portfolio.final_value()):,.2f}")
                print(f"  └────────────────────────────────────────────────────")

                print(f"  ✅  {ticker}: Return={results[ticker]['total_return']:.2%}  |  Trades={results[ticker]['trades_count']}  |  Fees=₹{total_fees:.2f}")

            except Exception as e:
                print(f"  ❌  Backtest error for {ticker}: {e}")
                import traceback
                print(f"  ── DEBUG: Full traceback ──")
                traceback.print_exc()
                print(f"  ──────────────────────────")
                results[ticker] = {"error": str(e)}

        return results
