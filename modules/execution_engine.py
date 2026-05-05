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

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEFAULT_INIT_CASH


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
        daily_returns = price_series.pct_change()
        rolling_vol = daily_returns.rolling(window=window).std()
        # Base slippage + volatility-proportional component
        slippage = 0.0005 + (rolling_vol.fillna(0) * 0.1)
        return slippage

    @staticmethod
    def estimate_fees(market: str = "IN") -> float:
        """Returns a flat fee percentage for Indian equity market."""
        # Indian equity: ~0.2% (brokerage + STT approx)
        return 0.002


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

    def __init__(self, init_cash: float = DEFAULT_INIT_CASH):
        self.init_cash = init_cash
        self.cost_agent = TransactionCostAgent()
        self.risk_agent = RiskAgent()

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
                high_prices = df["High"] if "High" in df.columns else df.get("high", close_prices)
                low_prices = df["Low"] if "Low" in df.columns else df.get("low", close_prices)
                volume = df["Volume"] if "Volume" in df.columns else df.get("volume", pd.Series(0, index=df.index))

                # Ensure numeric types
                close_prices = pd.to_numeric(close_prices, errors="coerce").dropna()
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
                    print(f"  ⚠️  pandas_ta not installed — indicator functions unavailable")

                namespace = {
                    "pd": pd,
                    "np": np,
                    "close_prices": close_prices,
                    "high_prices": high_prices,
                    "low_prices": low_prices,
                    "volume": volume,
                    "pandas_ta": _pta_proxy,
                }

                # Rewrite 'import pandas_ta' so it binds our proxy instead
                patched_code = code.replace("import pandas_ta", "pandas_ta = pandas_ta  # proxy already injected")
                exec(patched_code, namespace)

                entries = namespace.get("entries")
                exits = namespace.get("exits")

                if entries is None or exits is None:
                    print(f"  ❌  Code did not produce 'entries' or 'exits' for {ticker}")
                    results[ticker] = {"error": "Signal generation failed."}
                    continue

                # Align indices
                entries = entries.reindex(close_prices.index).fillna(False).astype(bool)
                exits = exits.reindex(close_prices.index).fillna(False).astype(bool)

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
                    if var_name in namespace and namespace[var_name] is not None:
                        ind = namespace[var_name]
                        if hasattr(ind, 'mean'):
                            print(f"  │ Indicator '{var_name}': mean={float(ind.mean()):.4f}, min={float(ind.min()):.4f}, max={float(ind.max()):.4f}, NaN={int(ind.isna().sum())}")

                print(f"  │")
                print(f"  │ ── First 10 ENTRY signal dates ──")
                entry_dates = entries[entries].head(10)
                if len(entry_dates) > 0:
                    for idx in entry_dates.index:
                        price_at = close_prices.loc[idx] if idx in close_prices.index else float('nan')
                        print(f"  │   {idx}  |  Price: ₹{price_at:.2f}")
                else:
                    print(f"  │   ⚠️  NO entry signals generated!")

                print(f"  │")
                print(f"  │ ── First 10 EXIT signal dates ──")
                exit_dates = exits[exits].head(10)
                if len(exit_dates) > 0:
                    for idx in exit_dates.index:
                        price_at = close_prices.loc[idx] if idx in close_prices.index else float('nan')
                        print(f"  │   {idx}  |  Price: ₹{price_at:.2f}")
                else:
                    print(f"  │   ⚠️  NO exit signals generated!")

                # Check for overlapping signals (entry + exit on same day)
                overlap_count = int((entries & exits).sum())
                if overlap_count > 0:
                    print(f"  │")
                    print(f"  │ ⚠️  WARNING: {overlap_count} days have BOTH entry AND exit signals!")

                print(f"  └────────────────────────────────────────────────────")

                # ══════════════════════════════════════════════════
                # STATE-AWARE SIGNAL FILTERING
                # ══════════════════════════════════════════════════
                # Ensure logical consistency:
                #   - Entry signals only fire when currently FLAT (no position)
                #   - Exit signals only fire when currently IN a position
                #   - If both fire on the same bar, exit wins (close first)
                raw_entries = entries.copy()
                raw_exits = exits.copy()

                filtered_entries = pd.Series(False, index=entries.index)
                filtered_exits = pd.Series(False, index=exits.index)
                in_position = False

                for i in range(len(entries)):
                    entry_sig = bool(entries.iloc[i])
                    exit_sig = bool(exits.iloc[i])

                    if in_position:
                        # Currently holding — only exits are valid
                        if exit_sig:
                            filtered_exits.iloc[i] = True
                            in_position = False
                        # Ignore entry signals while already in position
                    else:
                        # Currently flat — only entries are valid
                        if entry_sig and not exit_sig:
                            filtered_entries.iloc[i] = True
                            in_position = True
                        elif entry_sig and exit_sig:
                            # Conflict: exit wins (can't exit when flat, so skip both)
                            pass
                        # Ignore exit signals while already flat

                entries = filtered_entries
                exits = filtered_exits

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

                # Dynamic slippage
                slippage = self.cost_agent.calculate_dynamic_slippage(close_prices)
                fees = self.cost_agent.estimate_fees("IN")

                # ── DEBUG: Slippage stats ───────────────────────
                print(f"\n  ┌── DEBUG: TRANSACTION COSTS ({ticker}) ──────────────────────────")
                print(f"  │ Slippage: mean={slippage.mean():.6f}, max={slippage.max():.6f}")
                print(f"  │ Flat fee rate: {fees:.4f} ({fees*100:.2f}%)")
                print(f"  └────────────────────────────────────────────────────")

                # Run VectorBT portfolio
                portfolio = vbt.Portfolio.from_signals(
                    close_prices,
                    entries,
                    exits,
                    freq="1D",
                    init_cash=self.init_cash,
                    slippage=slippage,
                    fees=fees,
                )

                # ══════════════════════════════════════════════════
                # DEBUG 2: MAIN LOOP TRACE (row-by-row simulation)
                # ══════════════════════════════════════════════════
                print(f"\n  ┌── DEBUG: ROW-BY-ROW EXECUTION TRACE ({ticker}) ─────────────────")
                print(f"  │ {'Date':<12} | {'Price':>9} | {'Entry':>5} | {'Exit':>5} | {'Pos':>3} | {'Cash':>12} | {'Shares':>8} | {'Equity':>12}")
                print(f"  │ {'─'*12}─┼─{'─'*9}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*3}─┼─{'─'*12}─┼─{'─'*8}─┼─{'─'*12}")

                equity_series = portfolio.value()
                cash_series = portfolio.cash()
                # Reconstruct share positions from equity and cash
                shares_series = (equity_series - cash_series) / close_prices
                shares_series = shares_series.fillna(0)

                sim_position = 0  # Track position for validation: 0=flat, 1=long
                prev_price = None
                prev_equity = None
                equity_stale_warnings = 0
                position_warnings = []
                trade_log = []

                for i, date in enumerate(close_prices.index):
                    price = float(close_prices.iloc[i])
                    entry = bool(entries.iloc[i])
                    exit_sig = bool(exits.iloc[i])
                    equity = float(equity_series.iloc[i]) if i < len(equity_series) else 0.0
                    cash = float(cash_series.iloc[i]) if i < len(cash_series) else 0.0
                    shares = float(shares_series.iloc[i]) if i < len(shares_series) else 0.0

                    # Determine current position state
                    current_pos = 1 if shares > 0.01 else 0

                    # Print first 20 rows, then every 50th row, plus all trade rows
                    is_trade_row = (entry and sim_position == 0) or (exit_sig and sim_position == 1)
                    if i < 20 or i % 50 == 0 or is_trade_row or i == len(close_prices) - 1:
                        date_str = str(date)[:10]
                        marker = ""
                        if entry and sim_position == 0:
                            marker = " ◀ BUY"
                        elif exit_sig and sim_position == 1:
                            marker = " ◀ SELL"
                        print(f"  │ {date_str:<12} | ₹{price:>8.2f} | {str(entry):>5} | {str(exit_sig):>5} | {current_pos:>3} | ₹{cash:>11.2f} | {shares:>8.2f} | ₹{equity:>11.2f}{marker}")

                    # ── DEBUG 5: POSITION VALIDATION ────────────
                    if entry and sim_position == 1:
                        warn = f"  │ ⚠️  POSITION WARNING on {str(date)[:10]}: BUY signal while ALREADY IN POSITION (pos={sim_position})"
                        position_warnings.append(warn)
                    if exit_sig and sim_position == 0:
                        warn = f"  │ ⚠️  POSITION WARNING on {str(date)[:10]}: SELL signal while NOT IN POSITION (pos={sim_position})"
                        position_warnings.append(warn)

                    # ── DEBUG 3: TRADE EXECUTION LOGS ───────────
                    if entry and sim_position == 0:
                        sim_position = 1
                        trade_log.append(f"  │ 🟢 BUY  executed on {str(date)[:10]} at ₹{price:.2f} | Shares: {shares:.2f} | Cash left: ₹{cash:.2f}")
                    elif exit_sig and sim_position == 1:
                        sim_position = 0
                        trade_log.append(f"  │ 🔴 SELL executed on {str(date)[:10]} at ₹{price:.2f} | Shares sold: {shares:.2f} | Cash: ₹{cash:.2f}")

                    # ── DEBUG 4: EQUITY CALCULATION CHECK ───────
                    if prev_price is not None and prev_equity is not None:
                        if abs(price - prev_price) > 0.01 and abs(equity - prev_equity) < 0.001 and current_pos == 1:
                            equity_stale_warnings += 1
                            if equity_stale_warnings <= 5:
                                print(f"  │ ⚠️  WARNING: Equity NOT updating on {str(date)[:10]} despite price change (₹{prev_price:.2f}→₹{price:.2f}) while in position!")

                    prev_price = price
                    prev_equity = equity

                print(f"  │ {'─'*12}─┴─{'─'*9}─┴─{'─'*5}─┴─{'─'*5}─┴─{'─'*3}─┴─{'─'*12}─┴─{'─'*8}─┴─{'─'*12}")
                print(f"  │")
                print(f"  │ ── TRADE EXECUTION LOG ──")
                if trade_log:
                    for tl in trade_log:
                        print(tl)
                else:
                    print(f"  │ ⚠️  NO TRADES were executed!")
                print(f"  │ Total Trades Logged: {len(trade_log)}")
                print(f"  │")

                # Print position warnings
                if position_warnings:
                    print(f"  │ ── POSITION WARNINGS ({len(position_warnings)}) ──")
                    for pw in position_warnings[:10]:
                        print(pw)
                    if len(position_warnings) > 10:
                        print(f"  │   … and {len(position_warnings) - 10} more warnings")
                else:
                    print(f"  │ ✅ No position validation warnings.")

                if equity_stale_warnings > 0:
                    print(f"  │")
                    print(f"  │ ⚠️  Total equity-stale warnings: {equity_stale_warnings}")
                else:
                    print(f"  │ ✅ Equity updates correctly on all price changes while in position.")

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
                print(f"  │ Final Equity:     ₹{float(equity_series.iloc[-1]):,.2f}")
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
