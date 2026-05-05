"""
Module 5: Analytics & Audit ("The Eyes")
────────────────────────────────────────
• PerformanceAnalyzer – Sharpe, Sortino, max drawdown, Calmar, etc.
• AuditAgent          – Factor attribution (alpha vs cost drag)
• ReportGenerator     – Interactive Plotly equity curves + QuantStats tearsheet
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import REPORTS_DIR


# ═══════════════════════════════════════════════════════════
#  Performance Analyzer
# ═══════════════════════════════════════════════════════════
class PerformanceAnalyzer:
    """Calculates comprehensive performance metrics from backtest results."""

    @staticmethod
    def compute_metrics(portfolio) -> dict:
        """Extract all key metrics from a VectorBT portfolio."""
        returns = portfolio.returns()
        cumulative = (1 + returns).cumprod()

        # ── DEBUG: Daily returns raw analysis ───────────────
        print(f"\n  ┌── DEBUG: PERFORMANCE ANALYZER — RETURNS ANALYSIS ────────────")
        print(f"  │ Returns series length: {len(returns)}")
        print(f"  │ Zero-return days: {(returns == 0).sum()}")
        print(f"  │ Positive days:    {(returns > 0).sum()}")
        print(f"  │ Negative days:    {(returns < 0).sum()}")
        print(f"  │ NaN days:         {returns.isna().sum()}")
        zero_pct = (returns == 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        if zero_pct > 80:
            print(f"  │ ⚠️  WARNING: {zero_pct:.1f}% zero-return days — strategy may be idle most of the time!")
        print(f"  └────────────────────────────────────────────────────")

        # Sharpe Ratio (annualized)
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        # ── DEBUG: Sharpe computation trace ─────────────────
        print(f"  ┌── DEBUG: SHARPE COMPUTATION ─────────────────────────────────")
        print(f"  │ mean_ret={float(mean_ret):.8f}  std_ret={float(std_ret):.8f}")
        print(f"  │ Sharpe = ({float(mean_ret):.8f} / {float(std_ret):.8f}) * sqrt(252) = {float(sharpe):.4f}")
        if std_ret == 0:
            print(f"  │ ⚠️  WARNING: std_ret is ZERO — Sharpe set to 0.0 (no variance in returns)")
        print(f"  └────────────────────────────────────────────────────")

        # Sortino Ratio
        downside = returns[returns < 0].std()
        sortino = (mean_ret / downside * np.sqrt(252)) if downside > 0 else 0.0

        # Max Drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min())

        # ── DEBUG: Drawdown trace ──────────────────────────
        print(f"  ┌── DEBUG: DRAWDOWN ──────────────────────────────────────────")
        print(f"  │ Max Drawdown: {max_dd:.4%}")
        max_dd_date = drawdown.idxmin()
        print(f"  │ Max Drawdown Date: {max_dd_date}")
        print(f"  │ Final cumulative return: {float(cumulative.iloc[-1]):.4f}")
        print(f"  └────────────────────────────────────────────────────")

        # Calmar Ratio
        total_ret = float(portfolio.total_return())
        # Approximate years
        n_days = len(returns)
        years = n_days / 252 if n_days > 0 else 1
        annualized_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
        calmar = annualized_ret / abs(max_dd) if max_dd != 0 else 0.0

        # Win Rate
        try:
            win_rate = float(portfolio.trades.win_rate())
        except Exception:
            win_rate = 0.0

        # Profit Factor
        try:
            winning = returns[returns > 0].sum()
            losing = abs(returns[returns < 0].sum())
            profit_factor = winning / losing if losing > 0 else float('inf')
        except Exception:
            profit_factor = 0.0

        # ── DEBUG: Profit factor trace ─────────────────────
        print(f"  ┌── DEBUG: PROFIT FACTOR ─────────────────────────────────────")
        try:
            print(f"  │ Sum of winning returns: {float(winning):.6f}")
            print(f"  │ Sum of losing returns:  {float(losing):.6f}")
            print(f"  │ Profit Factor:          {float(profit_factor):.4f}")
        except Exception:
            print(f"  │ Profit Factor computation failed")
        print(f"  └────────────────────────────────────────────────────")

        return {
            "total_return": total_ret,
            "annualized_return": annualized_ret,
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": max_dd,
            "calmar_ratio": float(calmar),
            "win_rate": win_rate,
            "profit_factor": float(profit_factor),
            "total_trades": int(portfolio.trades.count()) if hasattr(portfolio.trades, 'count') else 0,
            "volatility": float(std_ret * np.sqrt(252)),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "avg_daily_return": float(mean_ret),
            "trading_days": n_days,
        }


# ═══════════════════════════════════════════════════════════
#  Audit Agent
# ═══════════════════════════════════════════════════════════
class AuditAgent:
    """
    Post-trade analysis: attributes returns to alpha, risk, or cost factors.
    """

    @staticmethod
    def attribute_returns(backtest_result: dict) -> dict:
        """Decomposes total return into alpha, cost drag, and risk-adjusted components."""
        total_return = backtest_result.get("total_return", 0.0)
        total_fees = backtest_result.get("total_fees", 0.0)
        init_cash = 100000  # default

        cost_drag = total_fees / init_cash if init_cash > 0 else 0.0
        gross_return = total_return + cost_drag  # Return before costs
        alpha = gross_return  # Simplified: alpha = gross return (no benchmark subtraction)

        risk_info = backtest_result.get("risk", {})
        var_95 = risk_info.get("var_95", 0.0)

        # ── DEBUG: Attribution computation trace ───────────
        print(f"\n  ┌── DEBUG: AUDIT AGENT — RETURN ATTRIBUTION ──────────────────")
        print(f"  │ Total Return (net):  {total_return:.6f} ({total_return:.4%})")
        print(f"  │ Total Fees:          ₹{total_fees:.2f}")
        print(f"  │ Cost Drag:           {cost_drag:.6f} ({cost_drag:.4%})")
        print(f"  │ Gross Return:        {gross_return:.6f} ({gross_return:.4%})")
        print(f"  │ Alpha Estimate:      {alpha:.6f}")
        print(f"  │ VaR 95%:             {var_95:.4%}")
        risk_adj = total_return / abs(var_95) if var_95 != 0 else 0.0
        print(f"  │ Risk-Adj Return:     {risk_adj:.4f}")
        if total_fees > 0 and abs(cost_drag) > abs(total_return) * 0.5:
            print(f"  │ ⚠️  WARNING: Fees consume >50% of gross return!")
        print(f"  └────────────────────────────────────────────────────")

        return {
            "gross_return": gross_return,
            "net_return": total_return,
            "cost_drag": cost_drag,
            "alpha_estimate": alpha,
            "var_95": var_95,
            "risk_adjusted_return": total_return / abs(var_95) if var_95 != 0 else 0.0,
            "verdict": (
                "✅ Alpha Positive — Strategy generates excess returns."
                if total_return > 0
                else "⚠️ Negative Returns — Strategy underperforms. Review logic or regime."
            ),
        }


# ═══════════════════════════════════════════════════════════
#  Report Generator
# ═══════════════════════════════════════════════════════════
class ReportGenerator:
    """Generates interactive Plotly charts and a QuantStats HTML tearsheet."""

    @staticmethod
    def generate_equity_curve(portfolio, ticker: str, metrics: dict) -> go.Figure:
        """Creates an interactive equity curve with Plotly."""
        equity = portfolio.value()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Equity Curve", "Drawdown", "Returns Distribution"),
            vertical_spacing=0.08,
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity.index, y=equity.values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#00d4aa", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.1)",
            ),
            row=1, col=1,
        )

        # Drawdown
        returns = portfolio.returns()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        fig.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown.values,
                mode="lines",
                name="Drawdown",
                line=dict(color="#ff4757", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,71,87,0.2)",
            ),
            row=2, col=1,
        )

        # Returns distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                name="Daily Returns",
                marker_color="#5352ed",
                opacity=0.75,
                nbinsx=50,
            ),
            row=3, col=1,
        )

        # Layout
        title = (
            f"<b>{ticker}</b> — Backtest Report  |  "
            f"Return: {metrics['total_return']:.2%}  |  "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
            f"MaxDD: {metrics['max_drawdown']:.2%}"
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            template="plotly_dark",
            height=900,
            showlegend=False,
            margin=dict(l=60, r=30, t=80, b=40),
        )

        fig.update_yaxes(title_text="Value (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig

    @staticmethod
    def generate_quantstats_report(portfolio, ticker: str, benchmark_ticker: str = "^NSEI") -> str | None:
        """Generates a QuantStats HTML tearsheet. Returns the file path."""
        try:
            import quantstats as qs
            qs.extend_pandas()

            returns = portfolio.returns()
            returns.index = pd.to_datetime(returns.index)

            report_path = os.path.join(REPORTS_DIR, f"{ticker}_quantstats.html")
            qs.reports.html(
                returns,
                benchmark=benchmark_ticker,
                output=report_path,
                title=f"{ticker} Strategy Tearsheet",
            )
            return report_path
        except Exception as e:
            print(f"  ⚠️  QuantStats report failed: {e}")
            return None

    @staticmethod
    def generate_full_report(backtest_results: dict, strategy_info: dict) -> dict:
        """
        Generates complete analytics for all tickers.

        Returns dict with metrics, attribution, chart paths, and report paths per ticker.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_info,
            "tickers": {},
        }

        for ticker, result in backtest_results.items():
            if "error" in result:
                print(f"\n  ┌── DEBUG: SKIPPING {ticker} — error in backtest result ──────")
                print(f"  │ Error: {result['error']}")
                print(f"  └────────────────────────────────────────────────────")
                report["tickers"][ticker] = {"error": result["error"]}
                continue

            print(f"\n  ┌── DEBUG: GENERATING REPORT FOR {ticker} ─────────────────────")
            print(f"  │ Computing performance metrics …")
            print(f"  └────────────────────────────────────────────────────")

            portfolio = result["portfolio"]

            # Performance metrics
            metrics = PerformanceAnalyzer.compute_metrics(portfolio)

            # Factor attribution
            attribution = AuditAgent.attribute_returns(result)

            # ── DEBUG: Equity curve validation ────────────────
            equity_vals = portfolio.value()
            print(f"\n  ┌── DEBUG: EQUITY CURVE VALIDATION ({ticker}) ─────────────────")
            print(f"  │ Equity start: ₹{float(equity_vals.iloc[0]):,.2f}")
            print(f"  │ Equity end:   ₹{float(equity_vals.iloc[-1]):,.2f}")
            print(f"  │ Equity min:   ₹{float(equity_vals.min()):,.2f}")
            print(f"  │ Equity max:   ₹{float(equity_vals.max()):,.2f}")
            if float(equity_vals.iloc[0]) == float(equity_vals.iloc[-1]):
                print(f"  │ ⚠️  WARNING: Equity start == end — strategy may have had no effect!")
            print(f"  └────────────────────────────────────────────────────")

            # Generate equity curve
            fig = ReportGenerator.generate_equity_curve(portfolio, ticker, metrics)
            chart_path = os.path.join(REPORTS_DIR, f"{ticker}_equity_curve.html")
            fig.write_html(chart_path)

            # QuantStats tearsheet
            qs_path = ReportGenerator.generate_quantstats_report(portfolio, ticker)

            report["tickers"][ticker] = {
                "metrics": metrics,
                "attribution": attribution,
                "risk": result.get("risk", {}),
                "chart_path": chart_path,
                "quantstats_path": qs_path,
            }

            print(f"\n  📊  {ticker} Analytics:")
            print(f"      Total Return:     {metrics['total_return']:.2%}")
            print(f"      Annualized:       {metrics['annualized_return']:.2%}")
            print(f"      Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
            print(f"      Sortino Ratio:    {metrics['sortino_ratio']:.2f}")
            print(f"      Max Drawdown:     {metrics['max_drawdown']:.2%}")
            print(f"      Win Rate:         {metrics['win_rate']:.2%}")
            print(f"      Total Trades:     {metrics['total_trades']}")
            print(f"      Profit Factor:    {metrics['profit_factor']:.2f}")
            print(f"      ─── Attribution ───")
            print(f"      Cost Drag:        {attribution['cost_drag']:.4%}")
            print(f"      {attribution['verdict']}")
            if qs_path:
                print(f"      📄 QuantStats:    {qs_path}")
            print(f"      📈 Equity Chart:  {chart_path}")

        return report


# ═══════════════════════════════════════════════════════════
#  Console Summary Printer
# ═══════════════════════════════════════════════════════════
def print_final_summary(report: dict):
    """Prints a beautiful console summary of the full report."""
    print("\n" + "═" * 70)
    print("  🏁  BACKTESTING AGENT — FINAL REPORT")
    print("═" * 70)

    strategy = report.get("strategy", {})
    print(f"\n  📝 Strategy:")
    print(f"     Entry: {strategy.get('entry_logic', 'N/A')}")
    print(f"     Exit:  {strategy.get('exit_logic', 'N/A')}")
    print(f"     Period: {strategy.get('duration', 'N/A')}")

    for ticker, data in report.get("tickers", {}).items():
        if "error" in data:
            print(f"\n  ❌ {ticker}: {data['error']}")
            continue

        m = data["metrics"]
        a = data["attribution"]
        r = data.get("risk", {})

        print(f"\n  ┌──────────────────── {ticker} ────────────────────┐")
        print(f"  │ Total Return:    {m['total_return']:>9.2%}   │ Sharpe: {m['sharpe_ratio']:>6.2f}    │")
        print(f"  │ Annualized:      {m['annualized_return']:>9.2%}   │ Sortino: {m['sortino_ratio']:>5.2f}    │")
        print(f"  │ Max Drawdown:    {m['max_drawdown']:>9.2%}   │ Calmar: {m['calmar_ratio']:>6.2f}    │")
        print(f"  │ Win Rate:        {m['win_rate']:>9.2%}   │ Trades: {m['total_trades']:>6d}    │")
        print(f"  │ Volatility:      {m['volatility']:>9.2%}   │ P.Factor: {m['profit_factor']:>4.2f}  │")
        print(f"  │ Cost Drag:       {a['cost_drag']:>9.4%}   │ VaR 95%: {r.get('var_95', 0):>6.2%}  │")
        print(f"  │ {a['verdict'][:50]:<50s} │")
        print(f"  └─────────────────────────────────────────────────┘")

        if data.get("chart_path"):
            print(f"  📈 Equity curve: {data['chart_path']}")
        if data.get("quantstats_path"):
            print(f"  📄 Tearsheet:    {data['quantstats_path']}")

    print("\n" + "═" * 70)
