"""
main.py — Backtesting Agent: Unified Pipeline
══════════════════════════════════════════════
A single command-line interface that orchestrates all 5 modules:

  Module 1: NLP & Orchestration   → Parse natural language query
  Module 2: Strategy Synthesis    → Generate VectorBT code from intent
  Module 3: Data & Market         → Fetch & cache OHLCV data
  Module 4: Execution Engine      → Run vectorized backtest
  Module 5: Analytics & Audit     → Performance metrics & reports

Usage:
  python main.py                  # Interactive mode
  python main.py --query "..."    # Single-shot mode
"""

import argparse
import json
import sys
import warnings
import os

warnings.filterwarnings("ignore")

# Configure UTF-8 encoding for standard streams to prevent UnicodeEncodeError on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── Ensure project root is on path ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GEMINI_API_KEY, DEFAULT_MARKET
from modules.nlp_orchestration import (
    QueryInterpreter,
    DAGPlanner,
    RegistrationBus,
    MemoryAgent,
)
from modules.strategy_synthesis import AlphaAgent
from modules.data_connectivity import DataRouter
from modules.execution_engine import BacktestEngine
from modules.analytics_audit import ReportGenerator, print_final_summary


# ═══════════════════════════════════════════════════════════
#  Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════
class BacktestingPipeline:
    """
    End-to-end backtesting pipeline.
    Takes a natural language query and produces a full backtest report.
    """

    def __init__(self):
        # Validate API key
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
            print("❌  GEMINI_API_KEY is not set. Please add it to your .env file.")
            print("   Edit: e:\\backtesting3\\.env")
            sys.exit(1)

        # Initialize all agents
        print("\n🚀 Initializing Backtesting Agent …\n")

        self.bus = RegistrationBus()
        self.interpreter = QueryInterpreter()
        self.alpha_agent = AlphaAgent()
        self.data_router = DataRouter()
        self.engine = BacktestEngine()
        self.memory = MemoryAgent()

        # Register agents on the bus
        self.bus.register("QueryInterpreter", "NLP")
        self.bus.register("AlphaAgent", "CodeGen")
        self.bus.register("DataRouter", "DataFetcher")
        self.bus.register("BacktestEngine", "Executor")
        self.bus.register("ReportGenerator", "Analytics")

        print("  ✅  All agents initialized and registered.\n")

    def run(self, query: str, tickers: str, market: str = DEFAULT_MARKET) -> dict | None:
        """
        Execute the full backtesting pipeline.

        Args:
            query:   Natural language trading strategy description
            tickers: Comma-separated ticker symbols (e.g., RELIANCE, TCS, INFY)
            market:  Market identifier (IN for Indian equity)

        Returns:
            Full analytics report dict, or None on failure.
        """

        print("═" * 70)
        print("  📡  BACKTESTING AGENT — PIPELINE START")
        print("═" * 70)
        print(f"\n  Query:   {query}")
        print(f"  Tickers: {tickers}")
        print(f"  Market:  {market}\n")

        # ── Step 1: NLP & Orchestration ─────────────────────
        print("─" * 50)
        print("  STEP 1/5: NLP & Orchestration")
        print("─" * 50)

        self.bus.update_status("QueryInterpreter", "Working")
        result = self.interpreter.parse(query, tickers)

        if result["status"] != "success":
            print(f"\n  ❌  Pipeline aborted: {result['message']}")
            return None

        parsed = result["data"]
        print(f"  📋  Parsed Strategy:")
        print(f"       Entry:    {parsed['entry_logic']}")
        print(f"       Exit:     {parsed['exit_logic']}")
        print(f"       Duration: {parsed['duration']}")
        print(f"       Tickers:  {parsed['tickers']}")
        self.bus.update_status("QueryInterpreter", "Idle")

        # Build DAG
        planner = DAGPlanner()
        dag = planner.build_plan(parsed)
        exec_order = planner.get_execution_order()
        print(f"  🗂️  DAG Execution Order: {' → '.join(exec_order)}")

        # ── Step 2: Strategy Synthesis ──────────────────────
        print(f"\n{'─' * 50}")
        print("  STEP 2/5: Strategy Synthesis")
        print("─" * 50)

        self.bus.update_status("AlphaAgent", "Working")
        planner.update_node_status("Generate_Strategy_Code", "Running")

        strategy_code, phase_2_score = self.alpha_agent.generate_strategy_code(parsed)
        
        if strategy_code.startswith("# Error"):
            print(f"\n  ❌  Pipeline aborted: {strategy_code}")
            planner.update_node_status("Generate_Strategy_Code", "Failed")
            self.bus.update_status("AlphaAgent", "Idle")
            return None

        print(f"\n  🤖  Generated Strategy Code (Structural Confidence: {phase_2_score:.2f}):")
        print("  ┌─────────────────────────────────────────────")
        for line in strategy_code.strip().split("\n"):
            print(f"  │  {line}")
        print("  └─────────────────────────────────────────────\n")

        planner.update_node_status("Generate_Strategy_Code", "Completed")
        self.bus.update_status("AlphaAgent", "Idle")

        # ── Step 3: Data & Market Connectivity ──────────────
        print("─" * 50)
        print("  STEP 3/5: Data & Market Connectivity")
        print("─" * 50)

        self.bus.update_status("DataRouter", "Working")

        ticker_list = parsed["tickers"]
        duration = parsed.get("duration", "2y")

        # Fetch data for all tickers
        price_data = self.data_router.fetch_multiple(ticker_list, period=duration, market=market)

        if not price_data:
            print("  ❌  No data fetched. Pipeline aborted.")
            return None

        for t in ticker_list:
            planner.update_node_status(f"Fetch_Data_{t}", "Completed")
        self.bus.update_status("DataRouter", "Idle")

        # ── Step 4: Execution Engine ────────────────────────
        print(f"\n{'─' * 50}")
        print("  STEP 4/5: Execution Engine")
        print("─" * 50)

        self.bus.update_status("BacktestEngine", "Working")
        planner.update_node_status("Run_Backtest", "Running")

        backtest_results = self.engine.execute_strategy_code(strategy_code, price_data)

        planner.update_node_status("Run_Backtest", "Completed")
        self.bus.update_status("BacktestEngine", "Idle")

        # ── Step 5: Analytics & Audit ───────────────────────
        print(f"\n{'─' * 50}")
        print("  STEP 5/5: Analytics & Audit")
        print("─" * 50)

        self.bus.update_status("ReportGenerator", "Working")
        planner.update_node_status("Risk_Analysis", "Completed")
        planner.update_node_status("Generate_Report", "Running")

        strategy_info = {
            "entry_logic": parsed["entry_logic"],
            "exit_logic": parsed["exit_logic"],
            "duration": parsed["duration"],
            "tickers": parsed["tickers"],
            "raw_query": query,
        }

        report = ReportGenerator.generate_full_report(backtest_results, strategy_info)

        # Calculate CCI & Verdict for each ticker
        phase_1_score = parsed.get("phase_1_score", 1.0)
        if report and "tickers" in report:
            for ticker, data in report["tickers"].items():
                if "error" in data:
                    continue
                metrics = data.get("metrics", {})
                phase_3_score = metrics.get("phase_3_score", 0.0)
                
                # Calculate verdict
                verdict_data = gatekeeper_verdict(phase_1_score, phase_2_score, phase_3_score)
                data["verdict"] = verdict_data
                
                # Debug output for verification
                print(f"\n  [GATEKEEPER] Verdict for {ticker}:")
                print(f"    - CCI: {verdict_data['cci']}% (Linguistic: {phase_1_score:.2f}, Structural: {phase_2_score:.2f}, Statistical: {phase_3_score:.2f})")
                print(f"    - Verdict: {verdict_data['verdict']}")
                print(f"    - Action:  {verdict_data['action']}")

        planner.update_node_status("Generate_Report", "Completed")
        self.bus.update_status("ReportGenerator", "Idle")

        # ── Memory Logging ──────────────────────────────────
        self.memory.log(
            query=query,
            entry=parsed["entry_logic"],
            exit_logic=parsed["exit_logic"],
            status="Completed",
        )

        # ── Final Summary ──────────────────────────────────
        print_final_summary(report)

        # Cleanup
        self.memory.close()

        return report


def gatekeeper_verdict(phase_1_score: float, phase_2_score: float, phase_3_score: float) -> dict:
    """Calculates the Composite Confidence Index (CCI) and determines the final system action."""
    # Calculate CCI
    cci = (0.2 * phase_1_score) + (0.3 * phase_2_score) + (0.5 * phase_3_score)
    
    # Map to routing table
    if cci >= 0.85:
        verdict = "🟢 HIGH CONFIDENCE"
        action = "Approve for live paper trading. Capital allocation scaled to 100%."
    elif cci >= 0.70:
        verdict = "🟡 MARGINAL CONFIDENCE"
        action = "Hold for manual audit. Alert user to potential curve-fitting or missing parameters."
    else:
        verdict = "🔴 REJECTED"
        action = "Halt execution. Strategy rules are deemed statistical noise or unsafe."
        
    return {
        "cci": round(cci * 100, 2),
        "verdict": verdict,
        "action": action
    }


# ═══════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════
def interactive_mode():
    """Interactive CLI mode for the backtesting agent."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║            🤖  BACKTESTING AGENT  —  v1.0                    ║
║                                                              ║
║   Describe your trading strategy in plain English.           ║
║   The agent will parse, code, backtest, and analyze it.      ║
║                                                              ║
║   Type 'quit' to exit.                                       ║
╚══════════════════════════════════════════════════════════════╝
""")

    pipeline = BacktestingPipeline()

    while True:
        print("\n" + "─" * 60)
        query = input("📝 Enter your trading strategy:\n   → ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("   👋 Goodbye!")
            break
        if not query:
            print("   ⚠️  Please enter a strategy description.")
            continue

        tickers = input("📊 Enter ticker symbols (comma-separated):\n   → ").strip()
        if not tickers:
            print("   ⚠️  Please enter at least one ticker symbol.")
            continue

        market = input("🌍 Market? (IN / NSE / BSE) [default: IN]:\n   → ").strip() or "IN"

        pipeline.run(query, tickers, market)


def main():
    parser = argparse.ArgumentParser(
        description="Backtesting Agent — Natural Language to Backtest Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --query "Buy when RSI < 30, sell when RSI > 70" --tickers "RELIANCE,TCS" --market IN
        """,
    )
    parser.add_argument("--query", "-q", help="Trading strategy in natural language")
    parser.add_argument("--tickers", "-t", help="Comma-separated ticker symbols")
    parser.add_argument("--market", "-m", default="IN", help="Market: IN, NSE, BSE (default: IN)")

    args = parser.parse_args()

    if args.query and args.tickers:
        # Single-shot mode
        pipeline = BacktestingPipeline()
        pipeline.run(args.query, args.tickers, args.market)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
