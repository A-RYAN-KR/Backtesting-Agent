# 🤖 Backtesting Agent: Project Documentation

## 1. Problem Statement
In the world of quantitative finance, the gap between a **trading idea** (natural language) and a **validated backtest** (executable code) is significant. Traders often face:
- **Coding Barriers**: Implementing complex vectorized logic (like VectorBT) requires high technical proficiency.
- **Lookahead Bias**: Manual implementation often leads to "cheating" by using future data in entry signals.
- **Data Fragmentation**: Fetching, cleaning, and caching multi-market data is a repetitive and error-prone process.
- **Incomplete Analytics**: Many tools provide raw returns but fail to account for market-specific costs (STT, GST in India) or provide deep risk metrics like VaR and CVaR.

## 2. Objectives
The Backtesting Agent is designed to be an **autonomous quantitative researcher** that:
- **Bridges the Gap**: Converts plain English strategy descriptions into high-performance Python code.
- **Ensures Integrity**: Automatically applies lookahead-bias guards (shifts) and sanitizes code logic.
- **Automates Data**: Manages the end-to-end data lifecycle from fetching (YFinance) to local storage (DuckDB).
- **Professional Grade Analytics**: Generates institutional-level reports including interactive equity curves and factor attribution.

## 3. Architecture & Confidence Scoring Framework
The system follows a **Modular Agentic Architecture** with a built-in **Confidence Scoring Framework** that acts as an Execution Gatekeeper Pipeline:

```
           [User Prompt]
                 │
                 ▼
     ┌───────────────────────┐
     │  Phase 1: Linguistic  │ < 0.70 ──► [Halt / Clarify parameters]
     └───────────────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │  Phase 2: Structural  │ < 0.70 ──► [AlphaAgent Auto-Retry (Max 3x)]
     └───────────────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │  Phase 3: Statistical │ ──► Calculates p-value & OOS Decay
     └───────────────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │   Gatekeeper Engine   │ ──► CCI = 0.2*L + 0.3*S + 0.5*Stat
     └───────────────────────┘
                 │
         ┌───────┼───────┐
         ▼               ▼
    [CCI >= 85%]   [70% <= CCI < 85%]   [CCI < 70%]
     🟢 Approved     🟡 Manual Audit     🔴 Rejected
```

### Core Modules:
1.  **NLP & Orchestration (The Brain)**: 
    - Uses Gemini LLM to validate and interpret queries.
    - Employs a self-reflective schema (`TradingStrategyWithConfidence`) to calculate Phase 1 intent score (based on phrasing clarity and numerical completeness). Halts if score < 0.70.
    - Generates a **DAG (Directed Acyclic Graph)** to plan execution steps.
    - Manages a **Memory Agent** (Neo4j) to track historical strategy performance.
2.  **Strategy Synthesis (The Translator)**: 
    - Translates intent into **VectorBT** code.
    - Features a **Skill Library** of predefined technical indicators.
    - Employs a **Code Sanitizer** to enforce signal rules, lookahead bias guards (`.shift(1)`), and casing normalization.
    - Implements Phase 2 structural checks to grade code safety and retries LLM generation up to 3 times if confidence score < 0.70.
3.  **Data & Market Connectivity (The Heart)**: 
    - Routes requests to appropriate data providers (YFinance, OpenAlgo).
    - Implements **DuckDB** for high-speed local caching.
    - Includes an **Indian Equity Adaptor** for localized cost modeling (Brokerage, STT, etc.).
4.  **Execution Engine (The Engine)**: 
    - Runs vectorized backtests.
    - Injects a **PandasTAProxy** to handle case-insensitive indicator access.
    - Applies dynamic slippage based on historical volatility.
5.  **Analytics & Audit (The Eyes)**: 
    - Computes Sharpe, Sortino, Calmar, and Drawdown.
    - Implements Phase 3 statistical checks: computes the Sharpe standard error and p-value, and runs Out-of-Sample (OOS) decay ratio checks to detect curve-fitting.
    - Performs **Audit Attribution** (Alpha vs. Cost Drag).
    - Generates HTML reports via **Plotly** and **QuantStats**.

## 4. Workflow
The pipeline operates in five distinct stages:

1.  **Intent Extraction & Verification (Phase 1)**: The user enters a query (e.g., *"Buy when RSI < 30, Sell when RSI > 70"*). The **QueryInterpreter** extracts structured rules and calculates linguistic intent confidence. If intent score is < 0.70, it halts and asks for clarification.
2.  **DAG Planning**: The **DAGPlanner** identifies dependencies (e.g., data must be fetched before the backtest can run) and creates a task graph.
3.  **Synthesis, Sanitization & Code Confidence (Phase 2)**: The **AlphaAgent** writes the VectorBT logic. The **CodeSanitizer** applies lookahead-bias guards, checks security rules, and calculates structural code confidence. If code confidence is < 0.70, the agent automatically triggers correction retries.
4.  **Vectorized Execution**: The **BacktestEngine** executes the logic across all requested tickers in parallel, accounting for slippage and transaction fees.
5.  **Analytics, Statistical Confidence & CCI Gate (Phase 3 & Global Verdict)**: The system computes performance metrics, Sharpe error/p-value, and OOS decay ratio (Phase 3). It calculates the Composite Confidence Index (CCI) and determines the final verdict: Approve for trading (CCI >= 85%), Hold for manual audit (70% - 84%), or Reject/Halt (CCI < 70%). Interactive reports are saved to the `/reports` directory.

## 5. Future Scope
- **Real-Time Integration**: Connecting the "AlphaAgent" output directly to brokers (like Zerodha/AngelOne) for live paper trading.
- **Strategy Optimization**: Implementing a "Hyperparameter Agent" to automatically tune indicator windows (e.g., finding the best RSI length).
- **Multi-Asset Support**: Expanding the data router to handle Crypto, Forex, and Options data.
- **LLM-Based Refinement**: Using past backtest results (from Neo4j) to suggest improvements to the user's natural language strategy.

## 6. Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **LLM (Orchestration)** | Google Gemini (Gemini 2.5 Flash/Pro) |
| **Backtesting Framework** | VectorBT |
| **Data Fetching** | YFinance, OpenAlgo |
| **Database (Cache)** | DuckDB |
| **Database (Memory)** | Neo4j (Graph Database) |
| **Indicators** | Pandas-TA, TA-Lib |
| **Visualization** | Plotly, QuantStats |
| **Validation** | Pydantic |
