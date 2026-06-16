# 🤖 Backtesting Agent: Autonomous Quantitative Research Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?style=flat-square&logo=google-gemini&logoColor=white" alt="LLM Engine" />
  <img src="https://img.shields.io/badge/Backtest-VectorBT-orange?style=flat-square" alt="Backtest Engine" />
  <img src="https://img.shields.io/badge/Database-DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black" alt="Database Cache" />
  <img src="https://img.shields.io/badge/Visuals-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Charts" />
</p>

---

## 📈 Overview

The **Backtesting Agent** is an autonomous quantitative researcher that bridges the gap between natural language trading ideas and validated, lookahead-bias-safe backtests. By converting plain English strategy descriptions into high-performance Python code using VectorBT, the agent automates the end-to-end research workflow—from multi-market data ingestion and DuckDB caching to portfolio execution and institutional-grade risk analysis.

---

## 🧭 High-Level Execution Gatekeeper

The pipeline is managed by a **Confidence Scoring Framework** that enforces strict gates before any strategy is executed or approved:

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
     │  Phase 3: Statistical │ ──► Calculates Sharpe p-value
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

> [!NOTE]
> The **Composite Confidence Index (CCI)** acts as the deployment routing gate:
> - **🟢 Approved (CCI >= 85%)**: Automatically verified and scaled to 100% capital allocation.
> - **🟡 Manual Audit (70% <= CCI < 85%)**: Strategy held for user intervention to fix phrasing or parameters.
> - **🔴 Rejected (CCI < 70%)**: Halted immediately to prevent execution of invalid logic or security violations.

---

## 🏗️ Detailed Architecture & Component Pillars

The system is structured as five sequential, graph-orchestrated modules.

### 🧠 Module 1: NLP & Orchestration ("The Brain")

Processes raw queries, validates symbols, schedules execution nodes, and persists session states.

<p align="center">
  <img src="docs/assets/NLP.png" alt="NLP Architecture" width="800" style="border-radius: 8px;" />
</p>

#### Core Pillars:
- **Linguistic Quality Gate**: Calculates a weighted score checking phrasing clarity and numerical indicator completeness:
  $$\text{Intent Score} = 0.4 \times \text{Linguistic Confidence} + 0.6 \times \text{Numerical Completeness}$$
- **DAG Scheduling**: Constructs dynamic execution plans using `networkx`. This allows concurrent data fetching tasks to run in parallel.
- **Schema Enforcement**: Employs `Pydantic` structures (`TradingStrategyWithConfidence`) to prevent model hallucination drift.
- **State Logging**: Records execution paths in a **Neo4j** graph database, utilizing a thread-safe local memory fallback if connection is offline.

---

### 📝 Module 2: Strategy Synthesis ("The Translator")

Translates natural language intentions into clean, vectorized Python code compliant with VectorBT.

<p align="center">
  <img src="docs/assets/Strategy%20Synthesis.png" alt="Strategy Synthesis Architecture" width="800" style="border-radius: 8px;" />
</p>

#### Core Pillars:
- **AST Transformation (Lookahead Guard)**: Uses Python's native `ast` compiler to rewrite the generated source tree. It automatically appends `.shift(1).fillna(False)` to `entries` and `exits` assignments, mathematically isolating the strategy from future-lookahead bias.
- **Self-Correction Compiler**: Runs `compile()` inside validation wrappers. If syntax errors occur, the traceback is piped back to Gemini for up to 3 automated self-healing retries.
- **Strict Code Sanitization**: The `CodeSanitizer` enforces an import whitelist (`pandas_ta`), maps indicator string casing, and rejects blacklisted system functions (`os`, `sys`, `eval`, etc.).
- **Predefined Skill Library**: Pre-loads modular snippets for major indicators (`RSI`, `SMA`, `EMA`, `MACD`, `BBANDS`, `STOCH`, `ATR`) to guide code precision.

---

### 💾 Module 3: Data & Market Connectivity ("The Heart")

Manages market adapters, cache synchronization, and transaction costs modeling.

<p align="center">
  <img src="docs/assets/Data%20Connectivity.png" alt="Data Connectivity Architecture" width="800" style="border-radius: 8px;" />
</p>

#### Core Pillars:
- **DuckDB OLAP Cache**: Caches historical market rates using an embedded columnar DuckDB database. Columnar storage facilitates instant, zero-copy conversion into Pandas DataFrames.
- **Warm-Up Padding**: Fetches an extra 100 days of historical records prior to the target start date. This prevents leading edge indicator distortions (e.g. EMAs) by trimming the warm-up padding right before signals are processed.
- **Adapters & Routing**: Employs `OpenAlgoAgent` to handle Indian stock tickers (mapping to `.NS` or `.BO` automatically) and fetches via Yahoo Finance.
- **Cost Attributors**: Integrates the `IndianEquityAdaptor` to provide hook-points for Indian market fee structures (STT, stamp duty, GST, and broker commissions).

---

### ⚙️ Module 4: Execution & Portfolio Engine ("The Engine")

Runs the synthesized code safely in sandbox processes and processes signal evaluations.

<p align="center">
  <img src="docs/assets/Execution%20Engine.png" alt="Execution Engine Architecture" width="800" style="border-radius: 8px;" />
</p>

#### Core Pillars:
- **Process Sandboxing**: Spawns executions in separate child processes with a strict 10-second timeout. This isolates the orchestrator process from infinite loops, runtime hangs, or memory leaks.
- **Casing & Proxy Wrappers**: Uses `CaseInsensitiveDF` and `PandasTAProxy` to dynamically catch and resolve case mismatching (e.g. `close` vs `Close`, `MACD_12_26_9` vs `macd_12_26_9`).
- **Leverage Sizing Adjuster**: Uses the `RiskAgent` to calculate Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR). If VaR exceeds limits (default: -2.0% daily), it dynamically scales down the strategy leverage:
  $$\text{Leverage Factor} = \frac{\text{Maximum Acceptable VaR}}{\text{Strategy VaR}}$$
- **Conflict & Exit Handling**: Runs VectorBT signal execution with conflict resolutions (`upon_long_conflict='ignore'`, etc.) and forces market exit triggers on the final bar.

---

### 📊 Module 5: Analytics & Audit ("The Eyes")

Runs statistics, evaluates transaction costs drag, and generates interactive reports.

<p align="center">
  <img src="docs/assets/Analytics.png" alt="Analytics Architecture" width="800" style="border-radius: 8px;" />
</p>

#### Core Pillars:
- **KPI Matrix**: Calculates Sharpe, Sortino, Calmar, profit factor, and drawdowns. Sortino computation adjusts standard errors for single-day negative variances to avoid division-by-zero crashes.
- **Statistical P-Value Gate**: Computes the Sharpe standard error and statistical p-value to reject luck:
  $$\text{Sharpe Error} = \sqrt{\frac{1 + \frac{\text{Sharpe}^2}{2}}{N_{\text{days}}}}$$
  $$\text{T-Stat} = \frac{\text{Sharpe}}{\text{Sharpe Error}}$$
- **Dual Visualization Engine**: Generates interactive HTML Plotly dashboards (Equity curve, Drawdowns, and returns distributions) along with QuantStats Tearsheets comparing returns against the Nifty 50 index (`^NSEI`).
- **Attribution Audit**: Calculates gross returns, net returns, and fee drags:
  $$\text{Cost Drag} = \frac{\text{Total Transaction Fees}}{\text{Initial Cash}}$$
  Flags warnings if costs consume $> 50\%$ of the gross strategy return.

---

## 🔄 End-to-End Pipeline Execution

```mermaid
flowchart LR
    A[Natural Language Query] --> B[NLP Module: Validate & Plan]
    B --> C[CodeGen Module: AST Sanitization & Compile Check]
    C --> D[Data Module: Warm-up Padding & Cache Sync]
    D --> E[Execution Module: Process Sandbox & VectorBT]
    E --> F[Analytics Module: Risk Metrics & Plotly HTML Reports]
```

1. **Phase 1 Validation**: `QueryInterpreter` checks intent and parses tickers.
2. **Task Graphing**: `DAGPlanner` maps dependencies and runs data routing.
3. **Phase 2 Sanitization**: `AlphaAgent` codes the rules, applying AST lookahead guards and compile-checking the code.
4. **Sandboxed Testing**: `BacktestEngine` runs the simulation, trims padding, and calculates VaR.
5. **Phase 3 Statistics & Dashboarding**: `PerformanceAnalyzer` scores statistical viability, exports HTML logs, and prints the summary.

---

## 🛠️ Technology Stack & Mapping

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Orchestrator** | Python 3.10+ | Robust support for quantitative analysis libraries. |
| **LLM Reasoning** | Google Gemini | Fast generation speeds with native Pydantic schema validation. |
| **Dependency Graphs** | NetworkX | Simple DAG scheduling and topological task sorting. |
| **Vector Engine** | VectorBT | Incredibly fast vectorized matrix simulations built with Numba. |
| **Analytical Cache** | DuckDB | Embeddable OLAP database with zero-copy Pandas integration. |
| **Technical math** | NumPy & SciPy | High-performance linear algebra and statistical calculators. |
| **Report Cards** | QuantStats & Plotly | Polished tearsheets and interactive visual dashboards. |

---

## ⚙️ Configuration & Setup

### 1. Environment Config
Create a `.env` file in the project root:
```env
# API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

# Optional Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Configuration Defaults
DEFAULT_INIT_CASH=100000
DEFAULT_TIMEFRAME=1d
DEFAULT_MARKET=IN
```

### 2. Quickstart installation
```bash
# Clone the repository
git clone https://github.com/A-RYAN-KR/Backtesting-Agent.git
cd Backtesting-Agent

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### Interactive CLI Loop
Run the interactive console to test queries sequentially:
```bash
python main.py
```
> **Example prompt:** *"Buy RELIANCE when RSI < 30, sell when RSI > 70 for the last 2 years"*

### Single-shot Mode
Run a query immediately from the terminal:
```bash
python main.py --query "Buy TCS when SMA 20 crosses above SMA 50 for the last 1 year" --market "IN"
```

---

## 📂 Repository Structure

<details>
<summary><b>Click to expand project directory tree</b></summary>

```
.
├── config.py             # Configuration loader & defaults
├── main.py               # Unified pipeline runner & CLI entrypoint
├── README.md             # Aesthetic documentation & system blueprints
├── requirements.txt      # Required package list
├── .env                  # Secrets configuration
├── cache/                # DuckDB analytical cache database directory
│   └── trading_cache.db
├── reports/              # Target directory for exported Plotly & Tearsheets
└── modules/
    ├── __init__.py
    ├── nlp_orchestration.py   # Module 1: Query validator, planner, & Neo4j Memory
    ├── strategy_synthesis.py  # Module 2: LLM programmer, AST rewriter, & Sanitizer
    ├── data_connectivity.py   # Module 3: DuckDB interface & Yahoo Finance fetcher
    ├── execution_engine.py    # Module 4: Multiprocessing executor & RiskAgent
    └── analytics_audit.py     # Module 5: KPI analyzer & Plotly report card generator
```
</details>
