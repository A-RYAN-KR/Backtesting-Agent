"""
config.py — Central configuration for the Backtesting Agent.
Loads environment variables and provides system-wide defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ───────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── Neo4j (Optional) ──────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ─── Defaults ───────────────────────────────────────────────
DEFAULT_INIT_CASH = float(os.getenv("DEFAULT_INIT_CASH", 100000))
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1d")
DEFAULT_MARKET = os.getenv("DEFAULT_MARKET", "IN")

# ─── Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─── Gemini Model Config ────────────────────────────────────
GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"       # Validation, quick tasks
GEMINI_MODEL_PRO = "gemini-2.5-flash-lite"             # Code generation, complex reasoning
