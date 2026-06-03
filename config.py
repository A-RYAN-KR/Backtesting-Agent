"""
config.py — Central configuration for the Backtesting Agent.
Loads environment variables and provides system-wide defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ───────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")


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

# ─── OpenRouter Model Config ────────────────────────────────
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# ─── Gemini Model Config ────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

