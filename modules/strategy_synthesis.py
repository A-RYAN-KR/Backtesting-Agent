"""
Module 2: Strategy Synthesis & Skill Library ("The Translator")
──────────────────────────────────────────────────────────────
• SkillLibrary   – Pre-defined indicator templates
• CodeSanitizer  – Enforces shift(1) and validates TA usage
• AlphaAgent     – LLM-driven strategy code generation
"""

import re
import json
from google import genai
from google.genai import types

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GEMINI_API_KEY, GEMINI_MODEL_PRO

client = genai.Client(api_key=GEMINI_API_KEY)


# ═══════════════════════════════════════════════════════════
#  Skill Library
# ═══════════════════════════════════════════════════════════
class SkillLibrary:
    """Repository of pre-defined indicator skills using pandas_ta / TA-Lib."""

    SKILLS = {
        "RSI": {
            "description": "Relative Strength Index (Momentum)",
            "pandas_ta": "pandas_ta.rsi(close_prices, length={window})",
            "talib": "talib.RSI(close_prices, timeperiod={window})",
        },
        "SMA": {
            "description": "Simple Moving Average (Trend)",
            "pandas_ta": "pandas_ta.sma(close_prices, length={window})",
            "talib": "talib.SMA(close_prices, timeperiod={window})",
        },
        "EMA": {
            "description": "Exponential Moving Average (Trend)",
            "pandas_ta": "pandas_ta.ema(close_prices, length={window})",
            "talib": "talib.EMA(close_prices, timeperiod={window})",
        },
        "MACD": {
            "description": "Moving Average Convergence Divergence",
            "pandas_ta": "pandas_ta.macd(close_prices, fast=12, slow=26, signal=9)",
            "talib": "talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)",
        },
        "BBANDS": {
            "description": "Bollinger Bands (Volatility)",
            "pandas_ta": "pandas_ta.bbands(close_prices, length=20, std=2)",
            "talib": "talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)",
        },
        "STOCH": {
            "description": "Stochastic Oscillator",
            "pandas_ta": "pandas_ta.stoch(high, low, close)",
            "talib": "talib.STOCH(high, low, close)",
        },
        "ATR": {
            "description": "Average True Range (Volatility)",
            "pandas_ta": "pandas_ta.atr(high, low, close, length=14)",
            "talib": "talib.ATR(high, low, close, timeperiod=14)",
        },
    }

    @staticmethod
    def get_available_skills() -> dict:
        return SkillLibrary.SKILLS

    @staticmethod
    def format_skill_context() -> str:
        lines = ["AVAILABLE INDICATOR SKILLS (use pandas_ta library):"]
        for name, detail in SkillLibrary.SKILLS.items():
            lines.append(f"  - {name}: {detail['pandas_ta']}  |  {detail['description']}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Code Sanitizer
# ═══════════════════════════════════════════════════════════
class CodeSanitizer:
    """
    Ensures generated code is safe and free of lookahead bias.

    Shift policy:
      • ENTRY signals  → shift(1) applied to prevent buying on the
        same bar the condition fires (lookahead-bias guard).
      • EXIT signals   → NOT shifted.  An exit should execute as soon
        as the condition is true so that stop-losses and take-profits
        are not artificially delayed by one bar.
    """

    @staticmethod
    def enforce_entry_shift(code_string: str) -> str:
        """
        Applies .shift(1) ONLY to entry signals to prevent lookahead bias.
        Exit signals are left untouched so positions can close immediately.
        """
        def shift_entries(match):
            prefix = match.group(1)          # 'entries =' or 'entries='
            expression = match.group(2).strip()
            # If the LLM already applied a shift, don't double-shift
            if ".shift(" in expression:
                return f"{prefix} {expression}"
            return f"{prefix} ({expression}).shift(1).fillna(False)"

        def clean_exits(match):
            prefix = match.group(1)
            expression = match.group(2).strip()
            # Leave exits as-is (no shift); just ensure fillna for NaN safety
            if ".fillna(" in expression:
                return f"{prefix} {expression}"
            return f"{prefix} ({expression}).fillna(False)"

        code = re.sub(r'(entries\s*=\s*)([^#\n]+)', shift_entries, code_string)
        code = re.sub(r'(exits\s*=\s*)([^#\n]+)', clean_exits, code)
        return code

    @staticmethod
    def normalize_column_access(code_string: str) -> str:
        """
        Lowercases column name strings in bracket-style DataFrame access.
        Handles pandas_ta output columns like MACD_12_26_9 → macd_12_26_9.

        This catches patterns like:
          df['MACDS_12_26_9']  →  df['macds_12_26_9']
          df["BBU_20_2.0"]    →  df["bbu_20_2.0"]
        """
        def lower_single_quote(match):
            return f"['{match.group(1).lower()}']"

        def lower_double_quote(match):
            return f'["{match.group(1).lower()}"]'

        # Match bracket access with single or double quotes containing
        # patterns that look like indicator columns (uppercase + underscores + digits)
        code = re.sub(r"\['([A-Z][A-Za-z0-9_.]+)'\]", lower_single_quote, code_string)
        code = re.sub(r'\["([A-Z][A-Za-z0-9_.]+)"\]', lower_double_quote, code)
        return code

    @staticmethod
    def validate_indicators(code_string: str) -> tuple[bool, str]:
        """Ensures the code uses pandas_ta or talib."""
        if "pandas_ta" in code_string or "talib" in code_string or "ta." in code_string:
            return True, "Indicator library usage validated."
        return False, "Warning: Code may not use a standard indicator library."

    @staticmethod
    def sanitize(code_string: str) -> str:
        """Full sanitization pipeline."""
        # Remove markdown fences if LLM included them
        code = code_string.replace("```python", "").replace("```", "").strip()
        # Normalize column access casing (fixes pandas_ta column name mismatches)
        code = CodeSanitizer.normalize_column_access(code)
        code = CodeSanitizer.enforce_entry_shift(code)
        return code


# ═══════════════════════════════════════════════════════════
#  Alpha Agent
# ═══════════════════════════════════════════════════════════
class AlphaAgent:
    """Converts structured intent into executable VectorBT/pandas_ta code."""

    def generate_strategy_code(self, parsed_intent: dict) -> str:
        """Generates strategy code from parsed NLP intent."""

        skill_context = SkillLibrary.format_skill_context()

        prompt = f"""
        You are a quantitative trading architect.
        Generate Python code for a VectorBT strategy based on the following parsed intent.

        Parsed Intent:
        - Entry Logic: {parsed_intent.get('entry_logic', '')}
        - Exit Logic: {parsed_intent.get('exit_logic', '')}

        RULES:
        1. Assume `close_prices` is already provided as a Pandas Series (pd.Series).
        2. Assume `high_prices`, `low_prices`, `volume` are also available as Pandas Series if needed.
        3. {skill_context}
        4. Use `import pandas_ta` for indicator calculations.
        5. Do NOT wrap the code in a function. Just provide the variable assignments.
        6. Define the variables `entries` and `exits` as boolean Pandas Series.
        7. Output ONLY raw Python code. No markdown formatting, no explanations, no comments.
        8. Do NOT include any import statements for pandas or numpy; they are already imported.
        9. Only import pandas_ta if you use it.

        PANDAS_TA COLUMN NAMING (CRITICAL — follow exactly):
        - All column names are LOWERCASED. Always access columns in lowercase.
        - RSI returns a Series directly: `rsi = pandas_ta.rsi(close_prices, length=14)`
        - SMA returns a Series directly: `sma = pandas_ta.sma(close_prices, length=20)`
        - EMA returns a Series directly: `ema = pandas_ta.ema(close_prices, length=12)`
        - MACD returns a DataFrame with columns: 'macd_12_26_9', 'macdh_12_26_9', 'macds_12_26_9'
          Example: `macd_df = pandas_ta.macd(close_prices, fast=12, slow=26, signal=9)`
                   `macd_line = macd_df['macd_12_26_9']`
                   `signal_line = macd_df['macds_12_26_9']`
        - BBANDS returns a DataFrame with columns: 'bbl_20_2.0', 'bbm_20_2.0', 'bbu_20_2.0', 'bbb_20_2.0', 'bbp_20_2.0'
          Example: `bb = pandas_ta.bbands(close_prices, length=20, std=2)`
                   `upper = bb['bbu_20_2.0']`
                   `lower = bb['bbl_20_2.0']`
        - STOCH returns a DataFrame with columns: 'stochk_14_3_3', 'stochd_14_3_3'

        SIGNAL RULES (CRITICAL — follow exactly):
        - Do NOT apply .shift() to entries or exits. Shifting is handled downstream.
        - `entries` and `exits` should be pure boolean conditions based on the indicators.
        - Each signal must be independent — the downstream engine handles state tracking
          (i.e. it will ignore exit signals when not in a position, and ignore entry
          signals when already in a position).
        - Keep signals simple: entries = condition_to_buy, exits = condition_to_sell.

        Example output format:
        import pandas_ta
        rsi = pandas_ta.rsi(close_prices, length=14)
        entries = rsi < 30
        exits = rsi > 70
        """

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_PRO,
                contents=prompt,
            )
            raw_code = response.text
            safe_code = CodeSanitizer.sanitize(raw_code)

            is_valid, msg = CodeSanitizer.validate_indicators(safe_code)
            if not is_valid:
                print(f"  ⚠️  {msg}")

            return safe_code
        except Exception as e:
            return f"# Error generating code: {e}"
