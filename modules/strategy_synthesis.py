"""
Module 2: Strategy Synthesis & Skill Library ("The Translator")
──────────────────────────────────────────────────────────────
• SkillLibrary   – Pre-defined indicator templates
• CodeSanitizer  – Enforces shift(1) and validates TA usage
• AlphaAgent     – LLM-driven strategy code generation
"""

import re
import json
import ast
from config import GEMINI_MODEL
from modules.nlp_orchestration import stream_chat_completion

client = None


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
class EntryExitShifter(ast.NodeTransformer):
    """
    Appends lookahead bias guards and fillna attributes to BOTH entry 
    and exit conditions uniformly to prevent same-bar directional deadlocks.
    """

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            # FIX: Apply lookahead alignment shift to both vectors
            if name in ('entries', 'exits'):
                new_value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Attribute(value=node.value, attr='shift', ctx=ast.Load()),
                            args=[ast.Constant(value=1)],
                            keywords=[]
                        ),
                        attr='fillna',
                        ctx=ast.Load()
                    ),
                    args=[ast.Constant(value=False)],
                    keywords=[]
                )
                node.value = new_value
        return node


class CodeSanitizer:
    """
    Ensures generated code is safe and free of lookahead bias.

    Shift policy:
      • ENTRY & EXIT signals → shift(1) applied to prevent same-bar
        directional deadlocks and ensure consistent execution boundaries.
    """

    # Allowlisted import modules — everything else is blocked
    ALLOWED_IMPORTS = {"pandas_ta"}

    @staticmethod
    def enforce_entry_shift(code_string: str) -> str:
        """
        Applies .shift(1).fillna(False) to both entry and exit signals.
        Uses AST parsing to safely traverse and modify entry and exit signals.
        """
        try:
            tree = ast.parse(code_string)
            transformer = EntryExitShifter()
            modified_tree = transformer.visit(tree)
            ast.fix_missing_locations(modified_tree)
            return ast.unparse(modified_tree)
        except Exception as e:
            print(f"  [WARNING] AST parsing failed during entry shift enforcement: {e}. Falling back to regex.")
            # Simple fallback using regular expressions
            def shift_signals(match):
                prefix = match.group(1)
                expression = match.group(2).strip()
                if ".shift(" in expression:
                    return f"{prefix} {expression}"
                return f"{prefix} ({expression}).shift(1).fillna(False)"

            code = re.sub(r'(entries\s*=\s*)([^#\n]+)', shift_signals, code_string)
            code = re.sub(r'(exits\s*=\s*)([^#\n]+)', shift_signals, code)
            return code

    @staticmethod
    def check_security(code_string: str) -> None:
        """
        Validates that the code does not contain blocked keywords to prevent RCE.
        """
        blocked_keywords = ["import os", "sys", "eval", "open", "__import__"]
        for kw in blocked_keywords:
            # Word boundary regex checks to prevent false positives (like 'openalgo', 'system', 'open_prices')
            # while strictly blocking 'import os', 'sys', 'eval', 'open', '__import__'
            pattern = r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, code_string):
                raise ValueError(f"Security Error: Blocked keyword '{kw}' detected.")

    @staticmethod
    def enforce_import_allowlist(code_string: str) -> str:
        """
        Uses AST to reject all import statements except those in ALLOWED_IMPORTS.
        Strips disallowed imports from the code instead of raising, to gracefully
        handle LLMs that import datetime, math, etc.
        """
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            # If code can't be parsed, skip this check (compile check will catch it later)
            return code_string

        blocked_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split('.')[0]
                    if module_root == "pandas_ta":
                        blocked_lines.add(node.lineno)
                        print(f"  [WARNING] Stripped pandas_ta import: 'import {alias.name}'")
                    elif module_root not in CodeSanitizer.ALLOWED_IMPORTS:
                        blocked_lines.add(node.lineno)
                        print(f"  [WARNING] Stripped disallowed import: 'import {alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split('.')[0]
                    if module_root == "pandas_ta":
                        blocked_lines.add(node.lineno)
                        print(f"  [WARNING] Stripped pandas_ta import: 'from {node.module} import ...'")
                    elif module_root not in CodeSanitizer.ALLOWED_IMPORTS:
                        blocked_lines.add(node.lineno)
                        print(f"  [WARNING] Stripped disallowed import: 'from {node.module} import ...'")

        if not blocked_lines:
            return code_string

        # Remove blocked lines
        lines = code_string.split('\n')
        cleaned_lines = [line for i, line in enumerate(lines, 1) if i not in blocked_lines]
        return '\n'.join(cleaned_lines)

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
        """
        Full sanitization pipeline.
        Raises SyntaxError if the final code does not compile.
        """
        # Check security first
        CodeSanitizer.check_security(code_string)
        # Remove markdown fences if LLM included them
        code = code_string.replace("```python", "").replace("```", "").strip()
        # Enforce import allowlist (strip disallowed imports via AST)
        code = CodeSanitizer.enforce_import_allowlist(code)
        # Normalize column access casing (fixes pandas_ta column name mismatches)
        code = CodeSanitizer.normalize_column_access(code)
        code = CodeSanitizer.enforce_entry_shift(code)
        # Validate that entries and exits variables are explicitly assigned in the AST
        # This prevents LLM hallucinations of variables like buy_signals/sell_signals.
        try:
            tree = ast.parse(code)
            has_entries = False
            has_exits = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "entries":
                            has_entries = True
                        elif isinstance(target, ast.Name) and target.id == "exits":
                            has_exits = True
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name) and node.target.id == "entries":
                        has_entries = True
                    elif isinstance(node.target, ast.Name) and node.target.id == "exits":
                        has_exits = True
            
            if not has_entries or not has_exits:
                missing = []
                if not has_entries:
                    missing.append("entries")
                if not has_exits:
                    missing.append("exits")
                raise ValueError(f"Strategy code must explicitly assign to: {', '.join(missing)}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise SyntaxError(f"AST parsing failed: {e}")

        # Pre-execution syntax check — catch broken code before it hits exec()
        try:
            compile(code, '<llm_generated>', 'exec')
        except SyntaxError as e:
            raise SyntaxError(f"LLM generated invalid Python (line {e.lineno}): {e.msg}") from e
        return code


# ═══════════════════════════════════════════════════════════
#  Alpha Agent
# ═══════════════════════════════════════════════════════════
class AlphaAgent:
    """Converts structured intent into executable VectorBT/pandas_ta code."""

    MAX_RETRIES = 3

    def generate_strategy_code(self, parsed_intent: dict) -> str:
        """Generates strategy code from parsed NLP intent. Retries on syntax failures."""

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
        10. Do NOT import any other modules (datetime, math, etc.). They are NOT available.
        11. RISK TARGETS RULE: Never write mathematical expressions checking stop_loss or take_profit conditions.
        If the strategy mentions risk targets or stop losses, do NOT append them to the 'exits' boolean series.
        Keep 'exits' bound purely to technical indicator conditions.

        PANDAS_TA COLUMN NAMING (CRITICAL — follow exactly):
        - All column names are LOWERCASED. Always access columns in lowercase.
        - RSI returns a Series directly: `rsi = pandas_ta.rsi(close_prices, length=14)`
        - SMA returns a Series directly: `sma = pandas_ta.sma(close_prices, length=20)`
        - EMA returns a Series directly: `ema = pandas_ta.ema(close_prices, length=12)`
        - NEVER hardcode string keys for DataFrames returned by indicators like BBANDS or MACD because the suffixes change. 
        - Access columns by position using `.iloc`. 
          Example for BBANDS: 
          bb = pandas_ta.bbands(close_prices, length=20, std=2)
          lower = bb.iloc[:, 0]  # bbl is always column 0
          middle = bb.iloc[:, 1] # bbm is always column 1
          upper = bb.iloc[:, 2]  # bbu is always column 2
        - Example for MACD:
          macd_df = pandas_ta.macd(close_prices, fast=12, slow=26, signal=9)
          macd_line = macd_df.iloc[:, 0]    # macd line is always column 0
          histogram = macd_df.iloc[:, 1]    # histogram is always column 1
          signal_line = macd_df.iloc[:, 2]  # signal line is always column 2
        - STOCH returns a DataFrame with columns: 'stochk_14_3_3', 'stochd_14_3_3' (use positional access `.iloc` for these as well)

        CROSSOVER LOGIC (CRITICAL):
        - Pandas Series DO NOT have `.cross_above()` or `.cross_below()` methods. Do NOT hallucinate them.
        - If the logic requires Line A to cross above Line B, you MUST write: 
          `entries = (A > B) & (A.shift(1) <= B.shift(1))`
        - If the logic requires Line A to cross below Line B, you MUST write: 
          `exits = (A < B) & (A.shift(1) >= B.shift(1))`

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

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Build messages — include error feedback on retries
                messages = [{"role": "user", "content": prompt}]
                if last_error and attempt > 1:
                    messages.append({
                        "role": "user",
                        "content": f"Your previous code had a syntax error: {last_error}. "
                                   f"Please fix it and output ONLY valid Python code. No markdown, no explanations."
                    })

                raw_code = stream_chat_completion(
                    client=client,
                    model=GEMINI_MODEL,
                    messages=messages,
                    print_stream=True
                )
                safe_code = CodeSanitizer.sanitize(raw_code)

                is_valid, msg = CodeSanitizer.validate_indicators(safe_code)
                if not is_valid:
                    print(f"  ⚠️  {msg}")

                return safe_code

            except SyntaxError as e:
                last_error = str(e)
                print(f"\n  ⚠️  Code generation attempt {attempt}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    print(f"  🔄  Retrying code generation ...")
                    continue
                print(f"  ❌  All {self.MAX_RETRIES} code generation attempts failed.")
                return f"# Error: LLM generated invalid Python after {self.MAX_RETRIES} attempts. Last error: {e}"
            except Exception as e:
                return f"# Error generating code: {e}"
