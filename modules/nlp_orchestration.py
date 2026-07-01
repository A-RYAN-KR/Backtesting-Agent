"""
Module 1: NLP & Orchestration Layer ("The Brain")
──────────────────────────────────────────────────
• QueryValidator   – Validates trading queries via Gemini LLM
• QueryInterpreter – Chain-of-Thought parsing → structured strategy
• DAGPlanner       – Builds execution DAG for task orchestration
• RegistrationBus  – Tracks agent health and availability
• MemoryAgent      – Logs execution traces (Neo4j or simulation)
"""

import json
import re
from datetime import datetime
import time
from pydantic import BaseModel, Field, ValidationError
import networkx as nx
from config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import urllib.request
import urllib.error


import sys
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


# ─── Configure client placeholder for compatibility ────────
client = None

def _get_schema(model) -> str:
    if hasattr(model, "model_json_schema"):
        return json.dumps(model.model_json_schema(), indent=2)
    return json.dumps(model.schema(), indent=2)

def _strip_markdown_json(text: str) -> str:
    """Strip markdown ```json ... ``` fences that LLMs sometimes wrap around JSON responses."""
    stripped = text.strip()
    # Match ```json or ``` at start, ``` at end
    match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```$', stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped

def _safe_extract_text(res_json: dict) -> str:
    """
    Safely extract text content from a Gemini API response.
    Handles safety filter blocks, empty candidates, and missing parts gracefully.
    """
    if 'candidates' not in res_json or len(res_json['candidates']) == 0:
        # Check if the response was blocked by safety filters
        block_reason = res_json.get('promptFeedback', {}).get('blockReason', '')
        safety_ratings = res_json.get('promptFeedback', {}).get('safetyRatings', [])
        if block_reason:
            raise RuntimeError(f"Gemini API blocked the request. Reason: {block_reason}. Safety ratings: {safety_ratings}")
        raise RuntimeError(f"Gemini API returned no candidates. Full response: {json.dumps(res_json)[:500]}")

    candidate = res_json['candidates'][0]

    # Check candidate-level finish reason for blocks
    finish_reason = candidate.get('finishReason', '')
    if finish_reason == 'SAFETY':
        safety_ratings = candidate.get('safetyRatings', [])
        raise RuntimeError(f"Gemini API response blocked by safety filters. Ratings: {safety_ratings}")

    content = candidate.get('content', {})
    parts = content.get('parts', [])
    if not parts:
        if finish_reason and finish_reason != 'STOP':
            raise RuntimeError(f"Gemini API response has no content parts. Finish reason: {finish_reason}")
        return ""

    return parts[0].get('text', '')

def stream_chat_completion(client=None, model=None, messages=None, temperature=1.0, top_p=1.0, max_tokens=16384, response_format=None, response_schema=None, print_stream=True):
    target_model = GEMINI_MODEL
    if model and ("gemini" in model or model == "gemini-flash-latest"):
        target_model = model

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    # Format the messages payload for Gemini API
    contents = []
    system_instruction = None
    if messages:
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })

    # Prepare generation configuration
    generation_config = {
        "temperature": temperature,
        "topP": top_p,
    }
    if max_tokens:
        generation_config["maxOutputTokens"] = max_tokens

    if response_schema:
        if hasattr(response_schema, "model_json_schema"):
            schema_dict = response_schema.model_json_schema()
        elif hasattr(response_schema, "schema"):
            schema_dict = response_schema.schema()
        else:
            schema_dict = response_schema
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = schema_dict
    elif response_format and response_format.get("type") == "json_object":
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": contents
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction
    if generation_config:
        payload["generationConfig"] = generation_config

    url = f"{GEMINI_BASE_URL}/models/{target_model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req) as response:
                res = response.read().decode("utf-8")
                res_json = json.loads(res)
                
                # Safely extract text — handles safety blocks, empty candidates, missing parts
                text_content = _safe_extract_text(res_json)

                if print_stream and text_content:
                    print(text_content, end="", flush=True)

                return text_content
                
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            if e.code in (429, 503) and attempt < max_retries - 1:
                print(f"\n⚠️ Gemini API returned HTTP {e.code} (Overloaded/Rate-limited). Retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries}) ...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            print(f"\n❌ Gemini API HTTP Error {e.code}: {err_msg}")
            raise RuntimeError(f"Gemini API call failed: {e.code} - {err_msg}")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️ Gemini API call failed with exception: {e}. Retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries}) ...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            print(f"\n❌ Gemini API General Error: {e}")
            raise e



# ═══════════════════════════════════════════════════════════
#  Pydantic schemas for structured LLM output
# ═══════════════════════════════════════════════════════════
class ValidationResult(BaseModel):
    is_valid: bool
    error_message: str


class TradingStrategyWithConfidence(BaseModel):
    reasoning: str = Field(description="Chain of thought analysis of the user rules.")
    linguistic_confidence: float = Field(description="Score between 0.0 and 1.0 indicating clarity of phrasing and use of standard financial terms.")
    numerical_completeness: float = Field(description="Score between 0.0 and 1.0 indicating if all indicators have strict, explicit numbers.")
    entry_logic: str
    exit_logic: str
    duration: str
    tickers: list[str]
    capital_allocation: str = Field(description="Details on how capital should be split among the tickers (e.g., '50% RELIANCE, 50% TCS'). Default to '100% per ticker' if not specified.")

TradingStrategy = TradingStrategyWithConfidence


# ═══════════════════════════════════════════════════════════
#  Query Validator
# ═══════════════════════════════════════════════════════════
class QueryValidator:
    """Validates the trading query for missing data, vagueness, or fake indicators."""

    def validate(self, query: str, tickers: str | None = None) -> dict:
        # Static hard checks
        if not query or len(query.strip()) < 10:
            return {"is_valid": False, "error_message": "Query is too short or missing."}
        if tickers is not None and len(tickers.strip()) == 0:
            return {"is_valid": False, "error_message": "No tickers provided. Please specify at least one symbol."}

        schema_str = _get_schema(ValidationResult)
        tickers_part = f'\n        Tickers: "{tickers}"' if tickers else ''
        prompt = f"""
        You are a strict validation agent for a quantitative trading system.
        Review the following user query.

        Rules for Validity:
        1. The query MUST contain actionable trading logic (e.g., conditions for buying/selling).
        2. The technical indicators mentioned MUST be real, standard financial indicators (e.g., RSI, MACD, SMA, Bollinger Bands).
        3. The query MUST mention or imply ticker symbols to trade (e.g., RELIANCE, TCS, NIFTY, ^NSEI, etc.) unless they are specified separately.
        4. If the query uses made-up indicators, is completely vague, or does not specify ticker symbols, it is INVALID.

        User Query: "{query}"{tickers_part}

        If invalid, provide a clear, concise error_message explaining why. If valid, leave error_message as empty string.

        Return a JSON object conforming strictly to the following JSON Schema:
        {schema_str}
        """

        try:
            content = stream_chat_completion(
                client=client,
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_schema=ValidationResult,
                print_stream=False
            )
            result = ValidationResult.model_validate_json(content)
            return result.model_dump()
        except Exception as e:
            return {"is_valid": False, "error_message": f"Validation engine error: {e}"}


# ═══════════════════════════════════════════════════════════
#  Query Interpreter (Chain-of-Thought)
# ═══════════════════════════════════════════════════════════
class QueryInterpreter:
    """Converts natural language into a structured TradingStrategy via CoT reasoning."""

    def __init__(self):
        self.validator = QueryValidator()

    def parse(self, query: str, tickers: str | None = None) -> dict:
        print("  🔍  Validating query …")
        validation = self.validator.validate(query, tickers)

        if not validation.get("is_valid"):
            print(f"  ❌  Validation failed: {validation.get('error_message')}")
            return {"status": "error", "message": validation.get("error_message"), "data": None}

        print("  ✅  Validation passed. Parsing strategy …")

        schema_str = _get_schema(TradingStrategyWithConfidence)
        tickers_part = f' with tickers: "{tickers}"' if tickers else ''
        prompt = f"""
        You are an expert quantitative trading architect. Analyze this query: "{query}"{tickers_part}.
        Evaluate if the indicators are real, clear, and possess numeric parameters.
        Assign a linguistic_confidence (clarity of phrasing, score from 0.0 to 1.0) and numerical_completeness (presence of exact values like RSI < 30, score from 0.0 to 1.0).

        Analyze the following user trading query and extract the core rules, ticker symbols, and duration.
        Use Chain of Thought reasoning to explain how you interpret the indicators, ticker symbols, duration,
        and rules before outputting the final structure.

        IMPORTANT:
        - INDEX RECOGNITION (CRITICAL): If the user requests to trade an entire index (e.g., 'nifty 50', 'nifty bank', 'nifty it', 'nifty next 50', 'nifty 200', etc.), do NOT attempt to list the individual stock tickers yourself. Instead, pass the exact index macro name as a string literal in the tickers array. Supported index macros include:
          * "nifty50" (for Nifty 50)
          * "niftynext50" (for Nifty Next 50 / Nifty Junior)
          * "niftybank" (for Nifty Bank / Bank Nifty)
          * "niftyit" (for Nifty IT)
          * "niftyfinancialservices" (for Nifty Financial Services)
          * "niftymidcap50" (for Nifty Midcap 50)
          * "niftymidcap100" (for Nifty Midcap 100)
          * "niftysmallcap100" (for Nifty Smallcap 100)
          * "nifty100" (for Nifty 100)
          * "nifty200" (for Nifty 200)
          * "nifty500" (for Nifty 500)
          The downstream data router will handle historical component resolution dynamically.
        - duration should be a string like "2y", "6mo", "1y", "max" etc. that yfinance accepts as a period. If not specified or implied, default to a reasonable period (e.g., "2y").
        - tickers should be a list of ticker symbols, cleaned and uppercased (e.g., ["RELIANCE", "TCS"] or ["^NSEI"]).
        - entry_logic and exit_logic should be concise indicator-based rules.

        Ensure you output the parsed Entry Logic, Exit Logic, Duration, Tickers, and confidence scores.

        Return a JSON object conforming strictly to the following JSON Schema:
        {schema_str}
        """

        try:
            content = stream_chat_completion(
                client=client,
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_schema=TradingStrategyWithConfidence,
                print_stream=False
            )
            print(f"\n--- [DEBUG] Raw LLM Parse Response ---\n{content}\n---------------------------------------\n")

            strategy = TradingStrategyWithConfidence.model_validate_json(content)
            
            # Phase 1 Gatekeeper Logic
            intent_score = (0.4 * strategy.linguistic_confidence) + (0.6 * strategy.numerical_completeness)
            
            if intent_score < 0.70:
                print(f"  ❌  Phase 1 intent confidence check failed: score = {intent_score:.2f} < 0.70")
                return {
                    "status": "REJECTED", 
                    "message": f"Strategy prompt is too vague or lacks explicit numerical thresholds (score: {intent_score:.2f}). Please clarify your parameters.", 
                    "data": None
                }
                
            result_data = strategy.model_dump()
            result_data["phase_1_score"] = intent_score
            
            return {"status": "success", "message": "Strategy successfully parsed.", "data": result_data}
        except Exception as e:
            return {"status": "error", "message": f"Parsing failed: {e}", "data": None}


# ═══════════════════════════════════════════════════════════
#  DAG Planner
# ═══════════════════════════════════════════════════════════
class DAGPlanner:
    """Builds a Directed Acyclic Graph for execution ordering."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_plan(self, strategy_data: dict) -> nx.DiGraph:
        tickers = strategy_data.get("tickers", [])

        # Root
        self.graph.add_node("Parse_Intent", status="Completed")

        # Data fetch per ticker
        fetch_nodes = []
        for ticker in tickers:
            node = f"Fetch_Data_{ticker}"
            self.graph.add_node(node, status="Pending")
            self.graph.add_edge("Parse_Intent", node)
            fetch_nodes.append(node)

        # Code gen
        self.graph.add_node("Generate_Strategy_Code", status="Pending")
        self.graph.add_edge("Parse_Intent", "Generate_Strategy_Code")

        # Backtest (depends on ALL data and code gen)
        self.graph.add_node("Run_Backtest", status="Pending")
        for fn in fetch_nodes:
            self.graph.add_edge(fn, "Run_Backtest")
        self.graph.add_edge("Generate_Strategy_Code", "Run_Backtest")

        # Risk analysis
        self.graph.add_node("Risk_Analysis", status="Pending")
        self.graph.add_edge("Run_Backtest", "Risk_Analysis")

        # Report
        self.graph.add_node("Generate_Report", status="Pending")
        self.graph.add_edge("Run_Backtest", "Generate_Report")
        self.graph.add_edge("Risk_Analysis", "Generate_Report")

        return self.graph

    def get_execution_order(self) -> list:
        """Returns topologically sorted execution order."""
        return list(nx.topological_sort(self.graph))

    def update_node_status(self, node_name: str, status: str):
        if node_name in self.graph.nodes:
            self.graph.nodes[node_name]["status"] = status


# ═══════════════════════════════════════════════════════════
#  Registration Bus
# ═══════════════════════════════════════════════════════════
class RegistrationBus:
    """Coordinates and manages all agents dynamically."""

    def __init__(self):
        self.agents = {}

    def register(self, name: str, agent_type: str, status: str = "Idle"):
        self.agents[name] = {
            "type": agent_type,
            "status": status,
            "last_heartbeat": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def update_status(self, name: str, new_status: str):
        if name in self.agents:
            self.agents[name]["status"] = new_status
            self.agents[name]["last_heartbeat"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_available(self, agent_type: str) -> str | None:
        for name, info in self.agents.items():
            if info["type"] == agent_type and info["status"] == "Idle":
                return name
        return None

    def summary(self) -> dict:
        return self.agents


# ═══════════════════════════════════════════════════════════
#  Memory Agent (Neo4j or Simulation)
# ═══════════════════════════════════════════════════════════
class MemoryAgent:
    """Logs execution traces. Falls back to simulation mode if Neo4j is unavailable."""

    def __init__(self):
        self.active = False
        self.driver = None
        self._traces = []  # in-memory fallback

        if NEO4J_URI and NEO4J_PASSWORD:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                self.driver.verify_connectivity()
                self.active = True
                print("  🔗  Connected to Neo4j.")
            except Exception:
                pass

        if not self.active:
            print("  ⚠️  Neo4j unavailable — running Memory Agent in simulation mode.")

    def log(self, query: str, entry: str, exit_logic: str, status: str):
        trace = {
            "query": query,
            "entry": entry,
            "exit": exit_logic,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if self.active and self.driver:
            cypher = """
            MERGE (u:UserQuery {text: $query})
            MERGE (s:Strategy {entry: $entry, exit: $exit})
            MERGE (u)-[:TRANSLATED_TO]->(s)
            MERGE (e:Execution {status: $status, timestamp: datetime()})
            MERGE (s)-[:EXECUTED_AS]->(e)
            """
            with self.driver.session() as session:
                session.run(cypher, query=query, entry=entry, exit=exit_logic, status=status)
            print("  💾  Trace saved to Neo4j.")
        else:
            self._traces.append(trace)
            print(f"  💾  [SIM] Trace logged in memory (total: {len(self._traces)})")

    def close(self):
        if self.driver:
            self.driver.close()
