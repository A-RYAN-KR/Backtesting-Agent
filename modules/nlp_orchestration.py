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
from datetime import datetime
from pydantic import BaseModel
from google import genai
from google.genai import types
import networkx as nx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GEMINI_API_KEY, GEMINI_MODEL_FAST, GEMINI_MODEL_PRO, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ─── Configure Gemini Client ────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)


# ═══════════════════════════════════════════════════════════
#  Pydantic schemas for structured LLM output
# ═══════════════════════════════════════════════════════════
class ValidationResult(BaseModel):
    is_valid: bool
    error_message: str


class TradingStrategy(BaseModel):
    reasoning: str
    entry_logic: str
    exit_logic: str
    duration: str
    tickers: list[str]


# ═══════════════════════════════════════════════════════════
#  Query Validator
# ═══════════════════════════════════════════════════════════
class QueryValidator:
    """Validates the trading query for missing data, vagueness, or fake indicators."""

    def validate(self, query: str, tickers: str) -> dict:
        # Static hard checks
        if not query or len(query.strip()) < 10:
            return {"is_valid": False, "error_message": "Query is too short or missing."}
        if not tickers or len(tickers.strip()) == 0:
            return {"is_valid": False, "error_message": "No tickers provided. Please specify at least one symbol."}

        prompt = f"""
        You are a strict validation agent for a quantitative trading system.
        Review the following user query and tickers.

        Rules for Validity:
        1. The query MUST contain actionable trading logic (e.g., conditions for buying/selling).
        2. The technical indicators mentioned MUST be real, standard financial indicators (e.g., RSI, MACD, SMA, Bollinger Bands).
        3. If the query uses made-up indicators or is completely vague, it is INVALID.

        User Query: "{query}"
        Tickers: "{tickers}"

        If invalid, provide a clear, concise error_message explaining why. If valid, leave error_message as empty string.
        """

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_FAST,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ValidationResult,
                    temperature=0.0,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            return {"is_valid": False, "error_message": f"Validation engine error: {e}"}


# ═══════════════════════════════════════════════════════════
#  Query Interpreter (Chain-of-Thought)
# ═══════════════════════════════════════════════════════════
class QueryInterpreter:
    """Converts natural language into a structured TradingStrategy via CoT reasoning."""

    def __init__(self):
        self.validator = QueryValidator()

    def parse(self, query: str, tickers: str) -> dict:
        print("  🔍  Validating query …")
        validation = self.validator.validate(query, tickers)

        if not validation.get("is_valid"):
            print(f"  ❌  Validation failed: {validation.get('error_message')}")
            return {"status": "error", "message": validation.get("error_message"), "data": None}

        print("  ✅  Validation passed. Parsing strategy …")

        prompt = f"""
        You are an expert quantitative trading architect.
        Analyze the following user trading query and extract the core rules.
        Use Chain of Thought reasoning to explain how you interpret the indicators
        and rules before outputting the final structure.

        User Query: "{query}"
        Tickers: "{tickers}"

        IMPORTANT:
        - duration should be a string like "2y", "6mo", "1y", "max" etc. that yfinance accepts as a period.
        - tickers should be a list of ticker symbols, cleaned and uppercased.
        - entry_logic and exit_logic should be concise indicator-based rules.

        Ensure you output the parsed Entry Logic, Exit Logic, Duration, and the Tickers.
        """

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_PRO,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TradingStrategy,
                    temperature=0.1,
                ),
            )
            parsed = json.loads(response.text)
            return {"status": "success", "message": "Strategy successfully parsed.", "data": parsed}
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
