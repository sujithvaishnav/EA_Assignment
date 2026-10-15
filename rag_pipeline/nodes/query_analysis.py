"""
Query Analysis & Rewriting Node (Node 1 in LangGraph Workflow)
Handles:
1. analyze_query_node: Initial pass — classifies intent & expands query with technical terms.
2. rewrite_query_node: Self-corrective retry pass — reformulates query when initial retrieval yielded no relevant chunks.
"""

import os
import sys
import json
import re
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rag_pipeline.state import RAGState


# ── Pydantic Schema for Structured Analysis ───────────────────────────────────

class QueryAnalysisSchema(BaseModel):
    intent: Literal["conceptual", "how-to", "troubleshooting", "api_reference"] = Field(
        description="Classify the question: 'conceptual' (architecture/theory), 'how-to' (implementation/code tutorial), 'troubleshooting' (bugs/errors/gotchas), or 'api_reference' (signatures/params/syntax)."
    )
    expanded_query: str = Field(
        description="Expanded, keyword-rich query optimized for BM25 and vector retrieval. Adds specific framework names, class names, and technical keywords."
    )


def get_query_llm() -> ChatGroq:
    """Initializes Groq LLM client for query analysis and rewriting."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("[QueryAnalysis] GROQ_API_KEY is missing in .env")

    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        model=model_name,
        temperature=0.0,
        api_key=api_key,
        max_retries=2,
    )


# ── System Prompts ────────────────────────────────────────────────────────────

ANALYZE_PROMPT = """You are a Query Analysis assistant for a Technical Documentation RAG system.
The system indexes documentation for: LangGraph, FastAPI, Pydantic V2, ChromaDB, and LangChain RAG.

Given a user's question:
1. Classify its intent into ONE of:
   - "conceptual": high-level architecture, design patterns, definitions
   - "how-to": practical implementation, code tutorials, building a feature
   - "troubleshooting": debugging, common errors, cycle limits, gotchas
   - "api_reference": exact method names, parameters, decorators, configuration syntax

2. Formulate an "expanded_query" optimized for search:
   - Expand casual words with official library terms (e.g. "save history" -> "persist state checkpointer MemorySaver")
   - Include exact library/class/method names where known
   - Keep it concise, focused, and search-ready

Output ONLY valid JSON with this exact structure:
{{
  "intent": "conceptual | how-to | troubleshooting | api_reference",
  "expanded_query": "optimized search query string"
}}
"""

REWRITE_PROMPT = """You are a Query Reformulation assistant for a Technical Documentation RAG system.
A previous search for the user's question FAILED to find any relevant document chunks.

Your task is to rephrase the question from an alternative technical perspective:
1. Strip conversational filler words.
2. Break complex questions into foundational technical search terms.
3. Use synonyms, alternate library terms, or broader related concepts.
4. Focus on the core Python libraries: LangGraph, FastAPI, Pydantic V2, ChromaDB, LangChain.

Output ONLY the reformulated query string. Do NOT include quotes, bullet points, or commentary.
"""


# ── Node 1a: Initial Query Analysis & Expansion ──────────────────────────────

def analyze_query_node(state: RAGState) -> Dict[str, Any]:
    """
    LangGraph Node 1: Query Analysis.
    Classifies intent and expands the query with technical keywords.

    Reads:
      state["question"]
    Writes:
      {"query_intent": str, "rewritten_query": str}
    """
    question = state.get("question", "").strip()
    if not question:
        raise ValueError("[AnalyzeQueryNode] Error: No question provided in state.")

    print(f"\n[AnalyzeQueryNode] Analyzing question: '{question}'")

    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYZE_PROMPT),
        ("human", "Question: {question}\n\nJSON:")
    ])

    llm = get_query_llm()
    chain = prompt | llm

    try:
        response = chain.invoke({"question": question})
        raw_text = response.content.strip()

        # Extract JSON from potential code fences
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            intent = data.get("intent", "how-to")
            if intent not in ["conceptual", "how-to", "troubleshooting", "api_reference"]:
                intent = "how-to"
            expanded_query = data.get("expanded_query", question).strip()
        else:
            intent = "how-to"
            expanded_query = question

    except Exception as e:
        print(f"[AnalyzeQueryNode] Warning: Analysis fallback due to error: {e}")
        intent = "how-to"
        expanded_query = question

    print(f"  Intent:         {intent}")
    print(f"  Expanded Query: '{expanded_query}'")

    return {
        "query_intent": intent,
        "rewritten_query": expanded_query,
    }


# ── Node 1b: Query Rewriting on Retry ─────────────────────────────────────────

def rewrite_query_node(state: RAGState) -> Dict[str, Any]:
    """
    LangGraph Self-Correction Node: Query Rewriter.
    Triggered when all retrieved chunks are graded as irrelevant.

    Reads:
      state["question"], state["rewritten_query"], state["retry_count"]
    Writes:
      {"rewritten_query": str, "retry_count": int, "documents": [], "filtered_documents": []}
    """
    question = state.get("question", "")
    current_retry = state.get("retry_count", 0)
    prev_query = state.get("rewritten_query") or question

    new_retry_count = current_retry + 1
    print(f"\n[RewriteQueryNode] Self-Correction triggered (Attempt {new_retry_count}/2)")
    print(f"  Original: '{question}'")
    print(f"  Previous search that failed: '{prev_query}'")

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_PROMPT),
        ("human", "Original Question: {question}\nFailed Query: {prev_query}\n\nReformulated Query:")
    ])

    llm = get_query_llm()
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": question,
            "prev_query": prev_query
        })
        reformulated = response.content.strip().strip('"').strip("'")
        if not reformulated:
            reformulated = f"{question} documentation implementation"
    except Exception as e:
        print(f"[RewriteQueryNode] Warning: Rewrite fallback: {e}")
        reformulated = f"{question} guide"

    print(f"  Reformulated: '{reformulated}'")

    return {
        "rewritten_query": reformulated,
        "retry_count": new_retry_count,
        # Clear previous failed documents so new retrieval can populate fresh ones
        "documents": [],
        "filtered_documents": [],
    }


if __name__ == "__main__":
    print("Testing query_analysis standalone...")
    sample_state: RAGState = {
        "question": "How to save graph state across runs?",
        "query_intent": None,
        "rewritten_query": None,
        "documents": [],
        "filtered_documents": [],
        "document_grades": [],
        "generation": None,
        "retry_count": 0,
        "web_search_needed": False,
    }

    res = analyze_query_node(sample_state)
    sample_state.update(res)

    print("\nSimulating retry rewrite...")
    retry_res = rewrite_query_node(sample_state)
    print(f"Result: {retry_res}")
