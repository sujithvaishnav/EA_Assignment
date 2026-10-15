"""
End-to-End Pipeline Runner (Full LangGraph Workflow)
Invokes the compiled StateGraph:
  analyze_query -> retrieve -> grade_documents -> route_after_grading -> generate (or rewrite_query loop)

Usage:
  python run_pipeline.py
  python run_pipeline.py "How do I define path parameters in FastAPI?"
"""

import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rag_pipeline.graph import get_rag_graph


def run_rag_pipeline(question: str):
    print("\n" + "=" * 70)
    print("  INVOKING COMPILED LANGGRAPH WORKFLOW")
    print(f"  Question: '{question}'")
    print("=" * 70)

    graph = get_rag_graph()

    initial_state = {
        "question": question,
        "query_intent": None,
        "rewritten_query": None,
        "documents": [],
        "filtered_documents": [],
        "document_grades": [],
        "generation": None,
        "retry_count": 0,
        "web_search_needed": False,
        "web_search_used": False,
        "is_grounded": None,
        "hallucination_explanation": None,
        "regen_count": 0,
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("  FINAL ANSWER & CITATIONS")
    print("=" * 70 + "\n")
    # Safe printing for Windows console
    safe_output = final_state["generation"].encode("ascii", errors="replace").decode("ascii")
    print(safe_output)
    print("\n" + "=" * 70)

    return final_state


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How do conditional edges work in LangGraph?"

    run_rag_pipeline(query)
