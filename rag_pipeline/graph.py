import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langgraph.graph import StateGraph, START, END
from rag_pipeline.state import RAGState
from rag_pipeline.nodes import (
    analyze_query_node,
    retrieve_node,
    grade_documents_node,
    route_after_grading,
    generate_node,
    no_answer_fallback_node,
    rewrite_query_node,
    check_hallucination_node,
    route_after_hallucination,
    web_search_node,
)


def create_rag_graph():
    """
    Builds and compiles the complete Self-Corrective RAG workflow.
    """
    workflow = StateGraph(RAGState)

    # 1. Register Nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_hallucination", check_hallucination_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("no_answer", no_answer_fallback_node)

    # 2. Add Fixed Edges
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("generate", "check_hallucination")
    workflow.add_edge("no_answer", END)

    # Self-Correction Loop: re-enters retrieval with reformulated query
    workflow.add_edge("rewrite_query", "retrieve")

    # Web Search Fallback Loop: web search results flow into document grader
    workflow.add_edge("web_search", "grade_documents")

    # 3. Add Conditional Routing Edge after Document Grading
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
            "no_answer": "no_answer",
        },
    )

    # 4. Add Conditional Routing Edge after Hallucination Check
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {
            "useful": END,
            "regenerate": "generate",
        },
    )

    return workflow.compile()


# Cached compiled graph instance
_COMPILED_GRAPH = None


def get_rag_graph():
    """Returns the cached, compiled LangGraph runnable."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = create_rag_graph()
    return _COMPILED_GRAPH


if __name__ == "__main__":
    print("Initializing compiled LangGraph...")
    graph = get_rag_graph()
    print("LangGraph compiled successfully!")

    # Test run
    test_question = "How do conditional edges work in LangGraph?"
    initial_state = {
        "question": test_question,
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
    print("\n--- Final Graph Output ---")
    safe_gen = final_state["generation"].encode("ascii", errors="replace").decode("ascii")
    print(safe_gen)
