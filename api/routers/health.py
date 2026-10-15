"""
api.routers.health
------------------
Health check and system diagnostics endpoints.
"""

import os
import time
from fastapi import APIRouter
from api.schemas import HealthResponse
from rag_pipeline.nodes.retrieve import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from api.services.ingestion_service import get_vectorstore

router = APIRouter(tags=["Health & System"])
_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, summary="System Health & Readiness")
def health_check():
    """
    Returns health status of the RAG backend, verifying:
    - ChromaDB vector store connectivity and chunk counts
    - Groq LLM API configuration
    - Service uptime
    """
    chroma_ok = False
    total_chunks = 0
    try:
        vs = get_vectorstore()
        ids = vs.get().get("ids", [])
        total_chunks = len(ids)
        chroma_ok = True
    except Exception:
        chroma_ok = False

    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_ok = bool(groq_key and not groq_key.startswith("your_"))
    active_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    uptime = round(time.time() - _START_TIME, 2)

    return HealthResponse(
        status="healthy" if (chroma_ok and groq_ok) else "degraded",
        chroma_connected=chroma_ok,
        groq_configured=groq_ok,
        total_indexed_chunks=total_chunks,
        active_model=active_model,
        uptime_seconds=uptime,
    )


@router.get("/graph/topology", summary="LangGraph Workflow Topology")
def get_graph_topology():
    """
    Returns the visual Mermaid architecture diagram and node sequence
    of the compiled Self-Corrective LangGraph state machine.
    """
    mermaid_diagram = """
    graph TD
        START((Start)) --> analyze_query[Query Analysis Node]
        analyze_query --> retrieve[Hybrid Retrieval: BM25 + Dense + RRF]
        retrieve --> grade_documents[Document Grader Node]
        
        grade_documents -->|Relevant Docs Found| generate[Answer Generation Node]
        grade_documents -->|Irrelevant Chunks: Retry < 2| rewrite_query[Query Rewriter Node]
        rewrite_query --> retrieve
        
        grade_documents -->|Irrelevant Chunks: Max Retries| web_search[Hybrid Web Search: Tavily + Exa]
        web_search --> grade_documents
        
        grade_documents -->|Web Search Exhausted| no_answer[No Answer Fallback]
        no_answer --> END((End))
        
        generate --> check_hallucination[Hallucination Audit Node]
        check_hallucination -->|Grounded| END((End))
        check_hallucination -->|Hallucinated: Attempt 1| generate
    """
    return {
        "framework": "LangGraph",
        "nodes": [
            "analyze_query",
            "retrieve",
            "grade_documents",
            "rewrite_query",
            "web_search",
            "generate",
            "check_hallucination",
            "no_answer",
        ],
        "mermaid": mermaid_diagram.strip(),
    }
