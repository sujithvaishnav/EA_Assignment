"""
api.routers.query
-----------------
Endpoint for querying the technical documentation RAG pipeline.
"""

import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, status
from api.schemas import QueryRequest, QueryResponse, Citation
from rag_pipeline.graph import get_rag_graph

router = APIRouter(tags=["RAG Querying"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Technical Question",
)
def submit_query(payload: QueryRequest):
    """
    Submits a technical question to the Self-Corrective RAG pipeline.

    Workflow executed:
    1. **Query Analysis**: Intent classification and query expansion.
    2. **Hybrid Retrieval**: BM25 keyword + ChromaDB dense vectors fused via RRF.
    3. **Document Grading**: LLM verifies relevance of each chunk.
    4. **Self-Correction**: Rewrites query if initial chunks are irrelevant (up to 2 retries).
    5. **Web Search Fallback**: Automatically searches Tavily + Exa if vectorstore has no relevant data.
    6. **Grounded Generation**: Synthesizes answer adhering strictly to context.
    7. **Hallucination Check**: Audits answer against retrieved context.
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    question = payload.question.strip()

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

    try:
        graph = get_rag_graph()
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(exc)}",
        ) from exc

    latency = round(time.time() - start_time, 3)

    # Extract clean citations
    used_docs = final_state.get("filtered_documents", []) or final_state.get("documents", [])
    citations: List[Citation] = []
    seen = set()

    for d in used_docs:
        src = d.metadata.get("source", "unknown")
        sec = d.metadata.get("section", d.metadata.get("title", "General"))
        doc_name = d.metadata.get("document_name", src)
        key = (src, sec)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    source=src,
                    section=sec,
                    document_name=doc_name,
                )
            )

    generation = final_state.get("generation", "No answer could be generated.")

    return QueryResponse(
        query_id=query_id,
        question=question,
        answer=generation,
        citations=citations,
        query_intent=final_state.get("query_intent"),
        rewritten_query=final_state.get("rewritten_query"),
        web_search_used=final_state.get("web_search_used", False),
        is_grounded=final_state.get("is_grounded"),
        hallucination_explanation=final_state.get("hallucination_explanation"),
        retries_count=final_state.get("retry_count", 0),
        latency_seconds=latency,
    )
