"""
api.main
--------
FastAPI backend service for the Self-Corrective RAG Technical Documentation Assistant.
Features:
- RAG Querying via LangGraph workflow (POST /query)
- Document Ingestion from Files and URLs (POST /ingest, POST /ingest/url, DELETE /documents/{name})
- Corpus Inspection (GET /documents)
- User Feedback & Monitoring (POST /feedback, GET /feedback)
- System Health Diagnostics (GET /health, GET /graph/topology)
"""

import os
import sys
from contextlib import asynccontextmanager

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routers import health, query, documents, feedback
from rag_pipeline.nodes.retrieve import get_retriever
from rag_pipeline.graph import get_rag_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: Pre-warms the ChromaDB vectorstore,
    BM25 keyword index, and compiled LangGraph workflow on startup.
    """
    print("\n[FastAPI] Pre-warming HybridRetriever and LangGraph workflow...")
    try:
        retriever = get_retriever()
        print(f"[FastAPI] Vectorstore ready with {len(retriever.corpus_docs)} documents.")
        get_rag_graph()
        print("[FastAPI] LangGraph compiled and ready for inference.")
    except Exception as exc:
        print(f"[FastAPI] Warning during startup pre-warming: {exc}")
    yield
    print("[FastAPI] Shutting down backend service.")


app = FastAPI(
    title="Self-Corrective RAG Backend API",
    description="""
### Technical Documentation Assistant Backend
Powered by **LangGraph**, **Hybrid Search (Dense + BM25 RRF)**, **Groq LLM Grader/Generator**,
**Hybrid Web Search Fallback (Tavily + Exa RRF)**, and **Self-RAG Groundedness Audit**.

#### Features:
* **RAG Querying**: Submit questions, run iterative self-correction, fall back to web search if needed.
* **Document Ingestion**: Upload `.md`, `.txt`, `.pdf` or crawl public documentation URLs.
* **Corpus Management**: Inspect indexed chunks and sections, delete obsolete documents.
* **User Feedback**: Submit thumbs up/down ratings, track aggregate quality metrics.
* **Health & Diagnostics**: Real-time vector store check and LangGraph Mermaid topology.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local Streamlit, Gradio, or frontend clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Router Registration ────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(feedback.router)


@app.get("/", include_in_schema=False)
def root():
    """Redirects root URL to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
