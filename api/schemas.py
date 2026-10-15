"""
api.schemas
-----------
Pydantic V2 schemas for request validation and response serialization.
Includes Field constraints, type safety, and OpenAPI examples.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ── RAG Query Schemas ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="The technical question to query against the documentation corpus.",
        examples=["How do conditional edges work in LangGraph and what do they return?"],
    )


class Citation(BaseModel):
    source: str = Field(..., description="Document file name or web URL.")
    section: str = Field(..., description="Section header or document title.")
    document_name: Optional[str] = Field(None, description="Human readable name of the document.")


class QueryResponse(BaseModel):
    query_id: str = Field(..., description="Unique UUID assigned to this query.")
    question: str = Field(..., description="The original user question.")
    answer: str = Field(..., description="Generated grounded technical answer.")
    citations: List[Citation] = Field(default_factory=list, description="Referenced sources and sections.")
    query_intent: Optional[str] = Field(None, description="Classified intent (conceptual, how-to, api_reference, troubleshooting).")
    rewritten_query: Optional[str] = Field(None, description="Reformulated query if self-correction or expansion occurred.")
    web_search_used: bool = Field(False, description="Whether the hybrid web search fallback (Tavily+Exa) was triggered.")
    is_grounded: Optional[bool] = Field(None, description="Hallucination check result (True if grounded).")
    hallucination_explanation: Optional[str] = Field(None, description="Reasoning from the hallucination audit node.")
    retries_count: int = Field(0, description="Number of query reformulations attempted.")
    latency_seconds: float = Field(..., description="Total execution time in seconds.")


# ── Ingestion & Document Corpus Schemas ───────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str = Field(
        ...,
        description="Public URL of the documentation page to crawl, chunk, and index.",
        examples=["https://fastapi.tiangolo.com/tutorial/bigger-applications/"],
    )
    document_name: Optional[str] = Field(
        None,
        description="Optional friendly name for the document.",
        examples=["FastAPI Bigger Applications Guide"],
    )


class IngestResponse(BaseModel):
    status: str = Field("success", description="Status of the ingestion operation.")
    document_name: str = Field(..., description="Name of the ingested document.")
    source: str = Field(..., description="Source filename or URL.")
    chunks_added: int = Field(..., description="Number of chunks created and added to ChromaDB.")
    total_corpus_chunks: int = Field(..., description="Total chunks currently stored in the vectorstore.")
    message: str = Field(..., description="Human-readable result summary.")


class DocumentItem(BaseModel):
    document_name: str = Field(..., description="Name of the document.")
    source: str = Field(..., description="Original filename or URL.")
    chunk_count: int = Field(..., description="Number of chunks indexed for this document.")
    sample_sections: List[str] = Field(default_factory=list, description="Sample section titles indexed.")


class DocumentListResponse(BaseModel):
    total_documents: int = Field(..., description="Total count of distinct documents in corpus.")
    total_chunks: int = Field(..., description="Total chunks indexed in ChromaDB.")
    documents: List[DocumentItem] = Field(default_factory=list, description="List of indexed documents.")


class DeleteDocumentResponse(BaseModel):
    status: str = Field("success", description="Status of deletion.")
    document_name: str = Field(..., description="Name of the document deleted.")
    chunks_deleted: int = Field(..., description="Number of chunks removed from ChromaDB.")
    remaining_chunks: int = Field(..., description="Total chunks remaining in corpus.")


# ── Feedback Schemas ──────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: str = Field(
        ...,
        description="The query_id returned by POST /query that this feedback evaluates.",
        examples=["c9b1d8e2-45e1-4c17-91a5-82b5e90d1f7a"],
    )
    rating: Literal["thumbs_up", "thumbs_down"] = Field(
        ...,
        description="User rating: thumbs_up (positive) or thumbs_down (negative).",
    )
    comment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional feedback comment or correction.",
        examples=["Accurate answer with clear code example!"],
    )


class FeedbackResponse(BaseModel):
    status: str = Field("success", description="Status of feedback submission.")
    feedback_id: str = Field(..., description="Unique ID for this feedback entry.")
    message: str = Field(..., description="Confirmation message.")


class FeedbackItem(BaseModel):
    feedback_id: str
    query_id: str
    rating: str
    comment: Optional[str] = None
    created_at: str


class FeedbackStatsResponse(BaseModel):
    total_feedback: int = Field(..., description="Total feedback submissions received.")
    thumbs_up: int = Field(..., description="Count of positive feedback.")
    thumbs_down: int = Field(..., description="Count of negative feedback.")
    satisfaction_rate_percent: float = Field(..., description="Percentage of thumbs_up ratings.")
    feedback_entries: List[FeedbackItem] = Field(default_factory=list, description="List of recorded feedback entries.")


# ── Health & System Schemas ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Overall health state.")
    chroma_connected: bool = Field(..., description="ChromaDB vector store connection status.")
    groq_configured: bool = Field(..., description="Groq LLM API key status.")
    total_indexed_chunks: int = Field(..., description="Current count of indexed chunks in ChromaDB.")
    active_model: str = Field(..., description="Groq LLM model name in use.")
    uptime_seconds: float = Field(..., description="Backend service uptime in seconds.")
