"""
RAG Pipeline — Shared State Schema
All LangGraph nodes read from and write to this shared state.
"""

from typing import TypedDict, List, Optional, Annotated
import operator
from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    Central shared state that flows through every node in the LangGraph pipeline.

    Fields:
        question         : The raw user question as submitted.
        rewritten_query  : Optionally rewritten/expanded query (after query analysis node).
        documents        : Raw list of Document chunks retrieved from ChromaDB.
        filtered_documents : Documents that passed the relevance grader.
        document_grades  : Parallel list of grade strings per retrieved doc ("relevant"/"irrelevant").
        generation       : Final generated answer string.
        retry_count      : How many times retrieval has been retried (max 2 before fallback).
        web_search_needed: Flag set by the grader router if all docs are irrelevant.
    """
    question: str
    query_intent: Optional[str]
    rewritten_query: Optional[str]
    documents: List[Document]
    filtered_documents: List[Document]
    document_grades: Annotated[List[str], operator.add]
    generation: Optional[str]
    retry_count: int
    web_search_needed: bool
    web_search_used: Optional[bool]
    is_grounded: Optional[bool]
    hallucination_explanation: Optional[str]
    regen_count: int
