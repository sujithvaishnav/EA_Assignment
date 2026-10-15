"""
Node: Document Grader
─────────────────────────────────────────────────────
Responsibility:
    For each retrieved document chunk, call the Groq LLM to assess
    whether the chunk is RELEVANT or IRRELEVANT to the user's question.

Design:
    - Uses structured output (Pydantic) → no string parsing, no hallucination risk
    - Grades documents one-by-one → fine-grained filtering
    - Attaches grade results to state for the conditional routing edge

LangGraph Position:
    retrieve_node → [grade_documents_node] → route_after_grading (conditional edge)
"""

import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from rag_pipeline.state import RAGState

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Pydantic Schema for Structured Output
#     The LLM is forced to return ONLY this schema — no parsing logic needed.
# ─────────────────────────────────────────────────────────────────────────────

class DocumentGrade(BaseModel):
    """Structured relevance grade returned by the grader LLM."""

    score: Literal["relevant", "irrelevant"] = Field(
        description=(
            "Assess whether the document chunk contains information that "
            "directly helps answer the user's question. "
            "'relevant' = the chunk has useful information; "
            "'irrelevant' = the chunk does not help."
        )
    )
    reasoning: str = Field(
        description="One-sentence explanation for the grade (for debugging/logging)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Grader Prompt
#     Kept minimal and directive — models grade faster with tighter prompts.
# ─────────────────────────────────────────────────────────────────────────────

GRADER_SYSTEM_PROMPT = """You are an expert relevance grader for a RAG (Retrieval-Augmented Generation) system.

Your task is to assess whether a retrieved document chunk is useful for answering the user's question.

Grading rules:
- Grade as "relevant" if the chunk contains ANY factual information that helps answer the question,
  even partially or as background context.
- Grade as "irrelevant" ONLY if the chunk has no meaningful connection to the question topic.
- Do NOT require the chunk to fully answer the question — partial relevance counts.

You must respond with the structured format only."""

GRADER_HUMAN_TEMPLATE = """User Question:
{question}

Retrieved Document Chunk:
Source: {source} | Section: {section}
---
{document_content}
---

Grade this chunk."""

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", GRADER_SYSTEM_PROMPT),
    ("human", GRADER_HUMAN_TEMPLATE),
])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Grader LLM Chain
#     ChatGroq with structured output (.with_structured_output) is key:
#     it uses tool/function calling to enforce the Pydantic schema.
# ─────────────────────────────────────────────────────────────────────────────

def build_grader_chain():
    """Builds the LLM grader chain. Call once at startup, reuse across requests."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Please add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    # Use model name from .env (GROQ_MODEL) or default to qwen/qwen3.8-27b
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    llm = ChatGroq(
        model=model_name,
        temperature=0,  # Deterministic grading — no randomness
        api_key=groq_api_key,
    )
    print(f"[GradeDocuments] Using Groq model: {model_name}")

    # .with_structured_output() uses Groq's tool-calling to enforce the schema
    structured_llm = llm.with_structured_output(DocumentGrade)
    return grader_prompt | structured_llm


# Build the chain once (module-level singleton)
_grader_chain = None


def get_grader_chain():
    """Returns cached grader chain, building it on first call."""
    global _grader_chain
    if _grader_chain is None:
        _grader_chain = build_grader_chain()
    return _grader_chain


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LangGraph Node Function
#     This is the function registered as a node in StateGraph.
# ─────────────────────────────────────────────────────────────────────────────

def grade_documents_node(state: RAGState) -> dict:
    """
    LangGraph node: grades each retrieved document chunk for relevance.

    Reads from state:
        - state["question"]  : the user's original question
        - state["documents"] : list of Document chunks from the retriever

    Writes to state:
        - filtered_documents : only the chunks graded as "relevant"
        - document_grades    : list of "relevant"/"irrelevant" per chunk
    """
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        print("[GradeDocuments] No documents to grade. Flagging for web search.")
        return {
            "filtered_documents": [],
            "document_grades": [],
            "web_search_needed": True,
        }

    grader = get_grader_chain()
    filtered_docs = []
    grades = []

    print(f"\n[GradeDocuments] Grading {len(documents)} retrieved chunks...")
    print(f"  Question: {question!r}\n")

    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "unknown")
        content = doc.page_content

        try:
            result: DocumentGrade = grader.invoke({
                "question": question,
                "source": source,
                "section": section,
                "document_content": content,
            })

            grade = result.score
            reasoning = result.reasoning

            print(f"  Chunk [{i+1}/{len(documents)}] -> {grade.upper()}")
            print(f"    Source : {source} | Section: {section}")
            print(f"    Reason : {reasoning}\n")

            grades.append(grade)
            if grade == "relevant":
                filtered_docs.append(doc)

        except Exception as e:
            # Re-raise API/config errors immediately -- do not silently pass
            # A fail-open default would mask model name errors, auth issues, etc.
            print(f"  Chunk [{i+1}] -> GRADING ERROR: {e}")
            raise RuntimeError(
                f"Document grading failed for chunk {i+1}.\n"
                f"Original error: {e}\n"
                f"Hint: Check GROQ_API_KEY and GROQ_MODEL in your .env file."
            ) from e

    # Determine if we need web search as fallback
    relevant_count = grades.count("relevant")
    web_search_needed = relevant_count == 0

    print(f"[GradeDocuments] Results: {relevant_count}/{len(documents)} chunks relevant.")
    if web_search_needed:
        print("[GradeDocuments] WARNING: ALL chunks irrelevant -> flagging for query rewrite / web search.")

    return {
        "filtered_documents": filtered_docs,
        "document_grades": grades,
        "web_search_needed": web_search_needed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Conditional Edge Router
#     This function is registered with graph.add_conditional_edges()
#     It reads the state AFTER grade_documents_node and decides next node.
# ─────────────────────────────────────────────────────────────────────────────

def route_after_grading(state: RAGState) -> str:
    """
    Conditional edge function called after grade_documents_node.

    Returns:
        "generate"       -> if at least one relevant document exists
        "rewrite_query"  -> if all docs are irrelevant AND retry_count < 2 (before web search)
        "web_search"     -> if all docs are irrelevant AND retry_count >= 2 AND web search not yet tried
        "no_answer"      -> if all docs are irrelevant AND web search was already attempted
    """
    web_search_needed = state.get("web_search_needed", False)
    web_search_used = state.get("web_search_used", False)
    retry_count = state.get("retry_count", 0)
    filtered = state.get("filtered_documents", [])

    if filtered:
        print(f"[Router] {len(filtered)} relevant docs -> routing to GENERATE")
        return "generate"

    # If web search was already attempted and still yielded no relevant docs -> stop
    if web_search_used:
        print("[Router] No relevant docs found even after web search -> routing to NO_ANSWER")
        return "no_answer"

    # Vector store retrieval retry loop
    if retry_count < 2:
        print(f"[Router] No relevant docs, retry {retry_count + 1}/2 -> routing to REWRITE_QUERY")
        return "rewrite_query"

    # Vector store and query rewriting exhausted -> fallback to hybrid web search
    print("[Router] No relevant docs after max retries -> routing to WEB_SEARCH")
    return "web_search"
