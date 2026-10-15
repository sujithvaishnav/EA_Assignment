"""
Generation Node (Node 4 in LangGraph Workflow)
Generates grounded, citation-backed technical answers from filtered relevant documents.

Requirements addressed:
1. Grounded Answer: Uses filtered documents as strict ground truth to prevent hallucination.
2. Citations: Explicitly cites source files and section headers in the final answer.
3. Code Preservation: Preserves syntax, types, and runnable code snippets from the documentation.
4. Fallback Handling: Gracefully indicates when context is insufficient.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rag_pipeline.state import RAGState


# ── System Prompt for Technical Generation ───────────────────────────────────

GENERATION_SYSTEM_PROMPT = """You are an expert Technical Documentation AI Assistant.
Your task is to answer technical questions accurately and concisely using ONLY the provided documentation context.

RULES:
1. GROUNDEDNESS: Rely strictly on the context provided. Do NOT fabricate functions, parameters, or explanations not present in the documentation.
2. CODE EXAMPLES: When relevant, include clean, syntax-highlighted code blocks with proper language tags (e.g., ```python).
3. CITATIONS: You MUST cite your sources. At the end of your answer, include a "### Sources & Citations" section listing every source file and section used. Where appropriate, also reference sources inline (e.g., [Source: file_name -> section]).
4. INSUFFICIENT CONTEXT: If the provided documents do not contain enough information to completely answer the question, clearly state what is known from the context and what information is missing. Do not guess.
5. TONE: Professional, developer-friendly, clear, and direct.

Context Documents:
{context}
"""

GENERATION_USER_PROMPT = """Question: {question}

Please provide a comprehensive, accurate answer with citations based on the context above:"""


def format_context_documents(documents: List[Document]) -> str:
    """Formats retrieved document chunks with clear section headers and chunk IDs."""
    if not documents:
        return "No relevant documentation available."

    formatted_chunks = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "General")
        chunk_id = doc.metadata.get("chunk_id", f"c_{idx}")
        content = doc.page_content.strip()

        chunk_header = f"--- Document [{idx}] | Source: {source} | Section: {section} | Chunk ID: {chunk_id} ---"
        formatted_chunks.append(f"{chunk_header}\n{content}\n")

    return "\n".join(formatted_chunks)


def extract_unique_citations(documents: List[Document]) -> List[Dict[str, str]]:
    """Extracts unique source files and sections from the documents used for generation."""
    seen = set()
    citations = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "General")
        key = (source, section)
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": source,
                "section": section,
                "document_name": doc.metadata.get("document_name", source)
            })
    return citations


def get_generator_llm() -> ChatGroq:
    """Initializes the Groq LLM client for generation."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or "your_groq_api_key" in groq_api_key:
        raise ValueError(
            "[GenerateNode] GROQ_API_KEY is missing or invalid in .env."
        )

    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        model=model_name,
        temperature=0.1,  # Low temperature for factual consistency and precision
        api_key=groq_api_key,
        max_retries=2,
    )


def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    LangGraph Node 4: Answer Generation.

    Reads:
      - state["question"] (or state["rewritten_query"])
      - state["filtered_documents"] (preferred) or state["documents"]
    Writes:
      - {"generation": str}
    """
    question = state.get("question")
    if not question:
        raise ValueError("[GenerateNode] Error: No question provided in state.")

    # Use filtered documents from grader; fall back to raw documents if not graded
    docs_to_use = state.get("filtered_documents")
    if docs_to_use is None or len(docs_to_use) == 0:
        docs_to_use = state.get("documents", [])

    if not docs_to_use:
        print("[GenerateNode] Warning: No documents available for generation.")
        fallback_msg = (
            "I could not find sufficient relevant information in the technical documentation "
            f"to answer the question: '{question}'. Please try rephrasing or asking about "
            "LangGraph, FastAPI, Pydantic V2, ChromaDB, or LangChain RAG."
        )
        return {"generation": fallback_msg}

    regen_count = state.get("regen_count", 0)
    is_grounded = state.get("is_grounded")
    hallucination_feedback = state.get("hallucination_explanation")

    system_prompt_to_use = GENERATION_SYSTEM_PROMPT
    if is_grounded is False and hallucination_feedback:
        print(f"\n[GenerateNode] REGENERATING answer (Attempt {regen_count + 1}/1) after hallucination audit:")
        print(f"  Audit feedback: {hallucination_feedback}")
        regen_count += 1
        system_prompt_to_use += f"\nCRITICAL WARNING: Your previous answer contained hallucinations:\n'{hallucination_feedback}'\nYou MUST stick strictly to the context and refrain from making any unsupported claims."
    else:
        print(f"\n[GenerateNode] Generating grounded answer using {len(docs_to_use)} relevant chunks...")

    context_str = format_context_documents(docs_to_use)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_to_use),
        ("human", GENERATION_USER_PROMPT),
    ])

    llm = get_generator_llm()
    chain = prompt | llm

    response = chain.invoke({
        "context": context_str,
        "question": question,
    })

    answer_text = response.content.strip()
    citations = extract_unique_citations(docs_to_use)

    print(f"[GenerateNode] Generation complete ({len(answer_text)} chars, {len(citations)} citations).")

    return {
        "generation": answer_text,
        "regen_count": regen_count,
        "is_grounded": None,  # Reset so hallucination check re-audits the new answer
    }


def no_answer_fallback_node(state: RAGState) -> Dict[str, Any]:
    """
    Fallback node triggered when all retrieved documents fail grading
    and maximum retries have been reached.
    """
    question = state.get("question", "")
    retry_count = state.get("retry_count", 0)

    print(f"[NoAnswerNode] Fallback triggered after {retry_count} retries.")
    msg = (
        f"I am unable to find relevant information in the technical documentation to answer: '{question}'.\n\n"
        "The system searched the corpus and attempted query reformulations, but no matching documentation "
        "chunks met the relevance threshold. Please consult the official documentation or refine your search terms."
    )
    return {"generation": msg}


if __name__ == "__main__":
    # Standalone smoke test
    print("Testing generate_node standalone...")
    from rag_pipeline.nodes.retrieve import get_retriever

    retriever = get_retriever()
    q = "How do conditional edges work in LangGraph?"
    sample_docs = retriever.retrieve(q, top_k=2)

    mock_state: RAGState = {
        "question": q,
        "rewritten_query": None,
        "documents": sample_docs,
        "filtered_documents": sample_docs,
        "document_grades": ["relevant", "relevant"],
        "generation": None,
        "retry_count": 0,
        "web_search_needed": False,
    }

    result = generate_node(mock_state)
    print("\n--- Generated Answer ---")
    print(result["generation"])
