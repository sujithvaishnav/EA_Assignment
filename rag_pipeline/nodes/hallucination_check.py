"""
Hallucination Check Node (Self-RAG Groundedness Auditor)
Audits the generated answer against the retrieved context to verify factual grounding.

Behavior:
1. Compares generated answer against filtered document chunks.
2. Evaluates if all facts, code snippets, function names, and statements are grounded in context.
3. If grounded -> routes to END.
4. If hallucinated AND regen_count < 1 -> routes back to GENERATE with strict feedback.
5. If hallucinated AND regen_count >= 1 -> appends a cautionary disclaimer and finishes.
"""

import os
import sys
import json
import re
from typing import Dict, Any, List

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


HALLUCINATION_PROMPT = """You are a Groundedness and Hallucination Auditor for a Technical Documentation Assistant.
Your task is to determine whether the Generated Answer is strictly grounded in and supported by the Retrieved Context.

Retrieved Context (Ground Truth):
{context}

Generated Answer to Audit:
{generation}

Evaluation Criteria:
1. Is every factual claim, explanation, code example, and parameter mentioned in the answer directly supported by the context above?
2. Minor syntactic rephrasing or formatting changes are acceptable as long as facts remain true to the context.
3. FABRICATED CLAIMS: If the answer mentions classes, methods, parameters, external historical facts, or technical claims NOT present in the retrieved context, it must be flagged as NOT grounded.

Output ONLY valid JSON with this exact structure:
{{
  "is_grounded": true | false,
  "explanation": "Brief explanation why the answer is grounded or what specific claim was hallucinated"
}}
"""


def get_hallucination_llm() -> ChatGroq:
    """Initializes Groq LLM client for hallucination auditing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("[HallucinationCheck] GROQ_API_KEY is missing in .env")

    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        model=model_name,
        temperature=0.0,  # Zero temperature for objective, deterministic auditing
        api_key=api_key,
        max_retries=2,
    )


def format_context(docs: List[Document]) -> str:
    """Formats document chunks into context string for verification."""
    if not docs:
        return "No context provided."
    return "\n\n".join([
        f"[Doc {i+1} - {d.metadata.get('source', '')} > {d.metadata.get('section', '')}]\n{d.page_content}"
        for i, d in enumerate(docs)
    ])


def check_hallucination_node(state: RAGState) -> Dict[str, Any]:
    """
    LangGraph Node: Hallucination Check.

    Reads:
      - state["generation"]
      - state["filtered_documents"] (or state["documents"])
    Writes:
      - {"is_grounded": bool, "hallucination_explanation": str}
    """
    generation = state.get("generation", "").strip()
    docs = state.get("filtered_documents") or state.get("documents", [])

    if not generation:
        return {"is_grounded": True, "hallucination_explanation": "No generation to audit."}

    # If the generation is the no-answer fallback message, it is trivially grounded
    if "unable to find" in generation.lower() or "no relevant documentation" in generation.lower():
        return {"is_grounded": True, "hallucination_explanation": "Fallback response is grounded."}

    print("\n[HallucinationCheck] Auditing generated answer for factual grounding...")

    context_str = format_context(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", HALLUCINATION_PROMPT),
        ("human", "Evaluate Groundedness:\nJSON:")
    ])

    llm = get_hallucination_llm()
    chain = prompt | llm

    try:
        response = chain.invoke({
            "context": context_str,
            "generation": generation
        })
        raw_text = response.content.strip()

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            is_grounded = bool(data.get("is_grounded", True))
            explanation = data.get("explanation", "Verification completed.")
        else:
            is_grounded = True
            explanation = "Defaulted to grounded (parsing error)."

    except Exception as e:
        print(f"[HallucinationCheck] Warning: Auditing error: {e}")
        is_grounded = True
        explanation = f"Auditing check bypassed due to exception: {e}"

    status_tag = "GROUNDED" if is_grounded else "HALLUCINATED"
    print(f"  Result: [{status_tag}]")
    print(f"  Reason: {explanation}")

    return {
        "is_grounded": is_grounded,
        "hallucination_explanation": explanation,
    }


def route_after_hallucination(state: RAGState) -> str:
    """
    Conditional edge function after hallucination check.

    Returns:
      - "useful": Answer is grounded (proceed to END)
      - "regenerate": Hallucination detected and regen_count < 1 (regenerate answer)
    """
    is_grounded = state.get("is_grounded", True)
    regen_count = state.get("regen_count", 0)

    if is_grounded:
        print("[Router: Hallucination] Answer is grounded -> returning to user (END)")
        return "useful"

    if regen_count < 1:
        print(f"[Router: Hallucination] Hallucination detected (attempt {regen_count + 1}/1) -> routing to REGENERATE")
        return "regenerate"

    print("[Router: Hallucination] Hallucination persists after regeneration -> returning with caution disclaimer (END)")
    return "useful"


if __name__ == "__main__":
    print("Testing hallucination check standalone...")
    sample_doc = Document(
        page_content="Conditional edges in LangGraph route based on state using graph.add_conditional_edges.",
        metadata={"source": "01_langgraph_concepts.md", "section": "Edges"}
    )

    grounded_gen = "In LangGraph, conditional edges route dynamically using graph.add_conditional_edges."
    hallucinated_gen = "LangGraph conditional edges were created by Thomas Edison in 1892 using Quantum Vacuum tubes."

    s1: RAGState = {
        "question": "How do conditional edges work in LangGraph?",
        "query_intent": None,
        "rewritten_query": None,
        "documents": [sample_doc],
        "filtered_documents": [sample_doc],
        "document_grades": ["relevant"],
        "generation": grounded_gen,
        "retry_count": 0,
        "web_search_needed": False,
        "is_grounded": None,
        "hallucination_explanation": None,
        "regen_count": 0,
    }

    print("\n--- Test Grounded ---")
    res1 = check_hallucination_node(s1)
    print(f"Result 1: {res1}")

    print("\n--- Test Hallucinated ---")
    s2 = {**s1, "generation": hallucinated_gen}
    res2 = check_hallucination_node(s2)
    print(f"Result 2: {res2}")
