"""
Retrieval Quality Evaluation Script
Evaluates retrieval performance across test queries against the persistent ChromaDB collection.
Metrics:
- Hit Rate @ 1, Hit Rate @ 3, Hit Rate @ 5
- Mean Reciprocal Rank (MRR)
- Comparison: Standard Cosine Similarity vs MMR vs Hybrid (BM25 + Dense + RRF)
"""

import os
import sys
import json
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rag_pipeline.nodes.retrieve import get_retriever, HybridRetriever


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "technical_documentation"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 10 Ground-Truth Test Cases spanning the 5 technical documentation topics
TEST_BENCHMARK = [
    {
        "query": "How do conditional edges work in LangGraph?",
        "expected_doc": "01_langgraph_concepts.md",
        "expected_section_keyword": "Edges",
    },
    {
        "query": "How to persist state across graph runs using checkpointers?",
        "expected_doc": "01_langgraph_concepts.md",
        "expected_section_keyword": "Checkpointer",
    },
    {
        "query": "How do I define path parameters and query parameters in FastAPI?",
        "expected_doc": "02_fastapi_tutorial.md",
        "expected_section_keyword": "Parameters",
    },
    {
        "query": "How to use dependency injection with Depends in FastAPI?",
        "expected_doc": "02_fastapi_tutorial.md",
        "expected_section_keyword": "Depends",
    },
    {
        "query": "How do field_validator and model_validator work in Pydantic V2?",
        "expected_doc": "03_pydantic_v2_guide.md",
        "expected_section_keyword": "Validation",
    },
    {
        "query": "How to serialize models with model_dump and model_dump_json?",
        "expected_doc": "03_pydantic_v2_guide.md",
        "expected_section_keyword": "Serialization",
    },
    {
        "query": "What is the difference between RecursiveCharacterTextSplitter and MarkdownHeaderTextSplitter?",
        "expected_doc": "04_langchain_rag.md",
        "expected_section_keyword": "Splitter",
    },
    {
        "query": "How does Maximal Marginal Relevance (MMR) balance relevance and diversity?",
        "expected_doc": "04_langchain_rag.md",
        "expected_section_keyword": "MMR",
    },
    {
        "query": "How to create a persistent ChromaDB client on disk?",
        "expected_doc": "05_chromadb_guide.md",
        "expected_section_keyword": "Persistent",
    },
    {
        "query": "What distance metrics are supported in ChromaDB collections?",
        "expected_doc": "05_chromadb_guide.md",
        "expected_section_keyword": "Distance",
    },
]


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def evaluate_retrieval(vectorstore: Chroma, hybrid_retriever: HybridRetriever, search_mode: str = "similarity", k: int = 5) -> Dict[str, Any]:
    print(f"\n--- Running Evaluation: Mode = '{search_mode.upper()}' (Top-k = {k}) ---")
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    query_details = []

    for idx, test_case in enumerate(TEST_BENCHMARK, 1):
        query = test_case["query"]
        expected_doc = test_case["expected_doc"]
        expected_sec = test_case["expected_section_keyword"].lower()

        if search_mode == "similarity":
            results = vectorstore.similarity_search_with_score(query, k=k)
            retrieved_docs = [doc for doc, score in results]
        elif search_mode == "mmr":
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": 15, "lambda_mult": 0.7}
            )
            retrieved_docs = retriever.invoke(query)
        elif search_mode == "hybrid":
            retrieved_docs = hybrid_retriever.retrieve(query=query, top_k=k)
        else:
            raise ValueError(f"Unknown mode: {search_mode}")

        # Check ranks of correct document matches
        match_rank = None
        for rank, doc in enumerate(retrieved_docs, start=1):
            doc_source = doc.metadata.get("source", "")
            doc_section = doc.metadata.get("section", "").lower()

            if doc_source == expected_doc and expected_sec in doc_section:
                match_rank = rank
                break
            elif doc_source == expected_doc and match_rank is None:
                # Secondary match if exact section keyword missed but correct doc hit
                match_rank = rank

        if match_rank == 1:
            hits_at_1 += 1
        if match_rank and match_rank <= 3:
            hits_at_3 += 1
        if match_rank and match_rank <= 5:
            hits_at_5 += 1

        rr = (1.0 / match_rank) if match_rank else 0.0
        reciprocal_ranks.append(rr)

        first_match = retrieved_docs[0] if retrieved_docs else None
        first_doc = first_match.metadata.get("source") if first_match else "None"
        first_sec = first_match.metadata.get("section") if first_match else "None"

        query_details.append({
            "idx": idx,
            "query": query,
            "expected": expected_doc,
            "expected_keyword": expected_sec,
            "match_rank": match_rank if match_rank else "Not in Top-k",
            "reciprocal_rank": round(rr, 4),
            "top_1_retrieved_source": first_doc,
            "top_1_retrieved_section": first_sec,
        })

        status_icon = "HIT" if (match_rank and match_rank <= 3) else "MISS"
        print(f"[{status_icon}] Q{idx}: '{query[:40]}...' -> Match Rank: {match_rank} (Top doc: {first_doc})")

    total_queries = len(TEST_BENCHMARK)
    hit_rate_at_1 = hits_at_1 / total_queries
    hit_rate_at_3 = hits_at_3 / total_queries
    hit_rate_at_5 = hits_at_5 / total_queries
    mrr = sum(reciprocal_ranks) / total_queries

    print(f"\nSummary for '{search_mode.upper()}':")
    print(f"  Hit Rate @ 1: {hit_rate_at_1*100:.1f}% ({hits_at_1}/{total_queries})")
    print(f"  Hit Rate @ 3: {hit_rate_at_3*100:.1f}% ({hits_at_3}/{total_queries})")
    print(f"  Hit Rate @ 5: {hit_rate_at_5*100:.1f}% ({hits_at_5}/{total_queries})")
    print(f"  MRR:          {mrr:.4f}")

    return {
        "search_mode": search_mode,
        "total_queries": total_queries,
        "hit_rate_at_1": hit_rate_at_1,
        "hit_rate_at_3": hit_rate_at_3,
        "hit_rate_at_5": hit_rate_at_5,
        "mrr": round(mrr, 4),
        "details": query_details
    }


def main():
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Error: ChromaDB directory '{CHROMA_PERSIST_DIR}' does not exist. Run ingest.py first.")
        return

    print("Loading vectorstore & hybrid retriever for retrieval evaluation...")
    vectorstore = load_vectorstore()
    hybrid_retriever = get_retriever()

    # Compare Similarity Search vs MMR vs Hybrid (BM25 + Dense + RRF)
    eval_sim = evaluate_retrieval(vectorstore, hybrid_retriever, search_mode="similarity", k=5)
    eval_mmr = evaluate_retrieval(vectorstore, hybrid_retriever, search_mode="mmr", k=5)
    eval_hybrid = evaluate_retrieval(vectorstore, hybrid_retriever, search_mode="hybrid", k=5)

    # Save results to JSON and Markdown reports
    json_path = os.path.join(BASE_DIR, "retrieval_evaluation_results.json")
    md_path = os.path.join(BASE_DIR, "retrieval_evaluation_report.md")

    report_data = {
        "similarity_search": eval_sim,
        "mmr_search": eval_mmr,
        "hybrid_search": eval_hybrid
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Retrieval Quality Benchmark Report\n\n")
        f.write("Evaluation performed across 10 technical queries comparing Dense Cosine Similarity, MMR, and Hybrid RRF.\n\n")
        f.write("## 1. Overall Summary\n\n")
        f.write("| Metric | Similarity (Dense Cosine) | MMR (Dense lambda=0.7) | Hybrid (BM25 + Dense + RRF) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Hit Rate @ 1** | **{eval_sim['hit_rate_at_1']*100:.1f}%** | **{eval_mmr['hit_rate_at_1']*100:.1f}%** | **{eval_hybrid['hit_rate_at_1']*100:.1f}%** |\n")
        f.write(f"| **Hit Rate @ 3** | **{eval_sim['hit_rate_at_3']*100:.1f}%** | **{eval_mmr['hit_rate_at_3']*100:.1f}%** | **{eval_hybrid['hit_rate_at_3']*100:.1f}%** |\n")
        f.write(f"| **Hit Rate @ 5** | **{eval_sim['hit_rate_at_5']*100:.1f}%** | **{eval_mmr['hit_rate_at_5']*100:.1f}%** | **{eval_hybrid['hit_rate_at_5']*100:.1f}%** |\n")
        f.write(f"| **Mean Reciprocal Rank (MRR)** | **{eval_sim['mrr']:.3f}** | **{eval_mmr['mrr']:.3f}** | **{eval_hybrid['mrr']:.3f}** |\n\n")
        f.write("## 2. Per-Query Breakdown\n\n")
        f.write("| # | Query | Expected Document | Similarity Rank | MMR Rank | Hybrid RRF Rank |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s_q, m_q, h_q in zip(eval_sim['details'], eval_mmr['details'], eval_hybrid['details']):
            f.write(f"| {s_q['idx']} | {s_q['query']} | `{s_q['expected']}` | Rank {s_q['match_rank']} | Rank {m_q['match_rank']} | Rank {h_q['match_rank']} |\n")

    print(f"\n[Saved Reports]")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
