"""
Hybrid Retrieval Node (BM25 Keyword + Dense Vector + Reciprocal Rank Fusion)
Standalone LangGraph node for Node 2 in the RAG workflow.

Architecture:
1. Dense Retrieval (ChromaDB + SentenceTransformers): Captures semantic context and explanations.
2. Keyword Retrieval (BM25Okapi): Captures exact code syntax, function names, and decorators without neighbor suppression.
3. Reciprocal Rank Fusion (RRF): Merges both rankings fairly using 1 / (60 + rank).
"""

import os
import sys
import re

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from rag_pipeline.state import RAGState


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "Data_ingestion", "chroma_db")
COLLECTION_NAME = "technical_documentation"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize_code_and_text(text: str) -> List[str]:
    """
    Tokenizes both technical code keywords and natural text.
    Preserves identifiers with underscores (e.g., field_validator, add_edge).
    """
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class HybridRetriever:
    """
    Hybrid Retriever combining Dense Vector search (ChromaDB)
    and Sparse Keyword search (BM25) fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        rrf_k: int = 60,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.rrf_k = rrf_k

        # 1. Initialize Dense Vector Store
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

        # 2. Build BM25 Index from all chunks in ChromaDB
        self._init_bm25()

    def _init_bm25(self):
        """Extracts all chunks from ChromaDB and initializes BM25Okapi."""
        data = self.vectorstore.get()
        contents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])

        self.corpus_docs: List[Document] = []
        tokenized_corpus: List[List[str]] = []

        for idx, text in enumerate(contents):
            meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            doc_id = ids[idx] if idx < len(ids) else f"doc_{idx}"
            meta["id"] = doc_id
            doc = Document(page_content=text, metadata=meta)
            self.corpus_docs.append(doc)
            tokenized_corpus.append(tokenize_code_and_text(text))

        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[HybridRetriever] Initialized with {len(self.corpus_docs)} documents.")

    def search_dense(self, query: str, top_k: int = 10) -> List[Document]:
        """Performs Dense Cosine Similarity search."""
        return self.vectorstore.similarity_search(query, k=top_k)

    def search_bm25(self, query: str, top_k: int = 10) -> List[Document]:
        """Performs BM25 Keyword search."""
        tokenized_query = tokenize_code_and_text(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        # Get top-k indices sorted descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        bm25_docs = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                bm25_docs.append(self.corpus_docs[idx])
        return bm25_docs

    def reciprocal_rank_fusion(
        self,
        dense_docs: List[Document],
        bm25_docs: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """
        Merges Dense and BM25 results using Reciprocal Rank Fusion (RRF).
        RRF Score = SUM( 1 / (rrf_k + rank) )
        """
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}

        def make_key(doc: Document) -> str:
            # Unique identifier using chunk_id or content hash
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                return f"{doc.metadata.get('source', '')}:{chunk_id}"
            return doc.page_content.strip()

        # Score Dense ranks (1-indexed)
        for rank, doc in enumerate(dense_docs, start=1):
            key = make_key(doc)
            doc_store[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))

        # Score BM25 ranks (1-indexed)
        for rank, doc in enumerate(bm25_docs, start=1):
            key = make_key(doc)
            doc_store[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort documents by total RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        final_docs: List[Document] = []
        for key in sorted_keys[:top_k]:
            doc = doc_store[key]
            # Attach RRF score to metadata for transparency and grading
            doc.metadata["rrf_score"] = round(rrf_scores[key], 6)
            final_docs.append(doc)

        return final_docs

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 10) -> List[Document]:
        """
        Full Hybrid Retrieval pipeline:
        1. Dense Top-10
        2. BM25 Top-10
        3. RRF Fusion -> Top-5
        """
        dense_results = self.search_dense(query, top_k=candidate_pool)
        bm25_results = self.search_bm25(query, top_k=candidate_pool)
        return self.reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)

    def reload(self):
        """Reloads the ChromaDB collection and rebuilds the BM25 index."""
        print("[HybridRetriever] Reloading vectorstore and rebuilding BM25 index...")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        self._init_bm25()


# Singleton instance to avoid reloading models on every node call
_GLOBAL_RETRIEVER: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = HybridRetriever()
    return _GLOBAL_RETRIEVER


def reload_retriever() -> HybridRetriever:
    """Forces reload of the retriever after document ingestion or deletion."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is not None:
        _GLOBAL_RETRIEVER.reload()
    else:
        _GLOBAL_RETRIEVER = HybridRetriever()
    return _GLOBAL_RETRIEVER


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    LangGraph Retrieval Node.

    Reads:
      state["rewritten_query"] (if present) OR state["question"]
    Writes:
      {"documents": List[Document]}
    """
    query = state.get("rewritten_query") or state.get("question")
    if not query:
        raise ValueError("[RetrieveNode] Error: No query or question provided in state.")

    retriever = get_retriever()
    documents = retriever.retrieve(query=query, top_k=5)

    print(f"\n[RetrieveNode] Retrieved {len(documents)} chunks via Hybrid (BM25 + Dense + RRF) for:")
    print(f"  Query: '{query}'")
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "unknown")
        rrf = doc.metadata.get("rrf_score", "N/A")
        print(f"  [{idx}] {source} -> {section} (RRF: {rrf})")

    return {"documents": documents}


if __name__ == "__main__":
    print("Testing HybridRetriever standalone...")
    test_query = "How do field_validator and model_validator work in Pydantic V2?"
    retriever = get_retriever()
    results = retriever.retrieve(test_query, top_k=5)
    print(f"\nResults for '{test_query}':")
    for r in results:
        print(f" - {r.metadata.get('source')} | {r.metadata.get('section')} | RRF: {r.metadata.get('rrf_score')}")
