"""
Data Ingestion Pipeline for Technical Documentation RAG Assistant
- Document Source: LangGraph, FastAPI, Pydantic, LangChain RAG, ChromaDB
- Chunking: Two-Pass (MarkdownHeaderTextSplitter -> RecursiveCharacterTextSplitter)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (Zero-Cost, Local Inference)
- Vector Store: Persistent ChromaDB
"""

import os
import glob
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DOCS_DIR = os.path.join(BASE_DIR, "raw_docs")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "technical_documentation"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking Parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_raw_markdown_files() -> List[Dict[str, str]]:
    """Loads all markdown files from the raw_docs directory."""
    md_files = glob.glob(os.path.join(RAW_DOCS_DIR, "*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown documents found in {RAW_DOCS_DIR}")

    docs_raw = []
    for filepath in sorted(md_files):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        docs_raw.append({
            "filename": filename,
            "filepath": filepath,
            "content": content
        })
    print(f"Loaded {len(docs_raw)} raw documents from {RAW_DOCS_DIR}")
    return docs_raw


def two_pass_chunking(raw_docs: List[Dict[str, str]]) -> List[Document]:
    """
    Two-Pass Chunking Strategy:
    1. MarkdownHeaderTextSplitter: preserves structural sections & hierarchy
    2. RecursiveCharacterTextSplitter: keeps chunk sizes uniform within sections without splitting code blocks
    """
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n```", "\n```", "\n\n", "\n", ". ", " ", ""]
    )

    all_chunks: List[Document] = []
    chunk_counter = 0

    for doc_item in raw_docs:
        filename = doc_item["filename"]
        content = doc_item["content"]

        # Pass 1: Header splitting
        header_splits = header_splitter.split_text(content)

        # Pass 2: Recursive character splitting
        sub_chunks = char_splitter.split_documents(header_splits)

        for sub_chunk in sub_chunks:
            chunk_counter += 1
            # Build unified section path
            h1 = sub_chunk.metadata.get("Header_1", "")
            h2 = sub_chunk.metadata.get("Header_2", "")
            h3 = sub_chunk.metadata.get("Header_3", "")
            section_hierarchy = " > ".join(filter(None, [h1, h2, h3]))

            # Enrich metadata
            sub_chunk.metadata.update({
                "source": filename,
                "document_name": filename.replace(".md", "").replace("_", " ").title(),
                "section": section_hierarchy or "General",
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "char_length": len(sub_chunk.page_content),
            })
            all_chunks.append(sub_chunk)

    print(f"Generated {len(all_chunks)} chunks across {len(raw_docs)} documents.")
    return all_chunks


def print_chunk_statistics(chunks: List[Document]):
    """Displays detailed statistics about chunk sizes and distribution."""
    lengths = [len(c.page_content) for c in chunks]
    doc_counts: Dict[str, int] = {}
    for c in chunks:
        doc = c.metadata.get("source", "unknown")
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    print("\n" + "="*50)
    print("           CHUNK INGESTION STATISTICS")
    print("="*50)
    print(f"Total Chunks:        {len(chunks)}")
    print(f"Average Chunk Size:  {sum(lengths) / len(chunks):.1f} chars")
    print(f"Min Chunk Size:      {min(lengths)} chars")
    print(f"Max Chunk Size:      {max(lengths)} chars")
    print("\nChunks per document:")
    for doc, count in sorted(doc_counts.items()):
        print(f"  - {doc}: {count} chunks")
    print("="*50 + "\n")


def build_and_persist_vectorstore(chunks: List[Document]) -> Chroma:
    """Embeds chunks using SentenceTransformers and persists to ChromaDB."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"Indexing {len(chunks)} chunks into ChromaDB at: {CHROMA_PERSIST_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    print(f"Successfully persisted vectorstore with collection: '{COLLECTION_NAME}'")
    return vectorstore


def main():
    print("Starting Data Ingestion Pipeline...")
    raw_docs = load_raw_markdown_files()
    chunks = two_pass_chunking(raw_docs)
    print_chunk_statistics(chunks)
    vectorstore = build_and_persist_vectorstore(chunks)
    print("Ingestion Pipeline completed successfully!")


if __name__ == "__main__":
    main()
