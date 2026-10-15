"""
api.services.ingestion_service
------------------------------
Handles document ingestion from file uploads (.md, .txt, .pdf) or web URLs,
runs two-pass chunking, persists to ChromaDB, and reloads the HybridRetriever.
"""

import io
import os
import re
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipeline.nodes.retrieve import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    reload_retriever,
)
from api.schemas import DocumentItem, DocumentListResponse

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


_EMBEDDINGS = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )



def extract_text_from_url(url: str) -> Tuple[str, str]:
    """Fetches web page content, extracts text, and returns (title, clean_text)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script, style, nav, footer tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else urlparse(url).path.split("/")[-1] or url

    # Extract headings and paragraphs as markdown-like text
    lines = []
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "pre", "code", "li"]):
        text = elem.get_text(separator=" ", strip=True)
        if not text:
            continue
        if elem.name == "h1":
            lines.append(f"\n# {text}\n")
        elif elem.name == "h2":
            lines.append(f"\n## {text}\n")
        elif elem.name == "h3":
            lines.append(f"\n### {text}\n")
        elif elem.name in ["pre", "code"]:
            lines.append(f"\n```\n{text}\n```\n")
        else:
            lines.append(f"{text}\n")

    content = "\n".join(lines).strip()
    return title, content


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from raw PDF bytes using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page_idx, page in enumerate(reader.pages, start=1):
        txt = page.extract_text()
        if txt:
            pages_text.append(f"\n## Page {page_idx}\n{txt}")
    return "\n".join(pages_text).strip()


def chunk_document_content(content: str, source: str, document_name: str) -> List[Document]:
    """Applies two-pass chunking to the text content."""
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n```", "\n```", "\n\n", "\n", ". ", " ", ""],
    )

    header_splits = header_splitter.split_text(content)
    sub_chunks = char_splitter.split_documents(header_splits)

    final_chunks: List[Document] = []
    for idx, chunk in enumerate(sub_chunks, start=1):
        h1 = chunk.metadata.get("Header_1", "")
        h2 = chunk.metadata.get("Header_2", "")
        h3 = chunk.metadata.get("Header_3", "")
        section = " > ".join(filter(None, [h1, h2, h3])) or "General"

        chunk.metadata.update({
            "source": source,
            "document_name": document_name,
            "section": section,
            "chunk_id": f"{re.sub(r'[^a-zA-Z0-9_]', '_', source)}_{idx:04d}",
            "char_length": len(chunk.page_content),
        })
        final_chunks.append(chunk)

    return final_chunks


def ingest_text_content(content: str, source: str, document_name: str) -> Tuple[int, int]:
    """
    Chunks, embeds, and indexes text content into ChromaDB.
    Returns (chunks_added, total_corpus_chunks).
    """
    chunks = chunk_document_content(content, source=source, document_name=document_name)
    if not chunks:
        raise ValueError(f"No extractable text chunks found in source: {source}")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    # Rebuild in-memory BM25 index immediately
    reload_retriever()

    # Get updated total count
    total_chunks = len(vectorstore.get().get("ids", []))
    return len(chunks), total_chunks


def list_indexed_documents() -> DocumentListResponse:
    """Inspects ChromaDB and returns grouped document statistics."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        col = client.get_collection(name=COLLECTION_NAME)
        data = col.get()
    except Exception:
        data = {"ids": [], "metadatas": []}

    metadatas = data.get("metadatas", []) or []
    doc_groups: Dict[str, Dict[str, Any]] = {}

    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source", "unknown")
        doc_name = meta.get("document_name", source)
        section = meta.get("section", "General")

        if source not in doc_groups:
            doc_groups[source] = {
                "document_name": doc_name,
                "source": source,
                "chunk_count": 0,
                "sections": set(),
            }

        doc_groups[source]["chunk_count"] += 1
        if section and section != "General":
            doc_groups[source]["sections"].add(section)

    items = []
    for src, info in sorted(doc_groups.items()):
        items.append(
            DocumentItem(
                document_name=info["document_name"],
                source=info["source"],
                chunk_count=info["chunk_count"],
                sample_sections=sorted(list(info["sections"]))[:5],
            )
        )

    return DocumentListResponse(
        total_documents=len(items),
        total_chunks=len(metadatas),
        documents=items,
    )


def delete_document_by_name(document_name: str) -> Tuple[int, int]:
    """
    Deletes all chunks associated with a document (matching source or document_name).
    Returns (chunks_deleted, remaining_chunks).
    """
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = client.get_collection(name=COLLECTION_NAME)
    data = col.get()

    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])

    matching_ids = []
    for idx, meta in enumerate(metadatas):
        if not meta:
            continue
        src = meta.get("source", "")
        name = meta.get("document_name", "")
        if (
            document_name.lower() in src.lower()
            or document_name.lower() in name.lower()
            or src.lower() == document_name.lower()
        ):
            matching_ids.append(ids[idx])

    if not matching_ids:
        raise FileNotFoundError(f"No indexed document found matching name: '{document_name}'")

    col.delete(ids=matching_ids)
    reload_retriever()

    remaining_chunks = len(col.get().get("ids", []))
    return len(matching_ids), remaining_chunks
