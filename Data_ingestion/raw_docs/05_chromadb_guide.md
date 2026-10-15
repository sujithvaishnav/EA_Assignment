# ChromaDB Vector Database Guide

## What is ChromaDB?
ChromaDB is an open-source, AI-native vector embedding database. It is lightweight, developer-friendly, and offers native Python support with both in-memory and persistent disk-backed storage modes. It is ideally suited for local RAG systems, development environments, and production document search.

## Core Architecture and Concepts

### 1. Collections
A Collection in ChromaDB is similar to a table in relational databases. It contains:
- Text documents (strings)
- Vector embeddings (lists of floats)
- Metadata (key-value dictionaries for filtering)
- Unique string IDs for each entry

### 2. Distance Metrics
ChromaDB supports three distance metrics for vector similarity search:
- **Cosine Distance** (`cosine`): Best for normalized embedding models like sentence-transformers and OpenAI. Distance ranges from 0 (identical) to 2 (opposite).
- **Squared L2 Distance** (`l2`): Euclidean distance squared.
- **Inner Product** (`ip`): Useful for unnormalized dot-product embeddings.

Configure the distance metric when creating a collection:
```python
collection = client.create_collection(
    name="tech_docs",
    metadata={"hnsw:space": "cosine"}
)
```

## Persistent Client Usage
To ensure vectors survive process restarts, use `PersistentClient`:

```python
import chromadb

# Initializes persistent ChromaDB on local filesystem
client = chromadb.PersistentClient(path="./Data_ingestion/chroma_db")

# Get or create collection
collection = client.get_or_create_collection(
    name="technical_documentation",
    metadata={"hnsw:space": "cosine"}
)
```

## Ingestion and Adding Data
Adding documents, embeddings, and metadata to ChromaDB:

```python
collection.add(
    ids=["doc_1", "doc_2"],
    documents=["LangGraph simplifies cyclic agents", "FastAPI is fast"],
    metadatas=[
        {"source": "langgraph.md", "section": "overview"},
        {"source": "fastapi.md", "section": "intro"}
    ]
)
```
If embeddings are not explicitly supplied, ChromaDB can use its default embedding function or a registered custom function.

## Querying and Metadata Filtering
ChromaDB allows querying by query text (or embedding) with optional metadata filters (`where` clause):

```python
# Semantic search with metadata filter
results = collection.query(
    query_texts=["How to use conditional edges?"],
    n_results=5,
    where={"source": "langgraph.md"}  # Exact metadata match
)

# Complex filtering using operators
results = collection.query(
    query_texts=["FastAPI routing"],
    n_results=3,
    where={
        "$and": [
            {"source": {"$eq": "fastapi.md"}},
            {"chunk_index": {"$lt": 10}}
        ]
    }
)
```

## Integration with LangChain
LangChain provides seamless integration with ChromaDB via `langchain-chroma`:

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="tech_docs",
    embedding_function=embeddings,
    persist_directory="./Data_ingestion/chroma_db"
)
```
