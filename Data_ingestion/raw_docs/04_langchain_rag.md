# LangChain Retrieval-Augmented Generation (RAG) Architecture

## What is RAG?
Retrieval-Augmented Generation (RAG) is a pattern that enhances Large Language Models by retrieving relevant factual information from external knowledge sources and feeding that context into the LLM prompt. This prevents hallucinations, grounds answers in verified data, and allows private domain knowledge without costly model retraining.

## The Standard RAG Pipeline
The complete RAG pipeline consists of two main stages:

### Stage 1: Ingestion Pipeline (Offline)
1. **Load**: Ingest unstructured data from files, web pages, or databases using `DocumentLoader`.
2. **Transform / Split**: Split large documents into manageable text chunks using text splitters.
3. **Embed**: Transform text chunks into dense mathematical vector representations using embedding models.
4. **Store**: Index vectors and chunk metadata into a vector database (e.g., ChromaDB, FAISS).

### Stage 2: Retrieval & Generation (Online)
1. **Retrieve**: Given a user query, embed the query and retrieve top-k semantically relevant chunks from the vector store.
2. **Augment**: Assemble the retrieved chunks into a prompt template alongside instructions and user question.
3. **Generate**: Pass the augmented prompt to the LLM to synthesize a grounded, accurate response.

## Text Chunking Strategies in LangChain

### 1. RecursiveCharacterTextSplitter
The recommended general-purpose splitter. It uses a hierarchy of separators (`["\n\n", "\n", " ", ""]`) to split text while keeping related paragraphs together:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
```

### 2. MarkdownHeaderTextSplitter
Specifically tailored for structured Markdown documents. It splits along headings (`#`, `##`, `###`) and attaches the header trail as metadata to each resulting chunk, preserving context:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)
header_chunks = markdown_splitter.split_text(markdown_content)
```

## Vector Stores and Retrievers

### 1. Vector Store Operations
Vector stores index chunk embeddings and metadata for fast approximate nearest neighbor (ANN) search:
- `.add_documents(docs)`: Adds documents and vectors.
- `.similarity_search(query, k=4)`: Performs standard cosine or L2 similarity search.
- `.similarity_search_with_score(query, k=4)`: Returns documents alongside their distance scores.

### 2. Maximal Marginal Relevance (MMR) Retrieval
Standard similarity search can retrieve redundant chunks that cover the same fact repeatedly. MMR balances two objectives:
1. **Relevance**: How similar each chunk is to the query.
2. **Diversity**: How novel each chunk is relative to already selected chunks.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           # Number of documents to return
        "fetch_k": 20,    # Number of candidates to consider before reranking
        "lambda_mult": 0.7 # 1.0 = maximum similarity, 0.0 = maximum diversity
    }
)
```

## Provenance and Citations
A high-quality RAG system always tracks chunk provenance. By preserving metadata (`source_url`, `section`, `document_name`, `chunk_id`), the generation node can cite the exact origins of every claim in its generated answer.
