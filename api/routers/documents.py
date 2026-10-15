"""
api.routers.documents
---------------------
Corpus management endpoints:
- GET  /documents       : list indexed documents & chunk statistics
- POST /ingest          : ingest new document via file upload or form URL
- POST /ingest/url      : ingest new document via JSON body
- DELETE /documents/{name} : delete document chunks from corpus
"""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from api.schemas import (
    DocumentListResponse,
    IngestResponse,
    IngestURLRequest,
    DeleteDocumentResponse,
)
from api.services.ingestion_service import (
    list_indexed_documents,
    ingest_text_content,
    extract_text_from_url,
    extract_text_from_pdf,
    delete_document_by_name,
)

router = APIRouter(tags=["Document & Corpus Management"])


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Indexed Documents",
)
def get_indexed_documents():
    """
    Returns all documents currently indexed in the ChromaDB vector store,
    including chunk counts, original filenames/URLs, and sample section hierarchies.
    """
    try:
        return list_indexed_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to inspect corpus: {str(exc)}",
        ) from exc


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Document (File Upload or URL Form)",
)
async def ingest_document(
    file: Optional[UploadFile] = File(None, description="Markdown (.md), Text (.txt), or PDF (.pdf) file."),
    url: Optional[str] = Form(None, description="Web page documentation URL to scrape and ingest."),
    document_name: Optional[str] = Form(None, description="Optional custom name for the document."),
):
    """
    Ingests technical documentation from either a file upload or a web URL.

    - **File upload**: Supports `.md`, `.txt`, `.pdf`.
    - **Web URL**: Fetches live HTML, strips boilerplate, and structures headers.
    - Applies two-pass chunking, embeds via sentence-transformers, and updates ChromaDB & BM25.
    """
    if not file and not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either a file upload or a URL.",
        )

    try:
        # Case 1: File Upload
        if file is not None:
            filename = file.filename or "uploaded_document"
            doc_name = document_name or filename.rsplit(".", 1)[0].replace("_", " ").title()
            contents = await file.read()

            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(contents)
            else:
                text = contents.decode("utf-8", errors="replace")

            chunks_added, total = ingest_text_content(
                content=text,
                source=filename,
                document_name=doc_name,
            )
            return IngestResponse(
                status="success",
                document_name=doc_name,
                source=filename,
                chunks_added=chunks_added,
                total_corpus_chunks=total,
                message=f"Successfully indexed {chunks_added} chunks from file '{filename}'.",
            )

        # Case 2: Web URL
        else:
            assert url is not None
            page_title, text = extract_text_from_url(url.strip())
            doc_name = document_name or page_title or url

            chunks_added, total = ingest_text_content(
                content=text,
                source=url.strip(),
                document_name=doc_name,
            )
            return IngestResponse(
                status="success",
                document_name=doc_name,
                source=url.strip(),
                chunks_added=chunks_added,
                total_corpus_chunks=total,
                message=f"Successfully crawled and indexed {chunks_added} chunks from '{url}'.",
            )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ingestion failed: {str(exc)}",
        ) from exc


@router.post(
    "/ingest/url",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Document from URL (JSON Body)",
)
def ingest_from_url_json(payload: IngestURLRequest):
    """
    Convenience endpoint for ingesting documentation directly via JSON payload.
    """
    try:
        url = payload.url.strip()
        page_title, text = extract_text_from_url(url)
        doc_name = payload.document_name or page_title or url

        chunks_added, total = ingest_text_content(
            content=text,
            source=url,
            document_name=doc_name,
        )
        return IngestResponse(
            status="success",
            document_name=doc_name,
            source=url,
            chunks_added=chunks_added,
            total_corpus_chunks=total,
            message=f"Successfully crawled and indexed {chunks_added} chunks from '{url}'.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"URL ingestion failed: {str(exc)}",
        ) from exc


@router.delete(
    "/documents/{document_name}",
    response_model=DeleteDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Document from Corpus",
)
def delete_document(document_name: str):
    """
    Deletes all chunks associated with a document name or filename from ChromaDB
    and automatically rebuilds the BM25 keyword index.
    """
    try:
        deleted_count, remaining = delete_document_by_name(document_name)
        return DeleteDocumentResponse(
            status="success",
            document_name=document_name,
            chunks_deleted=deleted_count,
            remaining_chunks=remaining,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(exc)}",
        ) from exc
