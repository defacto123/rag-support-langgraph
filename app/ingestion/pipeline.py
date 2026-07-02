"""Ingestion pipeline: the single public entry point for adding a document.

Wires the building blocks together in order:
    load -> chunk (+ enrich metadata) -> store in Qdrant

The API and UI call `ingest_document(path)` and never touch the internals.
"""

from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_document
from app.ingestion.vectorstore import add_chunks
from app.models import get_embeddings


def ingest_document(file_path: str) -> dict:
    """Load, chunk, and store a single document.

    Returns a small summary useful for logging and API responses.
    """
    docs = load_document(file_path)

    embeddings = get_embeddings()
    chunks, doc_type = chunk_documents(docs, embeddings)

    stored = add_chunks(chunks)

    return {
        "file": docs[0].metadata.get("source", file_path),
        "doc_type": doc_type,
        "pages": len(docs),
        "chunks": stored,
        "status": "success",
    }
