"""Seed the demo 'documents' collection with the English demo knowledge base.

Reads every file under data/demo, chunks it with the normal (auto-detected)
strategy, and upserts embeddings into the Qdrant collection named by
QDRANT_COLLECTION (defaults to "documents").

  python -m scripts.seed_demo

The collection is dropped and recreated first, so re-runs are idempotent.
"""

from pathlib import Path

from qdrant_client import QdrantClient

from app.config import settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_document
from app.ingestion.vectorstore import get_vectorstore
from app.models import get_embeddings

DEMO_ROOT = Path("data/demo")


def _reset_collection() -> None:
    client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
    )
    if client.collection_exists(settings.qdrant_collection):
        print(f"Dropping existing collection '{settings.qdrant_collection}'…")
        client.delete_collection(settings.qdrant_collection)


def main() -> None:
    files = sorted(p for p in DEMO_ROOT.rglob("*") if p.is_file())
    if not files:
        print(f"No files under {DEMO_ROOT}.")
        return

    print(f"Seeding {len(files)} demo files into '{settings.qdrant_collection}'…")
    _reset_collection()

    embeddings = get_embeddings()
    vectorstore = get_vectorstore()

    total = 0
    for path in files:
        docs = load_document(str(path))
        chunks, doc_type = chunk_documents(docs, embeddings)
        vectorstore.add_documents(chunks)
        total += len(chunks)
        print(f"  {path.name}: {len(chunks)} chunks ({doc_type})")

    print(f"\nDONE. Stored {total} chunks in '{settings.qdrant_collection}'.")


if __name__ == "__main__":
    main()
