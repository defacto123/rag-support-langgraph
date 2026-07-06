"""Bulk-ingest the scraped MobiSystems KB into a Qdrant collection.

Reads every .md under data/uploads/mobisystems, chunks with the fast
embedding-free "fixed" strategy, and upserts embeddings in batches.

Target collection is taken from QDRANT_COLLECTION (set it to "mobisystems"):

  QDRANT_COLLECTION=mobisystems python -m scripts.ingest_mobisystems

The collection is dropped and recreated first, so re-runs are idempotent.
"""

from pathlib import Path

from qdrant_client import QdrantClient

from app.config import settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_document
from app.ingestion.vectorstore import get_vectorstore
from app.models import get_embeddings

KB_ROOT = Path("data/uploads/mobisystems")
BATCH = 128


def _reset_collection() -> None:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    if client.collection_exists(settings.qdrant_collection):
        print(f"Dropping existing collection '{settings.qdrant_collection}'…")
        client.delete_collection(settings.qdrant_collection)


def main() -> None:
    files = sorted(KB_ROOT.rglob("*.md"))
    if not files:
        print(f"No .md files under {KB_ROOT}. Run the scraper first.")
        return

    print(f"Ingesting {len(files)} files into '{settings.qdrant_collection}'…")
    _reset_collection()

    embeddings = get_embeddings()

    # Chunk everything first (fixed = no embedding calls during chunking).
    all_chunks = []
    failed = 0
    for i, path in enumerate(files, start=1):
        try:
            docs = load_document(str(path))
            chunks, _ = chunk_documents(docs, embeddings, force_doc_type="fixed")
            all_chunks.extend(chunks)
        except Exception as exc:  # keep going; report at the end
            failed += 1
            print(f"  ! failed {path.name}: {exc}")
        if i % 250 == 0:
            print(f"  chunked {i}/{len(files)} files… ({len(all_chunks)} chunks)")

    print(f"Total chunks: {len(all_chunks)} (failed files: {failed})")

    # Embed + upsert in batches to minimise API round-trips.
    vectorstore = get_vectorstore()
    stored = 0
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start : start + BATCH]
        vectorstore.add_documents(batch)
        stored += len(batch)
        print(f"  stored {stored}/{len(all_chunks)} chunks…")

    print(f"\nDONE. Stored {stored} chunks in '{settings.qdrant_collection}'.")


if __name__ == "__main__":
    main()
