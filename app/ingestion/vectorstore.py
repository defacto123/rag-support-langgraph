"""Qdrant vector store: create the collection and store chunks.

The collection must be created with the SAME vector size as our embedding
model produces (gemini-embedding-001 -> 3072) and COSINE distance, which
pairs with semantic similarity. A size mismatch makes inserts fail.
"""

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings
from app.models import get_embeddings


def _get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def get_vectorstore() -> QdrantVectorStore:
    """Return a QdrantVectorStore, creating the collection if missing."""
    client = _get_client()

    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,   # 3072 for gemini-embedding-001
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )


def add_chunks(chunks: list[Document]) -> int:
    """Embed and store chunks in Qdrant. Returns the number stored."""
    if not chunks:
        return 0

    vectorstore = get_vectorstore()
    # add_documents embeds page_content and stores vector + text + metadata.
    vectorstore.add_documents(chunks)
    return len(chunks)
