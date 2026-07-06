"""Chunking: split loaded documents into searchable pieces.

Strategy is chosen per document:
  - "legal"     -> split on article/paragraph markers (structure-aware)
  - "technical" -> larger blocks with more overlap
  - "semantic"  -> SemanticChunker cuts where the topic shifts (default
                   for unknown documents)

Why this matters: bad chunking (cutting mid-sentence, splitting related
info) is the #1 cause of poor RAG answers. Good retrieval cannot fix it.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Keyword sets used to guess the document type. Lowercase on purpose.
_LEGAL_KEYWORDS = ["член", "алинея", "наредба", "договор", "закон"]
_TECHNICAL_KEYWORDS = ["function", "class", "import", "def ", "api"]

# A type is only chosen if its keyword count exceeds this threshold,
# otherwise we fall back to semantic chunking.
_KEYWORD_THRESHOLD = 3

# Safeguard: SemanticChunker can produce very large chunks (a whole "topic").
# Anything longer than this many characters is re-split so retrieval stays
# precise. The structure-aware splitters already cap size via chunk_size.
_SEMANTIC_MAX_CHUNK_SIZE = 1200
_SEMANTIC_CHUNK_OVERLAP = 150


def detect_doc_type(docs: list[Document]) -> str:
    """Return 'legal', 'technical', or 'semantic' based on keyword counts."""
    text = " ".join(d.page_content for d in docs).lower()

    scores = {
        "legal": sum(text.count(k) for k in _LEGAL_KEYWORDS),
        "technical": sum(text.count(k) for k in _TECHNICAL_KEYWORDS),
    }

    best = max(scores, key=scores.get)
    if scores[best] > _KEYWORD_THRESHOLD:
        return best

    # Unknown / general document -> let meaning decide the cuts.
    return "semantic"


def get_splitter(doc_type: str, embeddings: Embeddings):
    """Return the right splitter for the detected document type."""
    if doc_type == "legal":
        # Cut on legal structure first, then fall back to generic separators.
        return RecursiveCharacterTextSplitter(
            separators=["\nЧлен ", "\nАлинея ", "\n\n", "\n", " "],
            chunk_size=600,
            chunk_overlap=100,
        )

    if doc_type == "technical":
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=800,
            chunk_overlap=200,
        )

    if doc_type == "fixed":
        # Fast, embedding-free fixed-size splitting. Ideal for bulk ingestion
        # of many short/medium docs where semantic chunking would add a costly
        # embedding pass per document for little benefit.
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=1000,
            chunk_overlap=150,
        )

    if doc_type == "semantic":
        # Measures similarity between sentences and splits on topic shifts.
        # Slower (calls the embedding model) but smart for unknown docs.
        return SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )

    # Safety fallback if an unexpected type ever reaches here.
    return RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)


def _cap_chunk_sizes(chunks: list[Document]) -> list[Document]:
    """Re-split any oversized semantic chunk into smaller pieces.

    Keeps the original metadata on every resulting piece.
    """
    guard = RecursiveCharacterTextSplitter(
        chunk_size=_SEMANTIC_MAX_CHUNK_SIZE,
        chunk_overlap=_SEMANTIC_CHUNK_OVERLAP,
    )

    capped: list[Document] = []
    for chunk in chunks:
        if len(chunk.page_content) <= _SEMANTIC_MAX_CHUNK_SIZE:
            capped.append(chunk)
        else:
            # split_documents preserves metadata across the new pieces.
            capped.extend(guard.split_documents([chunk]))
    return capped


def enrich_metadata(chunks: list[Document], doc_type: str) -> list[Document]:
    """Stamp each chunk with positional + type metadata.

    `source` and `page` are already set by the loaders; here we add the
    info that lets us cite sources precisely and debug retrieval later.
    """
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "doc_type": doc_type,
                "chunk_index": i,
                "total_chunks": total,
                "has_prev": i > 0,
                "has_next": i < total - 1,
            }
        )
    return chunks


def chunk_documents(
    docs: list[Document],
    embeddings: Embeddings,
    force_doc_type: str | None = None,
) -> tuple[list[Document], str]:
    """Detect type, pick a splitter, and split. Returns (chunks, doc_type).

    force_doc_type overrides auto-detection (e.g. "fixed" for fast bulk
    ingestion). When None, behaviour is unchanged.
    """
    doc_type = force_doc_type or detect_doc_type(docs)
    splitter = get_splitter(doc_type, embeddings)
    chunks = splitter.split_documents(docs)

    # Apply the size cap only to semantic output (structure-aware splitters
    # already enforce chunk_size).
    if doc_type == "semantic":
        chunks = _cap_chunk_sizes(chunks)

    # Drop empty / whitespace-only chunks: they carry no information and
    # waste embedding calls (or error out on some models).
    chunks = [c for c in chunks if c.page_content.strip()]

    # Add positional metadata after filtering so indices stay contiguous.
    chunks = enrich_metadata(chunks, doc_type)

    return chunks, doc_type
