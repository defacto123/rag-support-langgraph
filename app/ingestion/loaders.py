"""Document loaders: turn a file on disk into LangChain Documents.

Each loader returns a list of `Document` objects. A Document has:
  - page_content: the extracted text
  - metadata: dict with info like source file and page number

We attach a `source` (the filename) to every document here, because
that is the foundation of the "sources / citations" feature later.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

# Map file extension -> loader class.
# PyPDFLoader splits a PDF into one Document per page (keeps page numbers).
_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_document(file_path: str) -> list[Document]:
    """Load a single file into a list of Documents.

    Raises ValueError for unsupported file types so the caller fails
    fast with a clear message instead of a cryptic error deeper down.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    loader_cls = _LOADERS.get(ext)
    if loader_cls is None:
        supported = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {supported}"
        )

    docs = loader_cls(str(path)).load()

    # Guarantee every document carries the source filename. Some loaders
    # set metadata["source"] to the full path; we normalise to just the name.
    for doc in docs:
        doc.metadata["source"] = path.name

    return docs
