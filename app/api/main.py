"""FastAPI backend exposing the RAG agent over HTTP.

Endpoints:
  GET  /health  -> liveness check (used by deploy platforms)
  POST /upload  -> upload a document and ingest it into Qdrant
  POST /chat    -> ask a question; returns answer + sources
"""

import logging
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.agent.graph import ask
from app.config import settings
from app.ingestion.pipeline import ingest_document

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Document Assistant")

# Strips stray inline citation markers (e.g. "[Source 1]", "[Източник 2]")
# from the answer text. Citations are intentionally hidden from users; the
# source data is kept in the logs for debugging instead.
_CITATION_RE = re.compile(r"\s*\[(?:Source|Източник)\s*\d+\]", re.IGNORECASE)

# Where uploaded files are stored before ingestion.
_UPLOAD_DIR = Path("data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Save the uploaded file to disk and ingest it into the vector store."""
    if settings.disable_upload:
        # Pre-indexed / chat-only deployment: uploads are turned off.
        raise HTTPException(status_code=403, detail="Uploads are disabled.")

    dest = _UPLOAD_DIR / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    return ingest_document(str(dest))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the agent for one conversation turn.

    Source metadata is not shown to the user; it is logged here for
    debugging and the answer is cleaned of any inline citation markers.
    """
    result = ask(request.question, thread_id=request.thread_id)

    # Keep retrieval provenance available for debugging (logs only).
    logger.info(
        "chat thread=%s sources=%s",
        request.thread_id,
        result.get("sources", []),
    )
    logger.debug("chat thread=%s contexts=%s", request.thread_id,
                 result.get("contexts", []))

    answer = _CITATION_RE.sub("", result["answer"]).strip()
    return ChatResponse(answer=answer, sources=result["sources"])
