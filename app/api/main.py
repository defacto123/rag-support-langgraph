"""FastAPI backend exposing the RAG agent over HTTP.

Endpoints:
  GET  /health  -> liveness check (used by deploy platforms)
  POST /upload  -> upload a document and ingest it into Qdrant
  POST /chat    -> ask a question; returns answer + sources
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app.agent.graph import ask
from app.ingestion.pipeline import ingest_document

app = FastAPI(title="AI Document Assistant")

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
    dest = _UPLOAD_DIR / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    return ingest_document(str(dest))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the agent for one conversation turn."""
    result = ask(request.question, thread_id=request.thread_id)
    return ChatResponse(answer=result["answer"], sources=result["sources"])
