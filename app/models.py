"""Factory functions for the Gemini LLM and embedding models.

Everything that needs an LLM or embeddings imports it from here, so the
model configuration lives in exactly one place (driven by settings/.env).
"""

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from app.config import settings


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return the chat model. temperature=0 -> deterministic answers."""
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the embedding model used for both ingestion and search."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
