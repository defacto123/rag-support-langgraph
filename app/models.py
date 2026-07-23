"""Factory functions for the Gemini LLM and embedding models.

Everything that needs an LLM or embeddings imports it from here, so the
model configuration lives in exactly one place (driven by settings/.env).
"""

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from app.config import settings


def get_llm(
    temperature: float = 0.0,
    thinking_budget: int | None = None,
) -> ChatGoogleGenerativeAI:
    """Return the chat model.

    temperature=0 -> deterministic answers.

    thinking_budget controls Gemini 2.5's internal "thinking" tokens:
      - None  -> model default (used for the final answer, to keep quality).
      - 0     -> thinking disabled. Use for simple, mechanical calls
                 (routing, grading, query rewriting/contextualising), where
                 chain-of-thought adds latency and cost but no real value.
    """
    kwargs: dict = {}
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        **kwargs,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the embedding model used for both ingestion and search."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
