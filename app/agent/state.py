"""Shared state for the Adaptive RAG graph.

The state is the single object that flows through every node. A node reads
what it needs and returns a partial dict; LangGraph merges it into the
state. Defining it as a TypedDict documents exactly what data exists and
keeps the nodes honest about their inputs/outputs.
"""

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    # The user's current question (may be rewritten during the run).
    question: str

    # Routing decision: "retrieve" (search docs) or "direct" (answer without).
    route: str

    # Chunks retrieved from Qdrant for the current question.
    documents: list[Document]

    # Formatted context string built from `documents` (fed to the LLM).
    context: str

    # Whether the grader judged the retrieved context relevant.
    documents_relevant: bool

    # Structured source list shown to the user (source, page, score, ...).
    sources: list[dict]

    # The final generated answer.
    generation: str

    # How many times we have rewritten the question (loop guard).
    retries: int
