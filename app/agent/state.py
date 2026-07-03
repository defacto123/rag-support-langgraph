"""Shared state for the Adaptive RAG graph.

The state is the single object that flows through every node. A node reads
what it needs and returns a partial dict; LangGraph merges it into the
state. Defining it as a TypedDict documents exactly what data exists and
keeps the nodes honest about their inputs/outputs.
"""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Full conversation history. The `add_messages` reducer APPENDS new
    # messages instead of overwriting, so history accumulates across turns.
    messages: Annotated[list[AnyMessage], add_messages]

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

    # Full text of each retrieved chunk (used by evaluation / Ragas).
    contexts: list[str]

    # The final generated answer.
    generation: str

    # How many times we have rewritten the question (loop guard).
    retries: int

    # Whether the groundedness grader accepted the last generation.
    answer_grounded: bool

    # How many times we have regenerated the answer (loop guard).
    generation_retries: int

    # Consecutive clarification attempts on the SAME point (anti-repetition).
    clarify_count: int
