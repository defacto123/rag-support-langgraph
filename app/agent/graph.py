"""Assemble the Adaptive RAG graph.

Full shape:

    route -> (generate_direct | retrieve)
    retrieve -> grade_documents -> (generate | rewrite -> retrieve | respond_no_context)
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    generate,
    generate_direct,
    grade_documents,
    respond_no_context,
    retrieve,
    rewrite,
    route,
)
from app.agent.state import AgentState

# Max number of question rewrites before giving up (loop guard).
MAX_RETRIES = 2


def decide_route(state: AgentState) -> str:
    """Conditional edge out of `route`: search documents or answer directly."""
    return "retrieve" if state.get("route") == "retrieve" else "generate_direct"


def decide_after_grading(state: AgentState) -> str:
    """Conditional edge: route based on the grader's verdict.

    - relevant           -> generate the answer
    - not relevant, retries left -> rewrite the question and try again
    - not relevant, no retries    -> give up gracefully
    """
    if state.get("documents_relevant"):
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        return "rewrite"
    return "respond_no_context"


def build_graph():
    """Build and compile the agent graph."""
    builder = StateGraph(AgentState)

    builder.add_node("route", route)
    builder.add_node("generate_direct", generate_direct)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)
    builder.add_node("respond_no_context", respond_no_context)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        decide_route,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )
    builder.add_edge("generate_direct", END)
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "respond_no_context": "respond_no_context",
        },
    )
    # The loop: after rewriting, retrieve again with the new question.
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("respond_no_context", END)

    # The checkpointer persists state per thread_id, giving the agent memory
    # across turns of the same conversation.
    return builder.compile(checkpointer=MemorySaver())


# Compiled once and reused.
_graph = build_graph()


def ask(question: str, thread_id: str = "default") -> dict:
    """Run the agent for one turn of a conversation.

    thread_id identifies the conversation; reusing it resumes prior state
    (memory). New thread_id = a fresh, isolated conversation.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = _graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "retries": 0,
        },
        config=config,
    )
    return {
        "answer": result.get("generation", ""),
        "sources": result.get("sources", []),
    }
