"""Graph nodes for the Adaptive RAG agent.

Each node is a function `(state) -> partial state`. Nodes never mutate the
incoming state in place; they return only the keys they changed.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.models import get_llm
from app.retrieval.search import search


class GradeDocuments(BaseModel):
    """Structured verdict on whether the context answers the question."""

    relevant: bool = Field(
        description="True if the context contains information that helps "
        "answer the question, otherwise False."
    )


_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ти си оценител. Реши дали предоставеният контекст съдържа "
            "информация, полезна за отговор на въпроса. Върни само true/false.",
        ),
        (
            "human",
            "Контекст:\n{context}\n\nВъпрос: {question}",
        ),
    ]
)

# Prompt for the final answer. The model must ground its answer in the
# provided context and cite sources; if the context is empty it must say so
# instead of inventing an answer (anti-hallucination).
_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ти си асистент, който отговаря на въпроси само въз основа на "
            "предоставения контекст. Отговаряй на езика на въпроса. "
            "Цитирай източниците като [Източник N], когато ги ползваш. "
            "Ако контекстът не съдържа отговора, честно кажи, че нямаш "
            "информация по въпроса. Не измисляй.",
        ),
        (
            "human",
            "История на разговора:\n{history}\n\nКонтекст:\n{context}\n\n"
            "Въпрос: {question}\n\nОтговор:",
        ),
    ]
)


def _history_text(state: AgentState, max_msgs: int = 6) -> str:
    """Format recent conversation history (excluding the current question)."""
    msgs = state.get("messages", [])
    prior = msgs[:-1][-max_msgs:]  # drop the current turn, keep last N
    lines = []
    for m in prior:
        role = "Потребител" if m.type == "human" else "Асистент"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Дадени са история на разговор и последен въпрос. Преформулирай "
            "последния въпрос в САМОСТОЯТЕЛЕН въпрос, разбираем без историята "
            "(замести местоимения/препратки с конкретното нещо). Ако вече е "
            "самостоятелен, върни го непроменен. Върни само въпроса.",
        ),
        ("human", "История:\n{history}\n\nПоследен въпрос: {question}"),
    ]
)


class RouteQuery(BaseModel):
    """Decision on whether a question needs document retrieval."""

    needs_documents: bool = Field(
        description="True if answering requires searching the uploaded "
        "documents; False for greetings or general chit-chat."
    )


_ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Реши дали въпросът изисква търсене в качените документи. "
            "Поздрави, любезности и общи разговорни реплики НЕ изискват "
            "търсене. Конкретни въпроси за съдържание изискват търсене. "
            "Използвай историята, за да разбереш follow-up въпроси.",
        ),
        ("human", "История:\n{history}\n\nВъпрос: {question}"),
    ]
)


def route(state: AgentState) -> AgentState:
    """Classify the question: needs document retrieval or a direct reply."""
    router = get_llm().with_structured_output(RouteQuery)
    decision: RouteQuery = (_ROUTE_PROMPT | router).invoke(
        {"question": state["question"], "history": _history_text(state)}
    )
    return {"route": "retrieve" if decision.needs_documents else "direct"}


_DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ти си любезен асистент за документи. Отговори кратко на езика "
            "на потребителя. Ако е уместно, поясни, че можеш да отговаряш "
            "на въпроси относно качените документи.",
        ),
        ("human", "{question}"),
    ]
)


def generate_direct(state: AgentState) -> AgentState:
    """Answer general/greeting questions without touching the documents."""
    llm = get_llm(temperature=0.3)
    response = (_DIRECT_PROMPT | llm).invoke({"question": state["question"]})
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def retrieve(state: AgentState) -> AgentState:
    """Search the vector store for chunks relevant to the question.

    For follow-up questions, first rewrite the question into a standalone
    query using the conversation history, so the search is meaningful.
    """
    query = state["question"]
    history = _history_text(state)
    if history:
        llm = get_llm()
        query = (
            (_CONTEXTUALIZE_PROMPT | llm)
            .invoke({"history": history, "question": query})
            .content.strip()
        )

    result = search(query)
    return {
        "context": result["context"],
        "sources": result["sources"],
    }


def grade_documents(state: AgentState) -> AgentState:
    """Ask the LLM whether the retrieved context is relevant (yes/no)."""
    context = state.get("context")
    if not context:
        # Nothing retrieved -> definitely not relevant, skip the LLM call.
        return {"documents_relevant": False}

    grader = get_llm().with_structured_output(GradeDocuments)
    verdict: GradeDocuments = (_GRADE_PROMPT | grader).invoke(
        {"context": context, "question": state["question"]}
    )
    return {"documents_relevant": verdict.relevant}


def generate(state: AgentState) -> AgentState:
    """Generate the final answer grounded in the retrieved context."""
    context = state.get("context") or "(няма намерен контекст)"

    llm = get_llm()
    chain = _ANSWER_PROMPT | llm
    response = chain.invoke(
        {
            "context": context,
            "question": state["question"],
            "history": _history_text(state),
        }
    )

    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content)],
    }


_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Преформулирай въпроса така, че да е по-подходящ за търсене "
            "в база от документи. Запази смисъла и езика. Върни само "
            "преформулирания въпрос, без обяснения.",
        ),
        ("human", "Оригинален въпрос: {question}"),
    ]
)


def rewrite(state: AgentState) -> AgentState:
    """Reformulate the question to improve retrieval, and count the attempt."""
    llm = get_llm()
    response = (_REWRITE_PROMPT | llm).invoke({"question": state["question"]})
    return {
        "question": response.content.strip(),
        "retries": state.get("retries", 0) + 1,
    }


def respond_no_context(state: AgentState) -> AgentState:
    """Fallback answer when no relevant context was found."""
    msg = (
        "Нямам достатъчно информация в наличните документи, за да "
        "отговоря на този въпрос."
    )
    return {"generation": msg, "messages": [AIMessage(content=msg)]}
