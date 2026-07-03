"""Graph nodes for the Adaptive RAG agent.

Each node is a function `(state) -> partial state`. Nodes never mutate the
incoming state in place; they return only the keys they changed.
"""

from typing import Literal

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
    """Decision on how to handle the user's message."""

    intent: Literal["retrieve", "direct", "clarify"] = Field(
        description=(
            "'retrieve' for concrete questions about document content; "
            "'direct' for greetings/small talk; "
            "'clarify' when the user says they do not understand the "
            "previous answer and asks for a clearer explanation."
        )
    )


_ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Класифицирай намерението на потребителя:\n"
            "- retrieve: конкретен въпрос за съдържанието на документите "
            "(вкл. follow-up въпроси).\n"
            "- direct: поздрав, любезност, общ разговор.\n"
            "- clarify: потребителят НЕ е разбрал предишния отговор и иска "
            "по-ясно обяснение (напр. 'не разбирам', 'какво значи това', "
            "'къде е това').\n"
            "Използвай историята, за да разпознаеш follow-up и clarify.",
        ),
        ("human", "История:\n{history}\n\nСъобщение: {question}"),
    ]
)


def route(state: AgentState) -> AgentState:
    """Classify the message: retrieve, direct answer, or clarification."""
    router = get_llm().with_structured_output(RouteQuery)
    decision: RouteQuery = (_ROUTE_PROMPT | router).invoke(
        {"question": state["question"], "history": _history_text(state)}
    )
    return {"route": decision.intent}


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
    return {"generation": response.content, "clarify_count": 0}


class ElaborationResult(BaseModel):
    """A clarification with an explicit knowledge-source gate."""

    explanation: str = Field(description="The clearer re-explanation.")
    used_general_knowledge: bool = Field(
        description="True if the explanation relied on common general "
        "knowledge (not on the documents) to clarify a universal concept."
    )
    confident: bool = Field(
        description="True only if you are certain the explanation is "
        "correct and not misleading. False if there is any doubt."
    )
    same_point_as_before: bool = Field(
        description="True if the user is still stuck on the SAME point as "
        "the previous clarification; False if this is a NEW point."
    )


# After this many consecutive clarifications on the same point, stop
# rephrasing and show the raw source text instead.
MAX_CLARIFY_REPEATS = 2


_ELABORATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Потребителят не е разбрал предишния отговор. Обясни по-ясно и "
            "по-просто.\n"
            "ПРАВИЛА:\n"
            "1. Факти за документа взимай САМО от контекста — не измисляй.\n"
            "2. За общоизвестни понятия (напр. как работи мишка, какво е "
            "бутон) можеш да ползваш обща култура, НО само ако си сигурен; "
            "тогава сложи used_general_knowledge=true.\n"
            "3. Ако не можеш да поясниш сигурно и вярно, сложи "
            "confident=false.\n"
            "Отговаряй на езика на потребителя.",
        ),
        (
            "human",
            "История:\n{history}\n\nКонтекст от документите:\n{context}\n\n"
            "Молба за пояснение: {question}",
        ),
    ]
)


def elaborate(state: AgentState) -> AgentState:
    """Re-explain the previous answer with a knowledge-source gate.

    Domain facts stay grounded in the stored context; general-knowledge
    clarifications are allowed only when the model is confident, and are
    clearly labelled. Low confidence -> honest refusal instead of guessing.
    """
    grader = get_llm(temperature=0.2).with_structured_output(ElaborationResult)
    result: ElaborationResult = (_ELABORATE_PROMPT | grader).invoke(
        {
            "history": _history_text(state),
            "context": state.get("context", "(няма запазен контекст)"),
            "question": state["question"],
        }
    )

    # New point resets the counter; same point increments it (anti-repetition).
    if result.same_point_as_before:
        count = state.get("clarify_count", 0) + 1
    else:
        count = 1

    # Stuck on the same point too many times: stop rephrasing, show the
    # raw source text instead of looping on variations.
    if count > MAX_CLARIFY_REPEATS:
        context = state.get("context", "").strip()
        answer = (
            "Изглежда обяснението ми не помага. Ето точния текст от "
            f"документа, за да прецените сами:\n\n{context}"
            if context
            else "Изглежда не мога да поясня това по-добре. Опитайте да "
            "формулирате въпроса по друг начин или се свържете с поддръжка."
        )
    elif not result.confident:
        answer = (
            "Не съм сигурен как да поясня това по-точно, без риск от "
            "подвеждаща информация. Мога да покажа точния текст от "
            "документа или да опитам друг въпрос."
        )
    elif result.used_general_knowledge:
        answer = (
            f"{result.explanation}\n\n"
            "(Забележка: това пояснение е от обща култура, не от документа.)"
        )
    else:
        answer = result.explanation

    return {"generation": answer, "clarify_count": count}


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
        "contexts": result["contexts"],
        "clarify_count": 0,  # a fresh question resets the clarify streak
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
        "generation_retries": state.get("generation_retries", 0) + 1,
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
    return {"generation": msg}


def finalize(state: AgentState) -> AgentState:
    """Commit the accepted answer to the conversation history.

    Single place that appends to `messages`, so regeneration attempts do
    not pollute the history with intermediate answers.
    """
    return {"messages": [AIMessage(content=state.get("generation", ""))]}


class GradeAnswer(BaseModel):
    """Self-check on the generated answer (Self-RAG reflection)."""

    grounded: bool = Field(
        description="True if every factual claim in the answer is supported "
        "by the provided context (no invented facts)."
    )
    addresses_question: bool = Field(
        description="True if the answer actually addresses the question."
    )


_GRADE_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ти си строг проверител. Дадени са контекст, въпрос и генериран "
            "отговор. Провери: (1) grounded — всяко фактологично твърдение "
            "подкрепено ли е от контекста, без измислици; (2) "
            "addresses_question — отговорът адресира ли въпроса.",
        ),
        (
            "human",
            "Контекст:\n{context}\n\nВъпрос: {question}\n\n"
            "Отговор: {generation}",
        ),
    ]
)


def grade_answer(state: AgentState) -> AgentState:
    """Verify the generated answer is grounded in the context (Self-RAG)."""
    grader = get_llm().with_structured_output(GradeAnswer)
    verdict: GradeAnswer = (_GRADE_ANSWER_PROMPT | grader).invoke(
        {
            "context": state.get("context", ""),
            "question": state["question"],
            "generation": state.get("generation", ""),
        }
    )
    return {"answer_grounded": verdict.grounded and verdict.addresses_question}
