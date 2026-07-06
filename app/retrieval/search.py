"""Retrieval: turn a question into the most relevant chunks.

Uses the SAME embedding model as ingestion, so questions and chunks live
in the same vector space and are comparable.
"""

from typing import Union

from langchain_core.documents import Document

from app.ingestion.vectorstore import get_vectorstore

# A result may be a plain Document (MMR) or (Document, score) (similarity).
Result = Union[Document, tuple[Document, float]]


def _doc(result: Result) -> Document:
    """Extract the Document from a plain-doc or (doc, score) result."""
    return result[0] if isinstance(result, tuple) else result


def _script(text: str) -> str:
    """Rough script detector: 'cyrillic' vs 'latin'.

    The embedding model is multilingual, so a Bulgarian question can still
    pull English chunks. This lets us tell the two apart cheaply and
    deterministically (no extra dependency), which is enough to separate
    Bulgarian (Cyrillic) from English (Latin) — the bilingual case both
    knowledge bases use.
    """
    cyrillic = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    total = cyrillic + latin
    if total == 0:
        return "latin"
    # English text has no Cyrillic, while Bulgarian text keeps Latin product
    # names (MobiOffice, Windows, ...). So any meaningful Cyrillic presence
    # means Bulgarian, even when Latin characters are the majority.
    return "cyrillic" if cyrillic / total >= 0.2 else "latin"


def _prefer_question_language(
    query: str, results: list[Result], k: int
) -> list[Result]:
    """Reorder results so chunks in the question's language come first.

    Keeps relevance order within each language group, then trims to k.
    If no chunk matches the question's language, the others are used as-is
    (the generation step is instructed to translate them).
    """
    q_script = _script(query)
    same = [r for r in results if _script(_doc(r).page_content) == q_script]
    other = [r for r in results if _script(_doc(r).page_content) != q_script]
    return (same + other)[:k]


def search_similarity(
    query: str,
    k: int = 4,
) -> list[tuple[Document, float]]:
    """Return the k nearest chunks with their score.

    Each item is (Document, score). What the score means (similarity vs
    distance) depends on the backend, so we verify it experimentally
    before relying on thresholds.
    """
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query=query, k=k)


def search_mmr(
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> list[Document]:
    """Return k chunks balancing relevance and diversity (MMR).

    Fetches `fetch_k` candidates first, then picks the `k` that are most
    relevant AND most different from each other. Avoids returning several
    near-duplicate chunks and missing other useful context.

    lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity, 0.5 = balance.
    MMR does not expose a score, so this returns plain Documents.
    """
    vectorstore = get_vectorstore()
    return vectorstore.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )


def format_context(results: list[Result]) -> tuple[str, list[dict], list[str]]:
    """Turn retrieval results into (context, sources, contexts).

    context:  numbered blocks fed to the LLM so it can cite "Източник N".
    sources:  structured list shown to the user under the answer.
    contexts: the full text of each retrieved chunk, one per item, used by
              evaluation (Ragas) which needs per-chunk retrieved contexts.

    Accepts either plain Documents (MMR) or (Document, score) tuples
    (similarity), so it works with any search function above.
    """
    context_parts: list[str] = []
    sources: list[dict] = []
    contexts: list[str] = []

    for i, result in enumerate(results, start=1):
        # Normalise: split into document + optional score.
        if isinstance(result, tuple):
            doc, score = result
        else:
            doc, score = result, None

        context_parts.append(f"[Източник {i}]\n{doc.page_content}")
        contexts.append(doc.page_content)

        sources.append(
            {
                "index": i,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "score": round(score, 3) if score is not None else None,
                "preview": doc.page_content[:100],
            }
        )

    context = "\n\n---\n\n".join(context_parts)
    return context, sources, contexts


def search(
    query: str,
    k: int = 4,
    use_mmr: bool = True,
    score_threshold: float = 0.7,
) -> dict:
    """High-level retrieval entry point used by the agent.

    - use_mmr=True  -> diverse results (no score filtering).
    - use_mmr=False -> similarity results filtered by score_threshold,
      which drops weak matches so the LLM is not fed irrelevant context.

    Returns {query, context, sources, found}.
    """
    # Over-fetch a larger candidate pool so we can prefer chunks written in
    # the question's language before trimming down to k.
    pool = max(k * 3, 12)
    if use_mmr:
        candidates: list[Result] = search_mmr(query, k=pool, fetch_k=pool * 3)
    else:
        scored = search_similarity(query, k=pool)
        # Keep only matches at/above the threshold (score = similarity).
        candidates = [(doc, s) for doc, s in scored if s >= score_threshold]

    results = _prefer_question_language(query, candidates, k)
    context, sources, contexts = format_context(results)

    return {
        "query": query,
        "context": context,
        "sources": sources,
        "contexts": contexts,
        "found": len(sources) > 0,
    }
