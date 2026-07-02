"""Retrieval: turn a question into the most relevant chunks.

Uses the SAME embedding model as ingestion, so questions and chunks live
in the same vector space and are comparable.
"""

from typing import Union

from langchain_core.documents import Document

from app.ingestion.vectorstore import get_vectorstore

# A result may be a plain Document (MMR) or (Document, score) (similarity).
Result = Union[Document, tuple[Document, float]]


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


def format_context(results: list[Result]) -> tuple[str, list[dict]]:
    """Turn retrieval results into (context, sources).

    context: numbered blocks fed to the LLM so it can cite "Източник N".
    sources: structured list shown to the user under the answer.

    Accepts either plain Documents (MMR) or (Document, score) tuples
    (similarity), so it works with any search function above.
    """
    context_parts: list[str] = []
    sources: list[dict] = []

    for i, result in enumerate(results, start=1):
        # Normalise: split into document + optional score.
        if isinstance(result, tuple):
            doc, score = result
        else:
            doc, score = result, None

        context_parts.append(f"[Източник {i}]\n{doc.page_content}")

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
    return context, sources


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
    if use_mmr:
        results: list[Result] = search_mmr(query, k=k)
    else:
        scored = search_similarity(query, k=k)
        # Keep only matches at/above the threshold (score = similarity).
        results = [(doc, s) for doc, s in scored if s >= score_threshold]

    context, sources = format_context(results)

    return {
        "query": query,
        "context": context,
        "sources": sources,
        "found": len(sources) > 0,
    }
