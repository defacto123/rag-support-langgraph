"""Evaluate the RAG agent with Ragas against a golden dataset.

Metrics (all judged by the Gemini LLM / embeddings, not OpenAI):
  - faithfulness            : is the answer grounded in retrieved context?
  - answer_relevancy        : does the answer address the question?
  - context_precision       : are the retrieved chunks relevant (ranking)?
  - context_recall          : did retrieval capture the reference answer?

Run with:
  python -m app.eval.run_eval
"""

# --- Compatibility shim -----------------------------------------------------
# Ragas 0.4.3 imports Vertex AI paths that were removed from newer
# langchain-community. We never use Vertex AI (we use Gemini via AI Studio),
# so we stub the missing symbols BEFORE importing ragas.
import sys
import types

_vx = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAIStub:  # pragma: no cover - never instantiated
    pass


_vx.ChatVertexAI = _ChatVertexAIStub
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vx)

import langchain_community.llms as _llms  # noqa: E402

if not hasattr(_llms, "VertexAI"):
    class _VertexAIStub:  # pragma: no cover
        pass

    _llms.VertexAI = _VertexAIStub
# ---------------------------------------------------------------------------

import json  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.agent.graph import ask  # noqa: E402
from app.ingestion.pipeline import ingest_document  # noqa: E402
from app.models import get_embeddings, get_llm  # noqa: E402

_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
_REPORT_PATH = Path("docs/eval_report.md")
_SAMPLE_DOCS = [
    "data/demo/company_policies.md",
    "data/demo/warranty_and_support.md",
    "data/demo/service_agreement.md",
    "data/demo/faq.md",
]


def _ensure_documents_ingested() -> None:
    """Ingest the sample documents so retrieval has something to find."""
    for doc in _SAMPLE_DOCS:
        if Path(doc).exists():
            ingest_document(doc)


def _build_dataset() -> EvaluationDataset:
    golden = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
    samples = []
    for item in golden:
        # Fresh thread_id per question -> no memory leakage between cases.
        result = ask(item["question"], thread_id=str(uuid.uuid4()))
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"] or [""],
                reference=item["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


def _format_report(result) -> str:
    df = result.to_pandas()
    lines = ["# Ragas Evaluation Report", ""]

    # Aggregate scores (mean per metric).
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    lines.append("## Средни резултати\n")
    lines.append("| Метрика | Средно |")
    lines.append("|---|---|")
    for col in numeric_cols:
        lines.append(f"| {col} | {df[col].mean():.3f} |")
    lines.append("")

    # Per-question breakdown.
    lines.append("## По въпроси\n")
    header = ["#", "въпрос"] + numeric_cols
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i, row in df.iterrows():
        q = str(row.get("user_input", ""))[:50]
        vals = [f"{row[c]:.3f}" for c in numeric_cols]
        lines.append(f"| {i + 1} | {q} | " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("1/4 Индексирам примерните документи…")
    _ensure_documents_ingested()

    print("2/4 Пускам агента по въпросите от golden dataset…")
    dataset = _build_dataset()

    print("3/4 Оценявам с Ragas (Gemini като съдия)…")
    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    print("\n=== Резултати ===")
    print(result)

    report = _format_report(result)
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n4/4 Отчетът е записан в {_REPORT_PATH}")


if __name__ == "__main__":
    main()
