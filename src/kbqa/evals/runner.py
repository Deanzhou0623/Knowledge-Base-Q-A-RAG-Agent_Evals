import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from kbqa.config import Settings
from kbqa.embeddings import OpenAIEmbeddingProvider
from kbqa.evals.dataset import EvalCase, load_cases
from kbqa.evals.metrics import answer_metrics, retrieval_metrics
from kbqa.factory import build_service
from kbqa.io import atomic_write_text
from kbqa.llm import OpenAIAnswerGenerator
from kbqa.models import FALLBACK_ANSWER, ChatRequest, DocumentUnit, RetrievalResult
from kbqa.parsing import load_sections
from kbqa.prompts import CONTROL_PROMPT_VERSION, PROMPT_VERSION, build_grounded_input
from kbqa.service import extract_citations
from kbqa.transactions import TransactionStore


def _oracle_results(case: EvalCase, settings: Settings) -> list[RetrievalResult]:
    sections, _ = load_sections(settings.docs_path)
    by_citation: dict[str, DocumentUnit] = {unit.citation: unit for unit in sections}
    results: list[RetrievalResult] = []
    for citation in case.oracle_sources:
        unit = by_citation.get(citation)
        if unit is not None:
            results.append(
                RetrievalResult(
                    **unit.model_dump(), score=1.0, rank=len(results) + 1
                )
            )
    return results


def run_case(
    case: EvalCase,
    arm: str,
    settings: Settings,
    service: Any | None,
    generator: Any,
    transaction_store: TransactionStore,
) -> dict:
    base: dict[str, Any] = {
        "question_id": case.id,
        "category": case.category,
        "arm": arm,
        "answerable": case.answerable,
    }
    started = perf_counter()
    if arm == "llm_only":
        answer = generator.closed_book(case.question)
        base.update(
            answer=answer,
            citations=[],
            retrieved=[],
            retrieval_metrics=None,
            answer_metrics=answer_metrics(case, answer, []),
        )
    elif arm == "oracle":
        retrieved = _oracle_results(case, settings)
        structured = None
        if case.booking_id:
            structured = transaction_store.lookup(
                case.booking_id,
                as_of=case.as_of,
                expected_version=case.transaction_fixture_version,
            )
            if structured is not None:
                allowed_fields = {
                    reference.rsplit("#", 1)[1]
                    for reference in case.oracle_sources
                    if reference.startswith(f"booking:{case.booking_id}#")
                }
                structured = structured.model_copy(
                    update={
                        "fields": {
                            field: value
                            for field, value in structured.fields.items()
                            if field in allowed_fields
                        },
                        "references": [
                            reference
                            for reference in structured.references
                            if reference in case.oracle_sources
                        ],
                    }
                )
        answer = generator.grounded(
            build_grounded_input(case.question, retrieved, structured)
        ).strip()
        citations = extract_citations(answer)
        allowed = set(case.oracle_sources)
        if answer != FALLBACK_ANSWER and (
            not citations or any(citation not in allowed for citation in citations)
        ):
            answer = FALLBACK_ANSWER
            citations = []
        base.update(
            answer=answer,
            citations=citations,
            retrieved=[item.model_dump(mode="json") for item in retrieved],
            retrieval_metrics=None,
            answer_metrics=answer_metrics(case, answer, citations),
        )
    else:
        if service is None:
            raise RuntimeError(f"Missing service for evaluation arm {arm}")
        response = service.chat(
            ChatRequest(
                query=case.question,
                booking_id=case.booking_id,
                as_of=case.as_of,
                expected_fixture_version=case.transaction_fixture_version,
                requires_document_retrieval=case.requires_document_retrieval is not False,
            )
        )
        base.update(
            answer=response.answer,
            citations=response.citations,
            retrieved=[item.model_dump(mode="json") for item in response.retrieved],
            retrieval_metrics=(
                None
                if case.requires_document_retrieval is False
                else retrieval_metrics(case, response.retrieved)
            ),
            answer_metrics=answer_metrics(case, response.answer, response.citations),
        )
    base["total_ms"] = (perf_counter() - started) * 1000
    base["model"] = settings.openai_chat_model
    base["prompt_version"] = (
        CONTROL_PROMPT_VERSION if arm == "llm_only" else PROMPT_VERSION
    )
    base["generation_settings"] = {
        "max_output_tokens": settings.max_output_tokens,
        "deterministic_where_supported": True,
    }
    base["top_k"] = None if arm in {"llm_only", "oracle"} else settings.top_k
    base["rag_control_confound"] = arm == "llm_only"
    base["corpus_fingerprint"] = (
        getattr(service.retriever, "corpus_fingerprint", None) if service else None
    )
    return base


def run(dataset: Path, output: Path, arms: list[str]) -> list[dict]:
    cases = load_cases(dataset)
    settings = Settings()
    generator = OpenAIAnswerGenerator(
        settings.openai_chat_model,
        api_key=settings.openai_api_key,
        max_output_tokens=settings.max_output_tokens,
    )
    embeddings = OpenAIEmbeddingProvider(
        settings.openai_embedding_model, api_key=settings.openai_api_key
    )
    services: dict[str, Any] = {}
    for arm in arms:
        if arm not in {"llm_only", "bm25", "vector", "oracle"}:
            raise ValueError(f"Unsupported scaffold arm: {arm}")
        if arm in {"bm25", "vector"}:
            arm_settings = settings.model_copy(update={"retrieval_backend": arm})
            service = build_service(
                arm_settings, generator=generator, embeddings=embeddings
            )
            service.build_index()
            services[arm] = service

    transaction_store = TransactionStore(settings.transaction_fixture_path)
    records = [
        run_case(
            case, arm, settings, services.get(arm), generator, transaction_store
        )
        for case in cases
        for arm in arms
        if arm != "oracle" or case.answerable
    ]
    serialized = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    atomic_write_text(output, serialized)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation scaffold")
    parser.add_argument("--dataset", type=Path, default=Path("evals/cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/eval.jsonl"))
    parser.add_argument("--arms", nargs="+", default=["llm_only", "bm25", "vector"])
    args = parser.parse_args()
    records = run(args.dataset, args.output, args.arms)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
