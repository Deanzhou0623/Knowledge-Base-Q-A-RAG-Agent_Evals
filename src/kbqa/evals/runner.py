import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from kbqa.config import Settings
from kbqa.embeddings import OpenAIEmbeddingProvider
from kbqa.evals.dataset import EvalCase, EvaluationDataset, validate_dataset
from kbqa.evals.metrics import (
    GRADER_VERSION,
    answer_metrics,
    paraphrase_robustness,
    retrieval_metrics,
)
from kbqa.evals.reporting import build_summary, render_markdown_report
from kbqa.factory import build_service
from kbqa.io import atomic_write_json, atomic_write_text
from kbqa.llm import OpenAIAnswerGenerator
from kbqa.models import FALLBACK_ANSWER, ChatRequest, RetrievalResult, TokenUsage
from kbqa.parsing import load_sections
from kbqa.prompts import CONTROL_PROMPT_VERSION, PROMPT_VERSION, build_grounded_input
from kbqa.service import extract_citations
from kbqa.transactions import TransactionStore


ARMS = ("llm_only", "bm25", "vector", "oracle")
MAIN_ARMS = {"llm_only", "bm25", "vector"}
RUNNER_VERSION = "evaluation-runner-v1"
CONTROL_CONFOUND = (
    "RAG-minus-control compares a strict grounded prompt plus retrieved context "
    "against a closed-book prompt; it is not a context-only ablation."
)


class ControlledInputError(ValueError):
    pass


def _usage(provider: Any) -> TokenUsage:
    value = getattr(provider, "token_usage", None)
    return value if isinstance(value, TokenUsage) else TokenUsage()


def _usage_delta(after: TokenUsage, before: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=max(0, after.input_tokens - before.input_tokens),
        output_tokens=max(0, after.output_tokens - before.output_tokens),
        total_tokens=max(0, after.total_tokens - before.total_tokens),
    )


def _answer_cost(usage: TokenUsage, settings: Settings) -> float:
    return (
        usage.input_tokens * settings.answer_input_usd_per_million_tokens
        + usage.output_tokens * settings.answer_output_usd_per_million_tokens
    ) / 1_000_000


def _embedding_cost(usage: TokenUsage, settings: Settings) -> float:
    return (
        usage.input_tokens * settings.embedding_usd_per_million_tokens / 1_000_000
    )


def _index_configuration(service: Any, settings: Settings) -> dict[str, Any]:
    retriever = service.retriever
    if retriever.backend == "bm25":
        return {
            "backend": "bm25",
            "schema_version": retriever.schema_version,
            "tokenizer": "unicode_word_lowercase-v1",
            "index_path": str(retriever.index_path),
        }
    dimension = getattr(getattr(retriever, "_index", None), "d", None)
    return {
        "backend": "vector",
        "schema_version": retriever.schema_version,
        "embedding_model": retriever.embeddings.model,
        "dimension": dimension,
        "normalization": "l2",
        "distance_metric": "inner_product",
        "index_type": "IndexFlatIP",
        "chunk_words": retriever.chunk_words,
        "overlap": retriever.overlap,
        "index_path": str(retriever.index_path),
    }


def _oracle_context(
    case: EvalCase,
    settings: Settings,
    transaction_store: TransactionStore,
) -> tuple[list[RetrievalResult], Any]:
    sections, _ = load_sections(settings.docs_path)
    by_citation = {unit.citation: unit for unit in sections}
    gold = {source.citation: source for source in case.acceptable_sources}
    results: list[RetrievalResult] = []
    for citation in case.oracle_sources:
        unit = by_citation.get(citation)
        if unit is None:
            continue
        source = gold[citation]
        evidence = source.evidence_text
        if evidence is None and source.start_offset is not None:
            assert source.end_offset is not None
            evidence = unit.text[source.start_offset : source.end_offset]
        results.append(
            RetrievalResult(
                **unit.model_dump(exclude={"text"}),
                text=evidence or "",
                score=1.0,
                rank=len(results) + 1,
            )
        )

    structured = None
    if case.booking_id:
        structured = transaction_store.lookup(
            case.booking_id,
            as_of=case.as_of,
            expected_version=case.transaction_fixture_version,
        )
        if structured is not None:
            allowed_references = {
                citation
                for citation in case.oracle_sources
                if citation.startswith(f"booking:{case.booking_id}#")
            }
            allowed_fields = {
                reference.rsplit("#", 1)[1] for reference in allowed_references
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
                        if reference in allowed_references
                    ],
                }
            )
    return results, structured


def _empty_record(
    *,
    case: EvalCase,
    arm: str,
    trial: int,
    run_id: str,
    timestamp: str,
    dataset: EvaluationDataset,
    settings: Settings,
    concurrency: int,
    retries: int,
    cold_start: bool,
    answer_model: str,
    embedding_model: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "runner_version": RUNNER_VERSION,
        "dataset_version": dataset.manifest.dataset_version,
        "question_id": case.id,
        "question": case.question,
        "category": case.category,
        "requires_document_retrieval": case.requires_document_retrieval,
        "paraphrase_group_id": case.paraphrase_group_id,
        "trial": trial,
        "trials_per_question": dataset.manifest.trials_per_question,
        "timestamp": timestamp,
        "arm": arm,
        "status": "pending",
        "error": None,
        "attempts": 0,
        "cold_start": cold_start,
        "execution_state": "cold_index" if cold_start else "warm_index",
        "concurrency": concurrency,
        "configured_retries": retries,
        "answerable": case.answerable,
        "expected_facts": case.expected_facts,
        "expected_sources": [
            source.model_dump(mode="json") for source in case.acceptable_sources
        ],
        "oracle_sources": case.oracle_sources,
        "answer": None,
        "citations": None,
        "raw_answer": None,
        "raw_citations": None,
        "citation_guardrail_triggered": None,
        "retrieved": None,
        "structured_context": None,
        "retrieval_metrics": None,
        "answer_metrics": None,
        "failure_labels": [],
        "index_ms": None,
        "index_embedding_token_usage": None,
        "estimated_index_embedding_cost_usd": None,
        "retrieval_ms": None,
        "generation_ms": None,
        "total_ms": None,
        "answer_token_usage": None,
        "embedding_token_usage": None,
        "answer_input_tokens": None,
        "answer_output_tokens": None,
        "embedding_input_tokens": None,
        "estimated_answer_cost_usd": None,
        "estimated_embedding_cost_usd": None,
        "pricing": {
            "answer_input_usd_per_million_tokens": (
                settings.answer_input_usd_per_million_tokens
            ),
            "answer_output_usd_per_million_tokens": (
                settings.answer_output_usd_per_million_tokens
            ),
            "embedding_usd_per_million_tokens": (
                settings.embedding_usd_per_million_tokens
            ),
        },
        # Reported by the objects that ran, not by configuration, so a
        # record can never claim a model that was never invoked.
        "model": answer_model,
        "embedding_model": embedding_model if arm == "vector" else None,
        "prompt_version": (
            CONTROL_PROMPT_VERSION if arm == "llm_only" else PROMPT_VERSION
        ),
        "generation_settings": {
            "max_output_tokens": settings.max_output_tokens,
            "temperature": None,
            "deterministic_where_supported": True,
        },
        "grader_version": dataset.manifest.grader_version,
        "top_k": 3 if arm in {"bm25", "vector"} else None,
        "corpus_fingerprint": dataset.manifest.corpus_fingerprint,
        "index_configuration": None,
        "index_error": None,
        "transaction_fixture_version": (
            case.transaction_fixture_version
            or dataset.manifest.transaction_fixture_version
        ),
        "as_of": case.as_of.isoformat() if case.as_of else None,
        "rag_control_confound": CONTROL_CONFOUND,
    }


def run_case(
    case: EvalCase,
    arm: str,
    settings: Settings,
    service: Any | None,
    generator: Any,
    transaction_store: TransactionStore,
    *,
    dataset: EvaluationDataset,
    trial: int,
    run_id: str,
    index_metadata: dict[str, dict[str, Any]],
    embeddings: Any,
    cold_start: bool = False,
    concurrency: int,
    retries: int,
) -> dict[str, Any]:
    record = _empty_record(
        case=case,
        arm=arm,
        trial=trial,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
        settings=settings,
        concurrency=concurrency,
        retries=retries,
        cold_start=cold_start,
        answer_model=generator.model,
        embedding_model=getattr(embeddings, "model", None),
    )
    if arm in index_metadata:
        record.update(index_metadata[arm])
    started = perf_counter()
    for attempt in range(1, retries + 2):
        record["attempts"] = attempt
        try:
            if arm == "llm_only":
                generation_started = perf_counter()
                generation = generator.closed_book(case.question)
                record["generation_ms"] = (
                    perf_counter() - generation_started
                ) * 1000
                answer = generation.text.strip()
                citations = extract_citations(answer)
                supplied_citations: set[str] = set()
                retrieved = None
                structured = None
                raw_answer = answer
                raw_citations = list(citations)
                guardrail_triggered = False
            elif arm == "oracle":
                retrieved, structured = _oracle_context(
                    case, settings, transaction_store
                )
                prompt_input = build_grounded_input(
                    case.question, retrieved, structured
                )
                generation_started = perf_counter()
                generation = generator.grounded(prompt_input)
                record["generation_ms"] = (
                    perf_counter() - generation_started
                ) * 1000
                answer = generation.text.strip()
                citations = extract_citations(answer)
                supplied_citations = set(case.oracle_sources)
                raw_answer = answer
                raw_citations = list(citations)
                guardrail_triggered = False
                if answer != FALLBACK_ANSWER and (
                    not citations
                    or any(citation not in supplied_citations for citation in citations)
                ):
                    answer = FALLBACK_ANSWER
                    citations = []
                    guardrail_triggered = True
            else:
                if service is None:
                    raise RuntimeError(
                        record.get("index_error")
                        or f"Missing service for evaluation arm {arm}"
                    )
                embedding_before = _usage(embeddings)
                response = service.chat(
                    ChatRequest(
                        query=case.question,
                        booking_id=case.booking_id,
                        as_of=case.as_of,
                        expected_fixture_version=case.transaction_fixture_version,
                        requires_document_retrieval=(
                            case.requires_document_retrieval is not False
                        ),
                    )
                )
                embedding_after = _usage(embeddings)
                query_embedding_usage = _usage_delta(
                    embedding_after, embedding_before
                )
                record["embedding_token_usage"] = (
                    query_embedding_usage.model_dump()
                    if arm == "vector"
                    else None
                )
                record["embedding_input_tokens"] = (
                    query_embedding_usage.input_tokens
                    if arm == "vector"
                    else None
                )
                record["estimated_embedding_cost_usd"] = (
                    _embedding_cost(query_embedding_usage, settings)
                    if arm == "vector"
                    else None
                )
                answer = response.answer
                citations = response.citations
                retrieved = response.retrieved
                structured = response.structured_context
                supplied_citations = set(response.sources)
                record["retrieval_ms"] = response.retrieval_ms
                record["generation_ms"] = response.generation_ms
                raw_answer = response.raw_answer
                raw_citations = list(response.raw_citations)
                guardrail_triggered = response.citation_guardrail_triggered
                generation = type("Generation", (), {"token_usage": response.token_usage})()

            record["answer"] = answer
            record["citations"] = citations
            record["raw_answer"] = raw_answer
            record["raw_citations"] = raw_citations
            record["citation_guardrail_triggered"] = guardrail_triggered
            record["retrieved"] = (
                None
                if retrieved is None
                else [item.model_dump(mode="json") for item in retrieved]
            )
            record["structured_context"] = (
                structured.model_dump(mode="json") if structured is not None else None
            )
            record["retrieval_metrics"] = (
                retrieval_metrics(case, retrieved)
                if arm in {"bm25", "vector"} and retrieved is not None
                else None
            )
            record["answer_metrics"] = answer_metrics(
                case,
                answer,
                citations,
                arm=arm,
                supplied_citations=supplied_citations,
                raw_citations=raw_citations,
            )
            usage = generation.token_usage
            record["answer_token_usage"] = usage.model_dump()
            record["answer_input_tokens"] = usage.input_tokens
            record["answer_output_tokens"] = usage.output_tokens
            record["estimated_answer_cost_usd"] = _answer_cost(usage, settings)
            if (
                case.expected_failure_label
                and record["retrieval_metrics"]
                and record["retrieval_metrics"].get("hit_rate") is False
            ):
                record["failure_labels"].append(case.expected_failure_label)
            if (
                record["retrieval_metrics"]
                and record["retrieval_metrics"].get("unsupported_retrieval") is True
            ):
                record["failure_labels"].append(
                    "keyword_false_positive"
                    if arm == "bm25"
                    else "semantic_false_positive"
                )
            record["status"] = "ok"
            break
        except Exception as exc:  # preserve failed cases in the output contract
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["status"] = "error"
            if attempt > retries:
                break
    record["total_ms"] = (perf_counter() - started) * 1000
    return record


def _diagnose(records: list[dict[str, Any]]) -> None:
    by_case_trial: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case_trial.setdefault(
            (record["question_id"], record["trial"]), {}
        )[record["arm"]] = record
    for arms in by_case_trial.values():
        oracle = arms.get("oracle")
        for arm in ("bm25", "vector"):
            record = arms.get(arm)
            if not record or record["status"] != "ok":
                continue
            answer_score = record["answer_metrics"].get("correctness")
            retrieval_score = (
                record["retrieval_metrics"].get("recall_at_3")
                if record.get("retrieval_metrics")
                else None
            )
            oracle_score = (
                oracle["answer_metrics"].get("correctness")
                if oracle and oracle["status"] == "ok"
                else None
            )
            if oracle_score == 1.0 and answer_score != 1.0:
                label = (
                    "likely_retrieval_failure"
                    if retrieval_score is not None and retrieval_score < 1.0
                    else "likely_generation_or_prompt_failure"
                )
                record["failure_labels"].append(label)
            elif oracle_score is not None and oracle_score != 1.0:
                record["failure_labels"].append(
                    "oracle_annotation_model_prompt_review"
                )


def run(
    dataset: Path,
    output: Path,
    arms: list[str],
    *,
    manifest: Path | None = None,
    report: Path | None = None,
    summary_output: Path | None = None,
    trials: int | None = None,
    settings: Settings | None = None,
    generator: Any | None = None,
    embeddings: Any | None = None,
    allow_live: bool = False,
    require_main_arms: bool = True,
    concurrency: int = 1,
    retries: int = 0,
) -> list[dict[str, Any]]:
    selected = list(dict.fromkeys(arms))
    unknown = set(selected) - set(ARMS)
    if unknown:
        raise ValueError(f"Unsupported evaluation arm(s): {', '.join(sorted(unknown))}")
    if require_main_arms and not MAIN_ARMS.issubset(selected):
        raise ValueError("A controlled main run requires llm_only, bm25, and vector")
    if concurrency != 1:
        raise ValueError(
            "This runner currently requires concurrency=1 for interpretable latency"
        )
    if retries < 0:
        raise ValueError("retries cannot be negative")

    active_settings = settings or Settings()
    manifest_path = manifest or dataset.with_name("manifest.json")
    loaded = validate_dataset(
        dataset,
        manifest_path,
        docs_path=active_settings.docs_path,
        transaction_fixture_path=active_settings.transaction_fixture_path,
    )
    trial_count = trials or loaded.manifest.trials_per_question
    if trials is not None and trials != loaded.manifest.trials_per_question:
        raise ValueError("CLI trial count must match the frozen dataset manifest")
    if loaded.manifest.grader_version != GRADER_VERSION:
        raise ValueError(
            "Dataset grader_version does not match the implemented grader: "
            f"{loaded.manifest.grader_version!r} != {GRADER_VERSION!r}"
        )

    if generator is None or embeddings is None:
        if not allow_live:
            raise RuntimeError(
                "Live provider calls are disabled. Supply fakes in code or pass --live."
            )
        if not active_settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --live evaluation")
    answer_generator = generator or OpenAIAnswerGenerator(
        active_settings.openai_chat_model,
        api_key=active_settings.openai_api_key,
        max_output_tokens=active_settings.max_output_tokens,
    )
    embedding_provider = embeddings or OpenAIEmbeddingProvider(
        active_settings.openai_embedding_model,
        api_key=active_settings.openai_api_key,
    )

    # Controlled inputs must be asserted, not assumed. Records now carry the
    # model reported by the objects that actually ran, so an injected fake
    # can never be recorded as gpt-5.6-sol. A live run must additionally
    # match the configured models exactly.
    if allow_live and answer_generator.model != active_settings.openai_chat_model:
        raise ValueError(
            "Answer generator model does not match the configured answer model: "
            f"{answer_generator.model!r} != {active_settings.openai_chat_model!r}"
        )
    if allow_live and embedding_provider.model != active_settings.openai_embedding_model:
        raise ValueError(
            "Embedding provider model does not match the configured embedding "
            f"model: {embedding_provider.model!r} != "
            f"{active_settings.openai_embedding_model!r}"
        )

    services: dict[str, Any] = {}
    index_metadata: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for arm in ("bm25", "vector"):
        if arm not in selected:
            continue
        arm_settings = active_settings.model_copy(update={"retrieval_backend": arm})
        usage_before = _usage(embedding_provider)
        build_started = perf_counter()
        try:
            service = build_service(
                arm_settings,
                generator=answer_generator,
                embeddings=embedding_provider,
            )
            summary = service.build_index()
            index_ms = (perf_counter() - build_started) * 1000
            index_embedding_usage = _usage_delta(
                _usage(embedding_provider), usage_before
            )
            if summary.corpus_fingerprint != loaded.manifest.corpus_fingerprint:
                raise ControlledInputError(
                    f"{arm} index corpus fingerprint differs from frozen dataset"
                )
            fingerprints.add(summary.corpus_fingerprint)
            services[arm] = service
            index_metadata[arm] = {
                "index_ms": index_ms,
                "index_configuration": _index_configuration(
                    service, active_settings
                ),
                "index_embedding_token_usage": (
                    index_embedding_usage.model_dump()
                    if arm == "vector"
                    else None
                ),
                "estimated_index_embedding_cost_usd": (
                    _embedding_cost(index_embedding_usage, active_settings)
                    if arm == "vector"
                    else None
                ),
                "index_error": None,
            }
        except ControlledInputError:
            # Controlled-input violations invalidate the entire experiment.
            raise
        except Exception as exc:
            index_metadata[arm] = {
                "index_ms": (perf_counter() - build_started) * 1000,
                "index_configuration": None,
                "index_embedding_token_usage": None,
                "estimated_index_embedding_cost_usd": None,
                "index_error": f"{type(exc).__name__}: {exc}",
            }
    if len(fingerprints) > 1:
        raise ValueError("BM25 and Vector indexes do not share a corpus fingerprint")
    built = [service for service in services.values() if service is not None]
    if len({service.top_k for service in built}) > 1:
        raise ValueError("RAG arms do not share the same K")
    for service in built:
        if service.top_k != active_settings.top_k:
            raise ValueError(
                f"Service K {service.top_k} does not match the configured "
                f"K {active_settings.top_k}"
            )
        if service.generator.model != answer_generator.model:
            raise ValueError("A RAG service is using a different answer model")

    transaction_store = TransactionStore(
        active_settings.transaction_fixture_path
    )
    run_id = str(uuid.uuid4())
    records = [
        run_case(
            case,
            arm,
            active_settings,
            services.get(arm),
            answer_generator,
            transaction_store,
            dataset=loaded,
            trial=trial,
            run_id=run_id,
            index_metadata=index_metadata,
            embeddings=embedding_provider,
            concurrency=concurrency,
            retries=retries,
            cold_start=(trial == 1 and index == 0),
        )
        for trial in range(1, trial_count + 1)
        for index, case in enumerate(loaded.cases)
        for arm in selected
        if arm != "oracle" or case.answerable
    ]
    _diagnose(records)
    serialized = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    atomic_write_text(output, serialized)

    summary = build_summary(
        records,
        dataset=loaded,
        arms=selected,
        paraphrase=paraphrase_robustness(records),
    )
    summary_path = summary_output or output.with_suffix(".summary.json")
    report_path = report or output.with_suffix(".report.md")
    atomic_write_json(summary_path, summary)
    atomic_write_text(report_path, render_markdown_report(summary))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the controlled RAG evaluation"
    )
    parser.add_argument("--dataset", type=Path, default=Path("evals/cases.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("evals/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/eval.jsonl"))
    parser.add_argument(
        "--arms", nargs="+", choices=ARMS, default=["llm_only", "bm25", "vector"]
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly allow OpenAI answer and embedding API calls",
    )
    args = parser.parse_args()
    records = run(
        args.dataset,
        args.output,
        args.arms,
        manifest=args.manifest,
        trials=args.trials,
        retries=args.retries,
        allow_live=args.live,
    )
    failed = sum(record["status"] != "ok" for record in records)
    print(f"Wrote {len(records)} records to {args.output} ({failed} failed)")


if __name__ == "__main__":
    main()
