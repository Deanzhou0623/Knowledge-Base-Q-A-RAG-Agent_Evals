from collections import defaultdict
from typing import Any

from kbqa.evals.dataset import CATEGORIES, EvaluationDataset
from kbqa.evals.metrics import numeric_summary


CELL_METRICS = (
    ("answer_metrics", "correctness"),
    ("answer_metrics", "citation_accuracy"),
    ("answer_metrics", "unsupported_claim_rate"),
    ("answer_metrics", "fallback_correct"),
    ("answer_metrics", "refusal_calibrated"),
    ("retrieval_metrics", "recall_at_3"),
    ("retrieval_metrics", "section_recall_at_3"),
    ("retrieval_metrics", "hit_rate"),
    ("retrieval_metrics", "first_relevant_rank"),
    ("retrieval_metrics", "unsupported_retrieval"),
    (None, "retrieval_ms"),
    (None, "generation_ms"),
    (None, "answer_input_tokens"),
    (None, "answer_output_tokens"),
    (None, "embedding_input_tokens"),
    (None, "estimated_answer_cost_usd"),
    (None, "estimated_embedding_cost_usd"),
)


def _value(record: dict[str, Any], parent: str | None, key: str) -> Any:
    if parent is None:
        return record.get(key)
    nested = record.get(parent)
    return nested.get(key) if nested else None


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for parent, key in CELL_METRICS:
        values = [
            float(value)
            for record in records
            if record["status"] == "ok"
            and (value := _value(record, parent, key)) is not None
        ]
        metrics[key] = numeric_summary(values)
    return {
        "records": len(records),
        "successful": sum(record["status"] == "ok" for record in records),
        "failed": sum(record["status"] != "ok" for record in records),
        "metrics": metrics,
    }


def build_summary(
    records: list[dict[str, Any]],
    *,
    dataset: EvaluationDataset,
    arms: list[str],
    paraphrase: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for arm in arms:
        for category in CATEGORIES:
            cell_records = [
                record
                for record in records
                if record["arm"] == arm and record["category"] == category
            ]
            if cell_records:
                cells[f"{arm}:{category}"] = _aggregate(cell_records)

    improvement: dict[str, Any] = {}
    for category in ("company_specific", "generic_ecommerce"):
        control = cells.get(f"llm_only:{category}")
        if not control:
            continue
        for arm in ("bm25", "vector"):
            treatment = cells.get(f"{arm}:{category}")
            if not treatment:
                continue
            for metric in (
                "correctness",
                "unsupported_claim_rate",
                "fallback_correct",
            ):
                baseline = control["metrics"][metric]["mean"]
                result = treatment["metrics"][metric]["mean"]
                improvement[f"{arm}:{category}:{metric}"] = (
                    result - baseline
                    if result is not None and baseline is not None
                    else None
                )

    user_specific = {
        subtype: _aggregate(
            [
                record
                for record in records
                if record["category"] == "user_specific"
                and record["arm"] in {"bm25", "vector"}
                and bool(record["requires_document_retrieval"]) is requires_docs
            ]
        )
        for subtype, requires_docs in (
            ("transaction_only", False),
            ("transaction_plus_document", True),
        )
    }
    failures = [
        {
            "question_id": record["question_id"],
            "arm": record["arm"],
            "status": record["status"],
            "error": record["error"],
            "failure_labels": record["failure_labels"],
        }
        for record in records
        if record["status"] != "ok" or record["failure_labels"]
    ]
    first = records[0] if records else {}
    index_runs = {}
    for arm in ("bm25", "vector"):
        record = next((item for item in records if item["arm"] == arm), None)
        if record is not None:
            index_runs[arm] = {
                "index_ms": record["index_ms"],
                "configuration": record["index_configuration"],
                "embedding_token_usage": record[
                    "index_embedding_token_usage"
                ],
                "estimated_embedding_cost_usd": record[
                    "estimated_index_embedding_cost_usd"
                ],
                "error": record["index_error"],
            }
    return {
        "run_id": first.get("run_id"),
        "dataset_version": dataset.manifest.dataset_version,
        "corpus_tier": dataset.manifest.corpus_tier,
        "corpus_fingerprint": dataset.manifest.corpus_fingerprint,
        "trials_per_question": dataset.manifest.trials_per_question,
        "estimate_type": dataset.manifest.estimate_type,
        "minimum_questions_per_category": (
            dataset.manifest.minimum_questions_per_category
        ),
        "rag_control_confound": first.get("rag_control_confound"),
        "arm_category_cells": cells,
        "improvement_over_llm_only": improvement,
        "improvement_excluded_categories": ["unsupported", "user_specific"],
        "user_specific_rag_capability": user_specific,
        "paraphrase_robustness": paraphrase,
        "index_runs": index_runs,
        "failure_examples": failures,
    }


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Controlled RAG Evaluation Report",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Dataset: `{summary['dataset_version']}`",
        f"- Corpus tier: `{summary['corpus_tier']}`",
        f"- Trials per question: {summary['trials_per_question']} "
        f"({summary['estimate_type']})",
        f"- Minimum questions per category: "
        f"{summary['minimum_questions_per_category']}",
        "",
        "> Experimental limitation: " + summary["rag_control_confound"],
        "",
        "## Arm-by-category results",
        "",
        "| Arm / category | Success | Correctness | Recall@3 | Citation accuracy | "
        "Fallback | Retrieval ms | Generation ms | Answer cost | Embedding cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, cell in summary["arm_category_cells"].items():
        metrics = cell["metrics"]
        lines.append(
            f"| {name} | {cell['successful']}/{cell['records']} | "
            f"{_display(metrics['correctness']['mean'])} | "
            f"{_display(metrics['recall_at_3']['mean'])} | "
            f"{_display(metrics['citation_accuracy']['mean'])} | "
            f"{_display(metrics['fallback_correct']['mean'])} | "
            f"{_display(metrics['retrieval_ms']['mean'])} | "
            f"{_display(metrics['generation_ms']['mean'])} | "
            f"{_display(metrics['estimated_answer_cost_usd']['mean'])} | "
            f"{_display(metrics['estimated_embedding_cost_usd']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Index runs",
            "",
            "| Backend | Index ms | Embedding tokens | Embedding cost (USD) | Error |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for arm, run in summary["index_runs"].items():
        usage = run["embedding_token_usage"] or {}
        lines.append(
            f"| {arm} | {_display(run['index_ms'])} | "
            f"{_display(usage.get('input_tokens'))} | "
            f"{_display(run['estimated_embedding_cost_usd'])} | "
            f"{run['error'] or ''} |"
        )

    lines.extend(
        [
            "",
            "## Improvement over LLM-only",
            "",
            "Improvement is intentionally omitted for `unsupported` and "
            "`user_specific` cases.",
            "",
            "| Arm / category / metric | Difference |",
            "| --- | ---: |",
        ]
    )
    for name, value in summary["improvement_over_llm_only"].items():
        lines.append(f"| {name} | {_display(value)} |")

    lines.extend(["", "## Paraphrase robustness", ""])
    if summary["paraphrase_robustness"]:
        for name, value in summary["paraphrase_robustness"].items():
            lines.append(
                f"- `{name}`: hit rate {_display(value['hit_rate'])}, "
                f"consistency {_display(value['consistency'])}, "
                f"{value['members']} members"
            )
    else:
        lines.append("No eligible paraphrase groups were available.")

    lines.extend(["", "## Failure examples", ""])
    if summary["failure_examples"]:
        for failure in summary["failure_examples"]:
            labels = ", ".join(failure["failure_labels"]) or "execution_error"
            lines.append(
                f"- `{failure['arm']}:{failure['question_id']}` — {labels}"
                + (f" ({failure['error']})" if failure["error"] else "")
            )
    else:
        lines.append("No execution failures or classified failures.")
    lines.append("")
    return "\n".join(lines)
