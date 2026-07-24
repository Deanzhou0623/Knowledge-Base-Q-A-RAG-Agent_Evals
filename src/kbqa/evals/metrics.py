from typing import Any

from kbqa.evals.dataset import EvalCase
from kbqa.models import FALLBACK_ANSWER, RetrievalResult


def retrieval_metrics(case: EvalCase, results: list[RetrievalResult]) -> dict[str, Any]:
    document_gold = [source for source in case.acceptable_sources if ".md#" in source.citation]
    if not case.answerable or not document_gold:
        return {"recall_at_3": None, "hit_rate": None, "first_relevant_rank": None}

    full_hits: set[str] = set()
    section_hits: set[str] = set()
    first_rank: int | None = None
    for result in results[:3]:
        for gold in document_gold:
            if result.citation != gold.citation:
                continue
            section_hits.add(gold.citation)
            evidence_hit = False
            if gold.evidence_text is not None:
                evidence_hit = gold.evidence_text in result.text
            if evidence_hit:
                full_hits.add(gold.citation)
                first_rank = result.rank if first_rank is None else min(first_rank, result.rank)

    return {
        "recall_at_3": len(full_hits) / len(document_gold),
        "section_recall_at_3": len(section_hits) / len(document_gold),
        "hit_rate": bool(full_hits),
        "first_relevant_rank": first_rank,
    }


def answer_metrics(case: EvalCase, answer: str, citations: list[str]) -> dict[str, Any]:
    expected_fallback = not case.answerable
    return {
        "fallback_correct": (answer == FALLBACK_ANSWER) == expected_fallback,
        "citation_validity": (
            None
            if not case.answerable
            else bool(citations)
            and all(
                citation in {s.citation for s in case.acceptable_sources}
                for citation in citations
            )
        ),
    }
