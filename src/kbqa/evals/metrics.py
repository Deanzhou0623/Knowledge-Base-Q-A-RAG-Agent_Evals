import re
from collections import defaultdict
from statistics import fmean, pvariance
from typing import Any, Iterable

from kbqa.evals.dataset import EvalCase, GoldSource
from kbqa.models import FALLBACK_ANSWER, RetrievalResult


GRADER_VERSION = "deterministic-v2"
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "is",
    "it",
    "of",
    "on",
    "the",
    "through",
    "to",
}
FACT_TOKEN_RECALL_THRESHOLD = 0.6
NEGATIONS = {
    "no",
    "not",
    "never",
    "cannot",
    "cant",
    "dont",
    "doesnt",
    "isnt",
    "arent",
    "wont",
    "without",
}
SENTENCE_RE = re.compile(r"[.!?\n]+")


def normalize_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value.strip()).casefold()


def normalize_citation(value: str) -> str:
    citation = value.strip().replace("\\", "/")
    while citation.startswith("./"):
        citation = citation[2:]
    if ".md#" in citation:
        path, anchor = citation.rsplit("#", 1)
        return f"{path}#{anchor.casefold()}"
    return citation


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in WORD_RE.findall(normalize_text(value))
        if token not in STOPWORDS
    }


def _numeric_tokens(value: str) -> set[str]:
    return {token for token in WORD_RE.findall(normalize_text(value)) if any(c.isdigit() for c in token)}


def _negated(value: str) -> bool:
    return bool({token for token in WORD_RE.findall(normalize_text(value))} & NEGATIONS)


def _best_sentence(expected_tokens: set[str], answer: str) -> str:
    """The answer sentence that overlaps the expected fact most.

    Negation has to be judged against the clause that actually carries the
    matched tokens; scanning the whole answer would flag any response that
    happens to contain an unrelated "not" elsewhere.
    """
    sentences = [part for part in SENTENCE_RE.split(answer) if part.strip()]
    if not sentences:
        return answer
    return max(
        sentences,
        key=lambda sentence: len(expected_tokens & _content_tokens(sentence)),
    )


def _fact_coverage(expected: str, answer: str) -> float:
    """Deterministic token-overlap proxy for factual correctness.

    Bare content-token recall accepted a wrong number ("3 to 4 business days"
    against an expected "7 to 11 business days") and a negation of the expected
    fact, both of which shared most of their tokens with the reference. Numeric
    agreement and negation agreement are therefore gates, not contributions:
    failing either means the fact was not stated, whatever the overlap.

    This remains a lexical heuristic. It cannot detect a paraphrase that shares
    no vocabulary with the reference, and it is not a substitute for a rubric
    grader on a final benchmark.
    """
    expected_tokens = _content_tokens(expected)
    if not expected_tokens:
        return float(normalize_text(expected) in normalize_text(answer))

    expected_numbers = _numeric_tokens(expected)
    if expected_numbers and not expected_numbers <= _numeric_tokens(answer):
        return 0.0

    if _negated(expected) != _negated(_best_sentence(expected_tokens, answer)):
        return 0.0

    answer_tokens = _content_tokens(answer)
    return len(expected_tokens & answer_tokens) / len(expected_tokens)


def _evidence_hit(gold: GoldSource, result: RetrievalResult) -> bool:
    if gold.evidence_text is not None:
        return normalize_text(gold.evidence_text) in normalize_text(result.text)
    return False


def retrieval_metrics(case: EvalCase, results: list[RetrievalResult]) -> dict[str, Any]:
    acceptable_gold = [
        source for source in case.acceptable_sources if ".md#" in source.citation
    ]
    # Spec 06A: Oracle document references are the gold relevance labels, and
    # oracle_sources is the *minimal sufficient* evidence. Scoring against every
    # acceptable source would mark a retriever that returned exactly the minimal
    # evidence as incomplete. The broader acceptable set is still reported.
    oracle = {normalize_citation(citation) for citation in case.oracle_sources}
    document_gold = [
        source
        for source in acceptable_gold
        if normalize_citation(source.citation) in oracle
    ] or acceptable_gold
    if case.category == "unsupported":
        misleading = {
            normalize_citation(source) for source in case.unsupported_retrieval_sources
        }
        returned = {
            normalize_citation(result.citation) for result in results[:3]
        }
        return {
            "recall_at_3": None,
            "section_recall_at_3": None,
            "acceptable_recall_at_3": None,
            "hit_rate": None,
            "first_relevant_rank": None,
            "unsupported_retrieval": (
                bool(misleading & returned) if misleading else None
            ),
        }
    if (
        not case.answerable
        or not document_gold
        or (
            case.category == "user_specific"
            and case.requires_document_retrieval is False
        )
    ):
        return {
            "recall_at_3": None,
            "section_recall_at_3": None,
            "acceptable_recall_at_3": None,
            "hit_rate": None,
            "first_relevant_rank": None,
            "unsupported_retrieval": None,
        }

    full_hits: set[int] = set()
    section_hits: set[int] = set()
    first_rank: int | None = None
    for result in results[:3]:
        result_citation = normalize_citation(result.citation)
        for gold_index, gold in enumerate(document_gold):
            citation = normalize_citation(gold.citation)
            if result_citation != citation:
                continue
            section_hits.add(gold_index)
            if _evidence_hit(gold, result):
                full_hits.add(gold_index)
                first_rank = (
                    result.rank
                    if first_rank is None
                    else min(first_rank, result.rank)
                )

    acceptable_hits = {
        index
        for index, gold in enumerate(acceptable_gold)
        for result in results[:3]
        if normalize_citation(result.citation) == normalize_citation(gold.citation)
        and _evidence_hit(gold, result)
    }
    return {
        "recall_at_3": len(full_hits) / len(document_gold),
        "section_recall_at_3": len(section_hits) / len(document_gold),
        "acceptable_recall_at_3": (
            len(acceptable_hits) / len(acceptable_gold) if acceptable_gold else None
        ),
        "hit_rate": bool(full_hits),
        "first_relevant_rank": first_rank,
        "unsupported_retrieval": None,
    }


def answer_metrics(
    case: EvalCase,
    answer: str,
    citations: list[str],
    *,
    arm: str = "bm25",
    supplied_citations: Iterable[str] = (),
    raw_citations: list[str] | None = None,
) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)
    fact_coverages = [
        _fact_coverage(fact, answer) for fact in case.expected_facts
    ]
    matched_facts = [
        coverage
        for coverage in fact_coverages
        if coverage >= FACT_TOKEN_RECALL_THRESHOLD
    ]
    correctness = (
        len(matched_facts) / len(fact_coverages) if fact_coverages else None
    )
    exact_fallback = answer.strip() == FALLBACK_ANSWER
    is_contextual = arm in {"bm25", "vector", "oracle"}
    if case.answerable:
        fallback_correct = not exact_fallback
    elif is_contextual:
        fallback_correct = exact_fallback
    else:
        fallback_correct = (
            "cannot confirm" in normalized_answer
            or "don't know" in normalized_answer
            or "do not know" in normalized_answer
            or "uncertain" in normalized_answer
        )

    if arm == "llm_only":
        citation_accuracy = None
        citation_validity = None
    else:
        acceptable = {
            normalize_citation(source.citation)
            for source in case.acceptable_sources
        }
        supplied = {normalize_citation(citation) for citation in supplied_citations}
        normalized_citations = [normalize_citation(citation) for citation in citations]
        # Grade what the model actually produced. The served citations have
        # already had anything invalid stripped by the guardrail, so
        # grading them makes citation_validity unconditionally true.
        graded_citations = (
            [normalize_citation(citation) for citation in raw_citations]
            if raw_citations is not None
            else normalized_citations
        )
        citation_validity = all(
            citation in supplied for citation in graded_citations
        )
        citation_accuracy = (
            sum(citation in acceptable for citation in normalized_citations)
            / len(normalized_citations)
            if normalized_citations
            else (1.0 if not case.answerable and exact_fallback else 0.0)
        )

    if not case.answerable:
        unsupported_claim_rate = 0.0 if fallback_correct else 1.0
    elif arm == "llm_only":
        unsupported_claim_rate = (
            None if correctness is None else 1.0 - correctness
        )
    else:
        unsupported_claim_rate = (
            1.0
            if not citation_validity
            else (None if correctness is None else 1.0 - correctness)
        )

    return {
        "correctness": correctness,
        "matched_facts": len(matched_facts),
        "expected_facts": len(fact_coverages),
        "fact_coverages": fact_coverages,
        "fact_token_recall_threshold": FACT_TOKEN_RECALL_THRESHOLD,
        "citation_accuracy": citation_accuracy,
        "citation_validity": citation_validity,
        "unsupported_claim_rate": unsupported_claim_rate,
        "fallback_correct": fallback_correct,
        "exact_fallback": exact_fallback,
        "refusal_calibrated": fallback_correct if not case.answerable else None,
        "control_citation_violation": bool(citations) if arm == "llm_only" else None,
        "grader_version": GRADER_VERSION,
        "grader_explanation": (
            f"Matched {len(matched_facts)}/{len(fact_coverages)} expected facts "
            f"using content-token recall >= {FACT_TOKEN_RECALL_THRESHOLD:.1f}; "
            f"fallback_correct={fallback_correct}; "
            f"citation_validity={citation_validity}."
        ),
    }


def paraphrase_robustness(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in records:
        group = record.get("paraphrase_group_id")
        metrics = record.get("retrieval_metrics")
        if (
            record.get("status") == "ok"
            and group
            and record.get("arm") in {"bm25", "vector"}
            and metrics
            and metrics.get("hit_rate") is not None
        ):
            groups[(record["arm"], group)].append(bool(metrics["hit_rate"]))
    return {
        f"{arm}:{group}": {
            "members": len(outcomes),
            "hit_rate": sum(outcomes) / len(outcomes),
            "consistency": float(all(outcomes) or not any(outcomes)),
        }
        for (arm, group), outcomes in sorted(groups.items())
        if len(outcomes) >= 2
    }


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "variance": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "variance": pvariance(values) if len(values) > 1 else 0.0,
    }
