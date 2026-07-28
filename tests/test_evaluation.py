import json
from types import SimpleNamespace
from datetime import date
from pathlib import Path

import pytest

from kbqa.config import Settings
from kbqa.evals import runner
from kbqa.evals.dataset import EvalCase, load_cases, validate_dataset
from kbqa.evals.rubric import OpenAIRubricGrader, parse_rubric_reply
from kbqa.evals.metrics import (
    answer_metrics,
    paraphrase_robustness,
    retrieval_metrics,
)
from kbqa.io import corpus_fingerprint, sha256_file
from kbqa.llm import GeneratedAnswer
from kbqa.models import FALLBACK_ANSWER, RetrievalResult, TokenUsage

from conftest import FakeEmbeddings, FakeGenerator


class ControlledGenerator:
    model = "gpt-5.6-sol"

    def grounded(self, prompt_input: str) -> GeneratedAnswer:
        if "Where is the moon desk?" in prompt_input:
            text = FALLBACK_ANSWER
        elif "booking:BK-1#status" in prompt_input:
            text = "confirmed [booking:BK-1#status]"
        else:
            text = "seven days [policy.md#refunds]"
        return GeneratedAnswer(text=text, token_usage=TokenUsage(total_tokens=5))

    def closed_book(self, question: str) -> GeneratedAnswer:
        return GeneratedAnswer(
            text="I cannot confirm.", token_usage=TokenUsage(total_tokens=3)
        )


class FailingGenerator(ControlledGenerator):
    def grounded(self, prompt_input: str) -> GeneratedAnswer:
        raise RuntimeError("provider unavailable")


class FailingEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts):
        raise RuntimeError("embedding provider unavailable")


def _write_dataset(root: Path, *, trials: int = 1) -> tuple[Path, Path, Settings]:
    docs = root / "docs"
    fixtures = root / "fixtures"
    evals = root / "evals"
    docs.mkdir()
    fixtures.mkdir()
    evals.mkdir()
    (docs / "policy.md").write_text(
        "# Refunds\nRefunds take seven days.\n\n"
        "# Chargebacks\nA chargeback is a bank reversal.",
        encoding="utf-8",
    )
    (fixtures / "bookings.json").write_text(
        json.dumps(
            {
                "version": "v1",
                "bookings": [
                    {
                        "id": "BK-1",
                        "status": "confirmed",
                        "property_id": "P-1",
                        "check_in": "2026-09-01",
                        "free_cancellation_until": "2026-08-10T18:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cases = [
        {
            "id": "company",
            "category": "company_specific",
            "question": "When is the refund?",
            "answerable": True,
            "expected_facts": ["seven days"],
            "acceptable_sources": [
                {
                    "citation": "policy.md#refunds",
                    "evidence_text": "seven days",
                }
            ],
            "oracle_sources": ["policy.md#refunds"],
            "paraphrase_group_id": "refund",
        },
        {
            "id": "generic",
            "category": "generic_ecommerce",
            "question": "What is a chargeback?",
            "answerable": True,
            "expected_facts": ["bank reversal"],
            "acceptable_sources": [
                {
                    "citation": "policy.md#chargebacks",
                    "evidence_text": "bank reversal",
                }
            ],
            "oracle_sources": ["policy.md#chargebacks"],
        },
        {
            "id": "user",
            "category": "user_specific",
            "question": "What is booking BK-1's status?",
            "answerable": True,
            "expected_facts": ["confirmed"],
            "acceptable_sources": [{"citation": "booking:BK-1#status"}],
            "oracle_sources": ["booking:BK-1#status"],
            "requires_document_retrieval": False,
            "booking_id": "BK-1",
            "transaction_fixture_version": "v1",
            "as_of": "2026-08-01T12:00:00Z",
        },
        {
            "id": "unsupported",
            "category": "unsupported",
            "question": "Where is the moon desk?",
            "answerable": False,
            "expected_fallback": FALLBACK_ANSWER,
            "unsupported_retrieval_sources": ["policy.md#refunds"],
        },
    ]
    cases_path = evals / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )
    manifest = {
        "dataset_version": "test-v1",
        "annotation_date": date.today().isoformat(),
        "corpus_fingerprint": corpus_fingerprint(docs),
        "transaction_fixture_version": "v1",
        "minimum_questions_per_category": 1,
        "trials_per_question": trials,
        "estimate_type": (
            "single_run_point_estimate" if trials == 1 else "repeated_trials"
        ),
        "annotation_blinded": True,
        "corpus_tier": "primary_controlled",
        "grader_version": "deterministic-v2",
        "dataset_id": "test-dataset",
        "cases_file": cases_path.name,
        "cases_sha256": sha256_file(cases_path),
        "change_reason": "Test fixture frozen before execution.",
    }
    manifest_path = evals / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    settings = Settings(
        docs_path=docs,
        kb_path=root / ".kb",
        transaction_fixture_path=fixtures / "bookings.json",
    )
    return cases_path, manifest_path, settings


def test_dataset_requires_minimal_document_evidence(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "category": "company_specific",
                "question": "When?",
                "answerable": True,
                "expected_facts": ["seven days"],
                "acceptable_sources": [{"citation": "policy.md#refunds"}],
                "oracle_sources": ["policy.md#refunds"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evidence_text or both offsets"):
        load_cases(path)


def test_dataset_validates_fingerprint_evidence_and_category_cells(tmp_path):
    cases, manifest, settings = _write_dataset(tmp_path)
    loaded = validate_dataset(
        cases,
        manifest,
        docs_path=settings.docs_path,
        transaction_fixture_path=settings.transaction_fixture_path,
    )
    assert loaded.manifest.dataset_version == "test-v1"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["corpus_fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_dataset(
            cases,
            manifest,
            docs_path=settings.docs_path,
            transaction_fixture_path=settings.transaction_fixture_path,
        )


def test_retrieval_metrics_distinguish_heading_and_evidence_hits():
    case = EvalCase.model_validate(
        {
            "id": "case-1",
            "category": "company_specific",
            "question": "When?",
            "answerable": True,
            "expected_facts": ["7 to 11 days"],
            "acceptable_sources": [
                {"citation": "./policy.md#REFUNDS", "evidence_text": "7 to 11 days"}
            ],
            "oracle_sources": ["./policy.md#REFUNDS"],
        }
    )
    result = RetrievalResult(
        id="1",
        source_path="policy.md",
        heading="Refunds",
        anchor="refunds",
        citation="policy.md#refunds",
        text="A different sentence",
        score=1,
        rank=1,
    )
    metrics = retrieval_metrics(case, [result])
    assert metrics["section_recall_at_3"] == 1
    assert metrics["recall_at_3"] == 0
    assert metrics["first_relevant_rank"] is None


def test_transaction_only_and_llm_retrieval_metrics_are_na():
    case = EvalCase.model_validate(
        {
            "id": "user",
            "category": "user_specific",
            "question": "Status?",
            "answerable": True,
            "expected_facts": ["confirmed"],
            "acceptable_sources": [{"citation": "booking:BK-1#status"}],
            "oracle_sources": ["booking:BK-1#status"],
            "requires_document_retrieval": False,
            "booking_id": "BK-1",
            "transaction_fixture_version": "v1",
            "as_of": "2026-08-01T12:00:00Z",
        }
    )
    assert all(value is None for value in retrieval_metrics(case, []).values())


def test_answer_metrics_check_supplied_context_and_arm_specific_refusal():
    answerable = EvalCase.model_validate(
        {
            "id": "case-1",
            "category": "company_specific",
            "question": "When?",
            "answerable": True,
            "expected_facts": ["seven days"],
            "acceptable_sources": [
                {"citation": "policy.md#refunds", "evidence_text": "seven days"}
            ],
            "oracle_sources": ["policy.md#refunds"],
        }
    )
    metrics = answer_metrics(
        answerable,
        "seven days [policy.md#refunds]",
        ["policy.md#refunds"],
        arm="bm25",
        supplied_citations=["policy.md#other"],
    )
    assert metrics["correctness"] == 1
    assert metrics["citation_validity"] is False
    unsupported = EvalCase(
        id="case-2",
        category="unsupported",
        question="Where?",
        answerable=False,
        expected_fallback=FALLBACK_ANSWER,
    )
    assert answer_metrics(
        unsupported, FALLBACK_ANSWER, [], arm="bm25"
    )["fallback_correct"]
    llm = answer_metrics(
        unsupported, "I cannot confirm.", [], arm="llm_only"
    )
    assert llm["fallback_correct"] is True
    assert llm["citation_validity"] is None


def test_runner_executes_controlled_matrix_offline_and_writes_reports(tmp_path):
    cases, manifest, settings = _write_dataset(tmp_path, trials=2)
    output = tmp_path / "results" / "eval.jsonl"
    records = runner.run(
        cases,
        output,
        ["llm_only", "bm25", "vector", "oracle"],
        manifest=manifest,
        settings=settings,
        generator=ControlledGenerator(),
        embeddings=FakeEmbeddings(),
    )
    assert len(records) == 30
    assert output.exists()
    assert output.with_suffix(".summary.json").exists()
    assert output.with_suffix(".report.md").exists()
    assert len({frozenset(record) for record in records}) == 1
    assert all(record["status"] == "ok" for record in records)
    assert all(record["model"] == "gpt-5.6-sol" for record in records)
    assert all(record["trials_per_question"] == 2 for record in records)
    llm = next(record for record in records if record["arm"] == "llm_only")
    assert llm["retrieved"] is None
    assert llm["retrieval_metrics"] is None
    transaction_only = [
        record
        for record in records
        if record["question_id"] == "user"
        and record["arm"] in {"bm25", "vector"}
    ]
    assert all(
        record["retrieval_metrics"]["recall_at_3"] is None
        for record in transaction_only
    )
    summary = json.loads(
        output.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    assert summary["improvement_excluded_categories"] == [
        "unsupported",
        "user_specific",
    ]
    assert "bm25:company_specific" in summary["arm_category_cells"]
    assert all(
        ":unsupported:" not in key and ":user_specific:" not in key
        for key in summary["improvement_over_llm_only"]
    )


def test_runner_preserves_provider_failures_as_records(tmp_path):
    cases, manifest, settings = _write_dataset(tmp_path)
    records = runner.run(
        cases,
        tmp_path / "results.jsonl",
        ["llm_only", "bm25", "vector"],
        manifest=manifest,
        settings=settings,
        generator=FailingGenerator(),
        embeddings=FakeEmbeddings(),
        retries=1,
    )
    failed = [
        record
        for record in records
        if record["arm"] in {"bm25", "vector"}
    ]
    assert failed
    assert all(record["status"] == "error" for record in failed)
    assert all(record["attempts"] == 2 for record in failed)


def test_runner_preserves_backend_index_failure_as_case_errors(tmp_path):
    cases, manifest, settings = _write_dataset(tmp_path)
    records = runner.run(
        cases,
        tmp_path / "results.jsonl",
        ["llm_only", "bm25", "vector"],
        manifest=manifest,
        settings=settings,
        generator=ControlledGenerator(),
        embeddings=FailingEmbeddings(),
    )
    vector = [record for record in records if record["arm"] == "vector"]
    assert vector
    assert all(record["status"] == "error" for record in vector)
    assert all("embedding provider unavailable" in record["index_error"] for record in vector)
    assert all(record["status"] == "ok" for record in records if record["arm"] != "vector")


def test_live_calls_and_incomplete_matrix_require_explicit_opt_in(tmp_path):
    cases, manifest, settings = _write_dataset(tmp_path)
    with pytest.raises(RuntimeError, match="disabled"):
        runner.run(
            cases,
            tmp_path / "output.jsonl",
            ["llm_only", "bm25", "vector"],
            manifest=manifest,
            settings=settings,
        )
    with pytest.raises(ValueError, match="requires llm_only"):
        runner.run(
            cases,
            tmp_path / "output.jsonl",
            ["oracle"],
            manifest=manifest,
            settings=settings,
            generator=ControlledGenerator(),
            embeddings=FakeEmbeddings(),
        )


def test_oracle_only_execution_never_builds_or_searches_retrievers(
    tmp_path, monkeypatch
):
    cases, manifest, settings = _write_dataset(tmp_path)
    monkeypatch.setattr(
        runner,
        "build_service",
        lambda *args, **kwargs: pytest.fail("oracle invoked a retriever"),
    )
    records = runner.run(
        cases,
        tmp_path / "oracle.jsonl",
        ["oracle"],
        manifest=manifest,
        settings=settings,
        generator=ControlledGenerator(),
        embeddings=FakeEmbeddings(),
        require_main_arms=False,
    )
    assert len(records) == 3
    assert all(record["arm"] == "oracle" for record in records)


def test_paraphrase_robustness_requires_repeated_group_members():
    records = [
        {
            "status": "ok",
            "arm": "bm25",
            "paraphrase_group_id": "refund",
            "retrieval_metrics": {"hit_rate": True},
        },
        {
            "status": "ok",
            "arm": "bm25",
            "paraphrase_group_id": "refund",
            "retrieval_metrics": {"hit_rate": False},
        },
    ]
    assert paraphrase_robustness(records) == {
        "bm25:refund": {"members": 2, "hit_rate": 0.5, "consistency": 0.0}
    }


def test_records_report_the_model_that_actually_ran(tmp_path):
    # Previously every record wrote settings.openai_chat_model, so injecting a
    # fake still produced records claiming gpt-5.6-sol.
    cases, manifest, settings = _write_dataset(tmp_path)
    records = runner.run(
        cases,
        tmp_path / "results" / "eval.jsonl",
        ["llm_only", "bm25", "vector"],
        manifest=manifest,
        settings=settings,
        generator=FakeGenerator(FALLBACK_ANSWER),
        embeddings=FakeEmbeddings(),
    )

    assert {record["model"] for record in records} == {"fake-answer-model"}
    vector = [r for r in records if r["arm"] == "vector"]
    assert {r["embedding_model"] for r in vector} == {"fake-embeddings-v1"}


def test_citation_validity_grades_the_pre_guardrail_generation(tmp_path):
    # The service replaces a fabricated citation with the exact fallback, so
    # grading the served citations made citation_validity unconditionally true.
    cases, manifest, settings = _write_dataset(tmp_path)
    fabricating = FakeGenerator("Tomorrow. [invented.md#answer]")
    records = runner.run(
        cases,
        tmp_path / "results" / "eval.jsonl",
        ["llm_only", "bm25", "vector"],
        manifest=manifest,
        settings=settings,
        generator=fabricating,
        embeddings=FakeEmbeddings(),
    )

    rag = [r for r in records if r["arm"] in {"bm25", "vector"}]
    assert rag
    assert all(r["citation_guardrail_triggered"] for r in rag)
    assert all(r["answer"] == FALLBACK_ANSWER for r in rag)
    assert all(r["raw_answer"] == "Tomorrow. [invented.md#answer]" for r in rag)
    assert all(r["raw_citations"] == ["invented.md#answer"] for r in rag)
    assert all(r["answer_metrics"]["citation_validity"] is False for r in rag)


@pytest.mark.parametrize(
    ("answer", "expected_coverage"),
    [
        ("Approved refunds take 7 to 11 business days.", 1.0),
        # Wrong number: shares 5 of 7 content tokens, previously scored correct.
        ("Approved refunds take 3 to 4 business days.", 0.0),
        # Negation of the expected fact: previously scored a perfect 1.0.
        ("Approved refunds do not take 7 to 11 business days.", 0.0),
    ],
)
def test_fact_coverage_gates_on_numbers_and_negation(answer, expected_coverage):
    case = EvalCase.model_validate(
        {
            "id": "case-1",
            "category": "company_specific",
            "question": "How long do refunds take?",
            "answerable": True,
            "expected_facts": ["Approved refunds take 7 to 11 business days."],
            "acceptable_sources": [
                {"citation": "policy.md#refunds", "evidence_text": "7 to 11"}
            ],
            "oracle_sources": ["policy.md#refunds"],
        }
    )

    metrics = answer_metrics(
        case, answer, ["policy.md#refunds"], supplied_citations=["policy.md#refunds"]
    )

    assert metrics["fact_coverages"] == [expected_coverage]


def test_offset_only_gold_outside_oracle_is_resolved_and_scored(tmp_path):
    # An acceptable-but-not-oracle source specified by offsets kept
    # evidence_text None, so _evidence_hit could never match it and recall was
    # capped below 1.0 for both RAG arms regardless of what they retrieved.
    cases_path, manifest_path, settings = _write_dataset(tmp_path)
    records = [
        json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()
    ]
    for record in records:
        if record["id"] == "company":
            record["acceptable_sources"].append(
                {
                    "citation": "policy.md#chargebacks",
                    "start_offset": 0,
                    "end_offset": 12,
                }
            )
    cases_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["cases_sha256"] = sha256_file(cases_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = validate_dataset(
        cases_path,
        manifest_path,
        docs_path=settings.docs_path,
        transaction_fixture_path=settings.transaction_fixture_path,
    )

    case = next(c for c in dataset.cases if c.id == "company")
    extra = next(
        s for s in case.acceptable_sources if s.citation == "policy.md#chargebacks"
    )
    # Backfilled from the offsets, so retrieval scoring can now match it.
    assert extra.evidence_text == "A chargeback"


def test_rubric_reply_parsing_never_credits_a_malformed_verdict():
    assert parse_rubric_reply("SUPPORTED\nStates 7 to 11 days.").verdict == "supported"
    assert parse_rubric_reply("CONTRADICTED\nSays 3 days.").verdict == "contradicted"
    assert parse_rubric_reply("MISSING\nSilent on timing.").verdict == "missing"
    # A malformed or empty grader reply must not be scored as correct.
    for reply in ["", "   ", "maybe?", "I think it is fine"]:
        verdict = parse_rubric_reply(reply)
        assert verdict.verdict == "missing"
        assert verdict.score == 0.0
        assert verdict.supported is False


def test_openai_rubric_grader_is_pinned_and_records_its_identity():
    class FakeResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(output_text="SUPPORTED\nStates the refund window.")

    responses = FakeResponses()
    grader = OpenAIRubricGrader(client=SimpleNamespace(responses=responses))

    verdict = grader.grade_fact("How long?", "Refunds take 7 to 11 days.", "7 to 11 days.")

    assert verdict.supported is True
    assert grader.identity.model == "gpt-5.6-sol"
    assert grader.identity.prompt_version == "rubric-v1"
    assert responses.calls[0]["model"] == "gpt-5.6-sol"

    with pytest.raises(ValueError, match="gpt-5.6-sol"):
        OpenAIRubricGrader(model="another-model")


def test_records_carry_the_identity_of_the_grader_that_produced_them(tmp_path):
    cases_path, manifest_path, settings = _write_dataset(tmp_path)
    records = runner.run(
        cases_path,
        tmp_path / "results" / "eval.jsonl",
        ["llm_only", "bm25", "vector"],
        manifest=manifest_path,
        settings=settings,
        generator=ControlledGenerator(),
        embeddings=FakeEmbeddings(),
    )

    graded = [r for r in records if r["answer_metrics"]["expected_facts"]]
    assert graded
    for record in graded:
        metrics = record["answer_metrics"]
        assert metrics["grader_name"] == "lexical"
        assert metrics["grader_version"] == "deterministic-v2"
        assert len(metrics["fact_verdicts"]) == metrics["expected_facts"]
        assert all(v["verdict"] for v in metrics["fact_verdicts"])
