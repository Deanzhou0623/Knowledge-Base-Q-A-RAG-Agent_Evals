import json

import pytest

from kbqa.evals.dataset import EvalCase, load_cases
from kbqa.evals.metrics import answer_metrics, retrieval_metrics
from kbqa.evals import runner
from kbqa.models import FALLBACK_ANSWER, RetrievalResult

from conftest import FakeEmbeddings, FakeGenerator


def test_dataset_requires_minimal_document_evidence(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "category": "company_specific",
                "question": "When?",
                "answerable": True,
                "acceptable_sources": [{"citation": "policy.md#refunds"}],
                "oracle_sources": ["policy.md#refunds"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence_text"):
        load_cases(path)


def test_retrieval_metrics_distinguish_heading_and_evidence_hits():
    case = EvalCase.model_validate(
        {
            "id": "case-1",
            "category": "company_specific",
            "question": "When?",
            "answerable": True,
            "acceptable_sources": [
                {"citation": "policy.md#refunds", "evidence_text": "7 to 11 days"}
            ],
            "oracle_sources": ["policy.md#refunds"],
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


def test_answer_metrics_require_a_citation_and_grade_fallback():
    answerable = EvalCase.model_validate(
        {
            "id": "case-1",
            "category": "company_specific",
            "question": "When?",
            "answerable": True,
            "acceptable_sources": [
                {"citation": "policy.md#refunds", "evidence_text": "7 days"}
            ],
            "oracle_sources": ["policy.md#refunds"],
        }
    )
    unsupported = EvalCase(
        id="case-2",
        category="unsupported",
        question="Where?",
        answerable=False,
    )

    assert answer_metrics(answerable, "Seven days", [])["citation_validity"] is False
    assert answer_metrics(unsupported, FALLBACK_ANSWER, [])["fallback_correct"] is True


def test_runner_executes_all_arms_offline(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    fixtures = tmp_path / "fixtures"
    evals = tmp_path / "evals"
    docs.mkdir()
    fixtures.mkdir()
    evals.mkdir()
    (docs / "policy.md").write_text(
        "# Refunds\nRefunds take seven days.", encoding="utf-8"
    )
    (fixtures / "bookings.json").write_text(
        '{"version":"v1","bookings":[]}', encoding="utf-8"
    )
    dataset = evals / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "answerable",
                        "category": "company_specific",
                        "question": "When?",
                        "answerable": True,
                        "acceptable_sources": [
                            {
                                "citation": "policy.md#refunds",
                                "evidence_text": "seven days",
                            }
                        ],
                        "oracle_sources": ["policy.md#refunds"],
                    }
                ),
                json.dumps(
                    {
                        "id": "unsupported",
                        "category": "unsupported",
                        "question": "Where?",
                        "answerable": False,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runner,
        "OpenAIAnswerGenerator",
        lambda *args, **kwargs: FakeGenerator(FALLBACK_ANSWER),
    )
    monkeypatch.setattr(
        runner,
        "OpenAIEmbeddingProvider",
        lambda *args, **kwargs: FakeEmbeddings(),
    )

    output = tmp_path / "results" / "eval.jsonl"
    records = runner.run(
        dataset, output, ["llm_only", "bm25", "vector", "oracle"]
    )

    assert len(records) == 7
    assert output.exists()
    assert {record["arm"] for record in records} == {
        "llm_only",
        "bm25",
        "vector",
        "oracle",
    }
    assert not any(
        record["arm"] == "oracle" and record["question_id"] == "unsupported"
        for record in records
    )
