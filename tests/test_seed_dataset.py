import json
import shutil
from pathlib import Path

import pytest

from pydantic import ValidationError

from kbqa.evals.dataset import EvalCase, load_frozen_dataset
from kbqa.io import sha256_file
from kbqa.models import FALLBACK_ANSWER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "evals" / "seed-v1.manifest.json"
DOCS = PROJECT_ROOT / "docs"
TRANSACTIONS = PROJECT_ROOT / "fixtures" / "bookings.json"


def test_frozen_seed_dataset_validates_offline_and_covers_required_cases():
    dataset = load_frozen_dataset(MANIFEST, DOCS, TRANSACTIONS)

    assert dataset.manifest.dataset_version == "seed-v1"
    assert dataset.manifest.status == "frozen"
    assert dataset.manifest.oracle_evidence_selected_before_outputs is True
    assert len(dataset.cases) == 6
    assert {case.category for case in dataset.cases} == {
        "company_specific",
        "generic_ecommerce",
        "user_specific",
        "unsupported",
    }
    paraphrases = [
        case
        for case in dataset.cases
        if case.paraphrase_group_id == "refund-timeline"
    ]
    assert len(paraphrases) == 2
    assert {
        case.requires_document_retrieval
        for case in dataset.cases
        if case.category == "user_specific"
    } == {False, True}


def test_frozen_seed_rejects_a_changed_corpus(tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(DOCS, docs)
    (docs / "platform" / "refunds.md").write_text(
        "# Refund Policy\n\nChanged after the seed freeze.\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Corpus fingerprint mismatch"):
        load_frozen_dataset(MANIFEST, docs, TRANSACTIONS)


def test_frozen_seed_rejects_cases_changed_without_versioned_manifest(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        (PROJECT_ROOT / "evals" / "cases.jsonl").read_text(encoding="utf-8")
        + "\n",
        encoding="utf-8",
    )
    manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_payload["cases_file"] = cases.name
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Frozen cases hash mismatch"):
        load_frozen_dataset(manifest, DOCS, TRANSACTIONS)


def test_frozen_seed_rejects_unresolvable_minimal_evidence(tmp_path):
    original = (PROJECT_ROOT / "evals" / "cases.jsonl").read_text(encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        original.replace(
            '"evidence_text":"7 to 11 business days"',
            '"evidence_text":"an annotation that is absent"',
            1,
        ),
        encoding="utf-8",
    )
    manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_payload["cases_file"] = cases.name
    manifest_payload["cases_sha256"] = sha256_file(cases)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Evidence text does not resolve"):
        load_frozen_dataset(manifest, DOCS, TRANSACTIONS)


def test_case_contract_separates_transaction_only_from_document_retrieval():
    base = {
        "id": "booking-test",
        "category": "user_specific",
        "question": "What is my status?",
        "answerable": True,
        "expected_facts": ["The booking is confirmed."],
        "acceptable_sources": [{"citation": "booking:BK-1#status"}],
        "oracle_sources": ["booking:BK-1#status"],
        "booking_id": "BK-1",
        "transaction_fixture_version": "v1",
        "as_of": "2026-08-01T12:00:00Z",
    }

    transaction_only = EvalCase.model_validate(
        {**base, "requires_document_retrieval": False}
    )
    assert transaction_only.requires_document_retrieval is False

    with pytest.raises(ValueError, match="require document evidence"):
        EvalCase.model_validate({**base, "requires_document_retrieval": True})


def test_unsupported_case_requires_exact_fallback_contract():
    with pytest.raises(ValueError, match="exact fallback"):
        EvalCase.model_validate(
            {
                "id": "unsupported-test",
                "category": "unsupported",
                "question": "What is absent?",
                "answerable": False,
            }
        )

    case = EvalCase.model_validate(
        {
            "id": "unsupported-test",
            "category": "unsupported",
            "question": "What is absent?",
            "answerable": False,
            "expected_fallback": FALLBACK_ANSWER,
        }
    )
    assert case.expected_fallback == FALLBACK_ANSWER


def test_loaded_cases_are_frozen_against_in_process_edits():
    dataset = load_frozen_dataset(
        MANIFEST, DOCS, TRANSACTIONS
    )
    case = dataset.cases[0]

    with pytest.raises(ValidationError):
        case.question = "MUTATED"
    with pytest.raises(ValidationError):
        case.acceptable_sources[0].citation = "MUTATED"


def test_user_specific_cases_require_an_as_of_clock():
    with pytest.raises(ValueError, match="as_of"):
        EvalCase.model_validate(
            {
                "id": "booking-x",
                "category": "user_specific",
                "question": "What is the status of booking BK-1?",
                "answerable": True,
                "requires_document_retrieval": False,
                "booking_id": "BK-1",
                "transaction_fixture_version": "bookings-v1",
                "acceptable_sources": [{"citation": "booking:BK-1#status"}],
                "oracle_sources": ["booking:BK-1#status"],
            }
        )
