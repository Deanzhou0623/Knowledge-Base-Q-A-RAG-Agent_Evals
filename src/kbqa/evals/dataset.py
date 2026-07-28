import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kbqa.io import corpus_fingerprint, sha256_file
from kbqa.models import FALLBACK_ANSWER
from kbqa.parsing import load_sections
from kbqa.transactions import TransactionStore


CATEGORIES = (
    "company_specific",
    "generic_ecommerce",
    "user_specific",
    "unsupported",
)
DOCUMENT_CITATION_RE = re.compile(r"^.+\.md#[^#]+$")
STRUCTURED_CITATION_RE = re.compile(r"^[a-z][a-z0-9_-]*:[^#]+#[^#]+$")


def canonicalize_citation(value: str) -> str:
    citation = value.strip().replace("\\", "/")
    while citation.startswith("./"):
        citation = citation[2:]
    if ".md#" in citation:
        path, anchor = citation.rsplit("#", 1)
        return f"{path}#{anchor.casefold()}"
    return citation


class GoldSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation: str
    evidence_text: str | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=1)

    @field_validator("citation")
    @classmethod
    def normalize_source_citation(cls, value: str) -> str:
        return canonicalize_citation(value)

    @model_validator(mode="after")
    def require_evidence_for_documents(self) -> "GoldSource":
        if DOCUMENT_CITATION_RE.match(self.citation):
            has_text = self.evidence_text is not None and bool(self.evidence_text.strip())
            has_offsets = self.start_offset is not None and self.end_offset is not None
            if not has_text and not has_offsets:
                raise ValueError(
                    "Document gold sources require evidence_text or both offsets"
                )
            if (self.start_offset is None) != (self.end_offset is None):
                raise ValueError("Evidence offsets must be provided together")
            if has_offsets and self.end_offset <= self.start_offset:
                raise ValueError("end_offset must be greater than start_offset")
        elif not STRUCTURED_CITATION_RE.match(self.citation):
            raise ValueError(f"Unsupported citation format: {self.citation!r}")
        return self


class EvalCase(BaseModel):
    # Frozen: the manifest digest protects the cases file at load time only.
    # Without this a runner or metric could rewrite a question or its gold
    # labels mid-run with no signal.
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    category: Literal[
        "company_specific", "generic_ecommerce", "user_specific", "unsupported"
    ]
    question: str = Field(min_length=1)
    answerable: bool
    expected_facts: list[str] = Field(default_factory=list)
    # Spec 02: unsupported cases must declare the documented refusal.
    expected_fallback: str | None = None
    acceptable_sources: list[GoldSource] = Field(default_factory=list)
    oracle_sources: list[str] = Field(default_factory=list)
    paraphrase_group_id: str | None = None
    requires_document_retrieval: bool | None = None
    booking_id: str | None = None
    as_of: datetime | None = None
    transaction_fixture_version: str | None = None
    expected_failure_label: str | None = None
    unsupported_retrieval_sources: list[str] = Field(default_factory=list)

    @field_validator("oracle_sources", "unsupported_retrieval_sources")
    @classmethod
    def normalize_citation_lists(cls, values: list[str]) -> list[str]:
        return [canonicalize_citation(value) for value in values]

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must contain non-whitespace text")
        return question

    @model_validator(mode="after")
    def validate_case_contract(self) -> "EvalCase":
        if self.category == "user_specific":
            if self.requires_document_retrieval is None or self.booking_id is None:
                raise ValueError(
                    "user_specific cases require requires_document_retrieval and booking_id"
                )
            if self.transaction_fixture_version is None:
                raise ValueError(
                    "user_specific cases require transaction_fixture_version"
                )
            if self.as_of is None:
                raise ValueError(
                    "user_specific cases require an as_of evaluation clock so "
                    "transaction state resolves deterministically"
                )
        elif self.requires_document_retrieval is not None:
            raise ValueError(
                "requires_document_retrieval is valid only for user_specific cases"
            )
        if self.answerable:
            if not self.expected_facts:
                raise ValueError("Answerable cases require expected_facts")
            if not self.oracle_sources:
                raise ValueError("Answerable cases require oracle_sources")
        else:
            if self.category != "unsupported":
                raise ValueError("Only unsupported cases may be unanswerable")
            if self.expected_facts or self.acceptable_sources or self.oracle_sources:
                raise ValueError(
                    "Unsupported cases cannot define expected facts or gold evidence"
                )
            if self.expected_fallback != FALLBACK_ANSWER:
                raise ValueError(
                    "Unsupported cases must expect the exact fallback"
                )
        acceptable = {source.citation for source in self.acceptable_sources}
        if not set(self.oracle_sources).issubset(acceptable):
            raise ValueError("oracle_sources must also appear in acceptable_sources")
        document_sources = {
            source for source in acceptable if DOCUMENT_CITATION_RE.match(source)
        }
        structured_sources = acceptable - document_sources
        if self.answerable and self.category != "user_specific" and not document_sources:
            raise ValueError("Document-backed answerable cases require document evidence")
        if self.category == "user_specific":
            if not structured_sources:
                raise ValueError("user_specific cases require structured evidence")
            if self.requires_document_retrieval and not document_sources:
                raise ValueError(
                    "transaction-plus-document cases require document evidence"
                )
            if self.requires_document_retrieval is False and document_sources:
                raise ValueError(
                    "transaction-only cases cannot define document evidence"
                )
        return self


class DatasetManifest(BaseModel):
    """Unified dataset manifest.

    Merges the Spec 02 seed-freeze record (identity, cases digest, change
    reason) with the Spec 06 run-control fields (trials, minimum cell size,
    grader version). One manifest so a run cannot be controlled by one
    document while the freeze is recorded in another.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    status: Literal["frozen"] = "frozen"
    cases_file: str = Field(min_length=1)
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_evidence_selected_before_outputs: Literal[True] = True
    change_reason: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    annotation_date: date
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    transaction_fixture_version: str
    minimum_questions_per_category: int = Field(ge=1)
    trials_per_question: int = Field(default=1, ge=1)
    estimate_type: Literal["single_run_point_estimate", "repeated_trials"]
    annotation_blinded: bool
    corpus_tier: Literal["primary_controlled", "secondary_real_world"]
    grader_version: str = "deterministic-v2"
    notes: list[str] = Field(default_factory=list)

    @field_validator("cases_file")
    @classmethod
    def cases_file_must_be_relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("cases_file must be a safe relative path")
        return path.as_posix()

    @model_validator(mode="after")
    def trial_description_matches(self) -> "DatasetManifest":
        expected = (
            "single_run_point_estimate"
            if self.trials_per_question == 1
            else "repeated_trials"
        )
        if self.estimate_type != expected:
            raise ValueError(
                f"estimate_type must be {expected!r} for "
                f"{self.trials_per_question} trial(s)"
            )
        return self


class EvaluationDataset(BaseModel):
    manifest: DatasetManifest
    cases: list[EvalCase]


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid evaluation case at {path}:{line_number}: {exc}"
            ) from exc
    if not cases:
        raise ValueError("Evaluation dataset cannot be empty")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique")
    return cases


def load_manifest(path: Path) -> DatasetManifest:
    try:
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid evaluation manifest at {path}: {exc}") from exc


def validate_dataset(
    cases_path: Path,
    manifest_path: Path,
    *,
    docs_path: Path,
    transaction_fixture_path: Path,
) -> EvaluationDataset:
    cases = load_cases(cases_path)
    manifest = load_manifest(manifest_path)

    # The freeze is only real if the digest is checked on the file actually
    # being evaluated, not merely recorded in the manifest.
    expected_cases = manifest_path.parent / manifest.cases_file
    if expected_cases.resolve() != cases_path.resolve():
        raise ValueError(
            f"Manifest pins cases_file {manifest.cases_file!r} but {cases_path} "
            "was supplied"
        )
    observed_digest = sha256_file(cases_path)
    if observed_digest != manifest.cases_sha256:
        raise ValueError(
            "Evaluation cases do not match the frozen manifest digest: "
            f"{manifest.cases_sha256} != {observed_digest}"
        )

    if not manifest.annotation_blinded:
        raise ValueError("Final evaluation data must be annotated blind to arm outputs")

    observed_fingerprint = corpus_fingerprint(docs_path)
    if observed_fingerprint != manifest.corpus_fingerprint:
        raise ValueError(
            "Dataset corpus fingerprint does not match the indexed corpus: "
            f"{manifest.corpus_fingerprint} != {observed_fingerprint}"
        )

    try:
        fixture_payload = json.loads(
            transaction_fixture_path.read_text(encoding="utf-8")
        )
        fixture_version = str(fixture_payload["version"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid transaction fixture at {transaction_fixture_path}: {exc}"
        ) from exc
    if fixture_version != manifest.transaction_fixture_version:
        raise ValueError(
            "Transaction fixture version does not match the dataset manifest: "
            f"{fixture_version!r} != {manifest.transaction_fixture_version!r}"
        )

    counts = Counter(case.category for case in cases)
    missing = [
        category
        for category in CATEGORIES
        if counts[category] < manifest.minimum_questions_per_category
    ]
    if missing:
        raise ValueError(
            "Dataset does not meet minimum_questions_per_category for: "
            + ", ".join(missing)
        )

    sections, _ = load_sections(docs_path)
    by_citation = {unit.citation: unit for unit in sections}
    transaction_store = TransactionStore(transaction_fixture_path)
    resolved = [
        _validate_case_evidence(
            case,
            by_citation=by_citation,
            transaction_store=transaction_store,
            manifest=manifest,
        )
        for case in cases
    ]
    return EvaluationDataset(manifest=manifest, cases=resolved)


def _validate_case_evidence(
    case: EvalCase,
    *,
    by_citation: dict,
    transaction_store: TransactionStore,
    manifest: DatasetManifest,
) -> EvalCase:
    gold_by_citation = {source.citation: source for source in case.acceptable_sources}
    structured_context = None
    if case.booking_id:
        if case.transaction_fixture_version != manifest.transaction_fixture_version:
            raise ValueError(
                f"{case.id}: transaction version differs from dataset manifest"
            )
        structured_context = transaction_store.lookup(
            case.booking_id,
            as_of=case.as_of,
            expected_version=case.transaction_fixture_version,
        )
        if structured_context is None:
            raise ValueError(f"{case.id}: booking fixture does not resolve")

    # Every acceptable document source is resolved, not only the oracle subset.
    # An offset-only source outside oracle_sources previously kept
    # evidence_text None forever, so _evidence_hit could never match it and its
    # recall was silently capped for both RAG arms; an unresolvable citation
    # there also went unreported.
    resolved_sources: list[GoldSource] = []
    for gold in case.acceptable_sources:
        citation = gold.citation
        if not DOCUMENT_CITATION_RE.match(citation):
            resolved_sources.append(gold)
            continue
        unit = by_citation.get(citation)
        if unit is None:
            raise ValueError(f"{case.id}: unresolved document source {citation}")
        if gold.evidence_text is not None and gold.evidence_text not in unit.text:
            raise ValueError(
                f"{case.id}: evidence_text not found in source {citation}"
            )
        if gold.start_offset is not None:
            assert gold.end_offset is not None
            if gold.end_offset > len(unit.text):
                raise ValueError(
                    f"{case.id}: evidence offsets exceed source {citation}"
                )
            span = unit.text[gold.start_offset : gold.end_offset]
            if gold.evidence_text is not None and gold.evidence_text != span:
                raise ValueError(
                    f"{case.id}: evidence_text and offsets disagree for {citation}"
                )
            # GoldSource is frozen, so the backfill produces a copy.
            gold = gold.model_copy(update={"evidence_text": span})
        resolved_sources.append(gold)

    for citation in case.oracle_sources:
        if DOCUMENT_CITATION_RE.match(citation):
            if citation not in by_citation:
                raise ValueError(f"{case.id}: unresolved document source {citation}")
        elif structured_context is None or citation not in structured_context.references:
            raise ValueError(f"{case.id}: unresolved structured source {citation}")

    return case.model_copy(update={"acceptable_sources": resolved_sources})


# Spec 02 seed-dataset API, kept as thin wrappers over the unified validator so
# the seed freeze and the evaluation run cannot diverge.
FrozenDataset = EvaluationDataset


def load_frozen_dataset(
    manifest_path: Path,
    docs_path: Path,
    transaction_fixture_path: Path,
) -> EvaluationDataset:
    """Validate a frozen dataset from its manifest alone.

    Resolves the cases file through the manifest so a caller cannot pair a
    manifest with a different cases file.
    """
    manifest = load_manifest(manifest_path)
    return validate_dataset(
        manifest_path.parent / manifest.cases_file,
        manifest_path,
        docs_path=docs_path,
        transaction_fixture_path=transaction_fixture_path,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a frozen evaluation dataset without calling an "
        "LLM or a retriever"
    )
    parser.add_argument("--manifest", type=Path, default=Path("evals/manifest.json"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument(
        "--transactions", type=Path, default=Path("fixtures/bookings.json")
    )
    args = parser.parse_args()
    dataset = load_frozen_dataset(args.manifest, args.docs, args.transactions)
    print(
        f"Validated {dataset.manifest.dataset_version}: "
        f"{len(dataset.cases)} cases, corpus "
        f"{dataset.manifest.corpus_fingerprint[:8]}"
    )


if __name__ == "__main__":
    main()
