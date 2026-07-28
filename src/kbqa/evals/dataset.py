import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kbqa.io import corpus_fingerprint, sha256_file
from kbqa.models import FALLBACK_ANSWER, DocumentUnit
from kbqa.parsing import load_sections
from kbqa.transactions import TransactionStore


DOCUMENT_CITATION_RE = re.compile(r"^(?P<path>[^#]+\.md)#(?P<anchor>[^#]+)$")
STRUCTURED_CITATION_RE = re.compile(
    r"^(?P<record_type>[a-z][a-z0-9_-]*):(?P<record_id>[^#]+)#(?P<field>[a-z][a-z0-9_]*)$"
)


class GoldSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation: str
    evidence_text: str | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)

    @field_validator("citation")
    @classmethod
    def citation_must_be_canonical(cls, value: str) -> str:
        citation = value.strip()
        if not (
            DOCUMENT_CITATION_RE.fullmatch(citation)
            or STRUCTURED_CITATION_RE.fullmatch(citation)
        ):
            raise ValueError("citation must be a document or structured-record reference")
        return citation

    @model_validator(mode="after")
    def validate_evidence_span(self) -> "GoldSource":
        is_document = DOCUMENT_CITATION_RE.fullmatch(self.citation) is not None
        has_offsets = self.start_offset is not None or self.end_offset is not None
        if has_offsets and (self.start_offset is None or self.end_offset is None):
            raise ValueError("evidence offsets require both start_offset and end_offset")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("end_offset must be greater than start_offset")
        if is_document and self.evidence_text is None and not has_offsets:
            raise ValueError(
                "document gold sources require evidence_text or character offsets"
            )
        if not is_document and (
            self.evidence_text is not None or self.start_offset is not None
        ):
            raise ValueError("structured-record sources cannot define document evidence")
        return self


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Literal[
        "company_specific", "generic_ecommerce", "user_specific", "unsupported"
    ]
    question: str
    answerable: bool
    expected_facts: list[str] = Field(default_factory=list)
    acceptable_sources: list[GoldSource] = Field(default_factory=list)
    oracle_sources: list[str] = Field(default_factory=list)
    paraphrase_group_id: str | None = None
    requires_document_retrieval: bool | None = None
    booking_id: str | None = None
    as_of: datetime | None = None
    transaction_fixture_version: str | None = None
    expected_fallback: str | None = None

    @field_validator("id", "question")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must contain non-whitespace text")
        return stripped

    @field_validator("expected_facts", "oracle_sources")
    @classmethod
    def list_values_must_be_unique_and_nonblank(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("list values must contain non-whitespace text")
        if len(set(normalized)) != len(normalized):
            raise ValueError("list values must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_case_contract(self) -> "EvalCase":
        source_ids = [source.citation for source in self.acceptable_sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("acceptable source citations must be unique")

        if self.category == "user_specific":
            if (
                self.requires_document_retrieval is None
                or self.booking_id is None
                or self.transaction_fixture_version is None
            ):
                raise ValueError(
                    "user_specific cases require requires_document_retrieval, "
                    "booking_id, and transaction_fixture_version"
                )
            structured = [
                citation
                for citation in source_ids
                if STRUCTURED_CITATION_RE.fullmatch(citation)
            ]
            documents = [
                citation
                for citation in source_ids
                if DOCUMENT_CITATION_RE.fullmatch(citation)
            ]
            if not structured:
                raise ValueError("user_specific cases require structured-record evidence")
            if self.requires_document_retrieval and not documents:
                raise ValueError(
                    "transaction-plus-document cases require document evidence"
                )
            if not self.requires_document_retrieval and documents:
                raise ValueError(
                    "transaction-only cases cannot define document evidence"
                )
        elif any(
            value is not None
            for value in (
                self.requires_document_retrieval,
                self.booking_id,
                self.as_of,
                self.transaction_fixture_version,
            )
        ):
            raise ValueError("transaction fields are only valid for user_specific cases")

        if self.answerable:
            if not self.expected_facts:
                raise ValueError("Answerable cases require expected_facts")
            if not self.acceptable_sources:
                raise ValueError("Answerable cases require acceptable_sources")
            if not self.oracle_sources:
                raise ValueError("Answerable cases require oracle_sources")
            unknown_oracle = set(self.oracle_sources) - set(source_ids)
            if unknown_oracle:
                raise ValueError(
                    "oracle_sources must be selected from acceptable_sources: "
                    + ", ".join(sorted(unknown_oracle))
                )
            if self.expected_fallback is not None:
                raise ValueError("Answerable cases cannot expect fallback")
        else:
            if self.category != "unsupported":
                raise ValueError("Only unsupported cases may be unanswerable")
            if self.expected_facts or self.acceptable_sources or self.oracle_sources:
                raise ValueError("Unsupported cases cannot define gold evidence")
            if self.expected_fallback != FALLBACK_ANSWER:
                raise ValueError("Unsupported cases must expect the exact fallback")

        if self.category == "unsupported" and self.answerable:
            raise ValueError("unsupported cases must be unanswerable")
        return self


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    dataset_id: str
    dataset_version: str
    status: Literal["frozen"]
    annotation_date: date
    cases_file: str
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    transaction_fixture_version: str
    oracle_evidence_selected_before_outputs: Literal[True]
    change_reason: str

    @field_validator(
        "dataset_id",
        "dataset_version",
        "cases_file",
        "transaction_fixture_version",
        "change_reason",
    )
    @classmethod
    def manifest_text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("manifest values must contain non-whitespace text")
        return stripped

    @field_validator("cases_file")
    @classmethod
    def cases_file_must_be_relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("cases_file must be a safe relative path")
        return path.as_posix()


class FrozenDataset(BaseModel):
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
        raise ValueError("Evaluation dataset must contain at least one case")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique")
    return cases


def _resolve_document_source(
    source: GoldSource, sections: dict[str, DocumentUnit]
) -> None:
    section = sections.get(source.citation)
    if section is None:
        raise ValueError(f"Document citation does not resolve: {source.citation}")
    if source.evidence_text is not None and source.evidence_text not in section.text:
        raise ValueError(
            f"Evidence text does not resolve within {source.citation}: "
            f"{source.evidence_text!r}"
        )
    if source.start_offset is not None and source.end_offset is not None:
        if source.end_offset > len(section.text):
            raise ValueError(f"Evidence offsets exceed source text: {source.citation}")
        if (
            source.evidence_text is not None
            and section.text[source.start_offset : source.end_offset]
            != source.evidence_text
        ):
            raise ValueError(f"Evidence text and offsets disagree: {source.citation}")


def _resolve_structured_source(
    case: EvalCase, source: GoldSource, fixture_path: Path
) -> None:
    match = STRUCTURED_CITATION_RE.fullmatch(source.citation)
    if match is None:
        return
    if match.group("record_type") != "booking":
        raise ValueError(f"Unsupported structured record type: {source.citation}")
    if case.booking_id != match.group("record_id"):
        raise ValueError(f"Structured source does not match case booking: {source.citation}")
    context = TransactionStore(fixture_path).lookup(
        case.booking_id,
        as_of=case.as_of,
        expected_version=case.transaction_fixture_version,
    )
    if context is None or source.citation not in context.references:
        raise ValueError(f"Structured citation does not resolve: {source.citation}")


def validate_cases(
    cases: list[EvalCase], docs_path: Path, transaction_fixture_path: Path
) -> None:
    categories = {case.category for case in cases}
    required_categories = {
        "company_specific",
        "generic_ecommerce",
        "user_specific",
        "unsupported",
    }
    if categories != required_categories:
        missing = ", ".join(sorted(required_categories - categories))
        raise ValueError(f"Evaluation dataset is missing required categories: {missing}")

    groups: dict[str, list[EvalCase]] = {}
    for case in cases:
        if case.paraphrase_group_id:
            groups.setdefault(case.paraphrase_group_id, []).append(case)
    if not any(
        len(group) >= 2
        and all(
            case.answerable
            and any(
                DOCUMENT_CITATION_RE.fullmatch(source.citation)
                for source in case.acceptable_sources
            )
            for case in group
        )
        for group in groups.values()
    ):
        raise ValueError("Evaluation dataset requires an answerable policy paraphrase pair")

    sections, _ = load_sections(docs_path)
    by_citation = {section.citation: section for section in sections}
    for case in cases:
        by_source = {source.citation: source for source in case.acceptable_sources}
        for source in case.acceptable_sources:
            if DOCUMENT_CITATION_RE.fullmatch(source.citation):
                _resolve_document_source(source, by_citation)
            else:
                _resolve_structured_source(case, source, transaction_fixture_path)
        for citation in case.oracle_sources:
            source = by_source[citation]
            if DOCUMENT_CITATION_RE.fullmatch(citation):
                _resolve_document_source(source, by_citation)
            else:
                _resolve_structured_source(case, source, transaction_fixture_path)


def load_frozen_dataset(
    manifest_path: Path, docs_path: Path, transaction_fixture_path: Path
) -> FrozenDataset:
    try:
        manifest = DatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset manifest at {manifest_path}: {exc}") from exc

    cases_path = manifest_path.parent / manifest.cases_file
    if not cases_path.is_file():
        raise ValueError(f"Dataset cases file does not exist: {cases_path}")
    actual_cases_hash = sha256_file(cases_path)
    if actual_cases_hash != manifest.cases_sha256:
        raise ValueError(
            f"Frozen cases hash mismatch: expected {manifest.cases_sha256}, "
            f"found {actual_cases_hash}"
        )

    actual_fingerprint = corpus_fingerprint(docs_path)
    if actual_fingerprint != manifest.corpus_fingerprint:
        raise ValueError(
            f"Corpus fingerprint mismatch: expected {manifest.corpus_fingerprint}, "
            f"found {actual_fingerprint}"
        )

    fixture_payload = json.loads(transaction_fixture_path.read_text(encoding="utf-8"))
    actual_fixture_version = str(fixture_payload.get("version", ""))
    if actual_fixture_version != manifest.transaction_fixture_version:
        raise ValueError(
            "Transaction fixture version mismatch: "
            f"expected {manifest.transaction_fixture_version}, "
            f"found {actual_fixture_version}"
        )

    cases = load_cases(cases_path)
    user_versions = {
        case.transaction_fixture_version
        for case in cases
        if case.category == "user_specific"
    }
    if user_versions != {manifest.transaction_fixture_version}:
        raise ValueError(
            "User-specific cases must use the manifest transaction fixture version"
        )
    validate_cases(cases, docs_path, transaction_fixture_path)
    return FrozenDataset(manifest=manifest, cases=cases)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a frozen evaluation dataset without a model or retriever"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("evals/seed-v1.manifest.json")
    )
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument(
        "--transactions", type=Path, default=Path("fixtures/bookings.json")
    )
    args = parser.parse_args()
    dataset = load_frozen_dataset(args.manifest, args.docs, args.transactions)
    print(
        f"Validated {dataset.manifest.dataset_version}: "
        f"{len(dataset.cases)} cases, "
        f"corpus {dataset.manifest.corpus_fingerprint}"
    )


if __name__ == "__main__":
    main()
