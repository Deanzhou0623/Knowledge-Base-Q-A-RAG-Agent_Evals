import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GoldSource(BaseModel):
    citation: str
    evidence_text: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None

    @model_validator(mode="after")
    def require_evidence_for_documents(self) -> "GoldSource":
        if ".md#" in self.citation and self.evidence_text is None:
            raise ValueError(
                "This scaffold requires evidence_text for document gold sources; "
                "offset grading is reserved for a source-offset-aware implementation"
            )
        return self


class EvalCase(BaseModel):
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

    @model_validator(mode="after")
    def validate_case_contract(self) -> "EvalCase":
        if self.category == "user_specific":
            if self.requires_document_retrieval is None or self.booking_id is None:
                raise ValueError(
                    "user_specific cases require requires_document_retrieval and booking_id"
                )
        if self.answerable and not self.oracle_sources:
            raise ValueError("Answerable cases require oracle_sources")
        if not self.answerable and self.expected_facts:
            raise ValueError("Unsupported cases cannot define expected facts")
        return self


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid evaluation case at {path}:{line_number}: {exc}") from exc
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique")
    return cases
