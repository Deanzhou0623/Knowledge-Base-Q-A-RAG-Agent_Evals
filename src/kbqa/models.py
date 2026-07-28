from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


FALLBACK_ANSWER = "I cannot confirm from the knowledge base."


class DocumentUnit(BaseModel):
    id: str
    source_path: str
    heading: str
    anchor: str
    citation: str
    text: str
    heading_level: int = Field(default=1, ge=0, le=6)
    heading_path: list[str] = Field(default_factory=list)
    chunk_index: int | None = None


class RetrievalResult(DocumentUnit):
    score: float
    rank: int


class StructuredContext(BaseModel):
    record_type: str
    record_id: str
    fixture_version: str
    fields: dict[str, str | bool | int | float | None]
    references: list[str]


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class IndexSummary(BaseModel):
    backend: Literal["bm25", "vector"]
    files_indexed: int
    units_indexed: int
    index_path: str
    corpus_fingerprint: str
    created_at: datetime


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    booking_id: str | None = Field(default=None, max_length=100)
    as_of: datetime | None = None
    expected_fixture_version: str | None = None
    requires_document_retrieval: bool = True

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must contain non-whitespace text")
        return query


class ChatResponse(BaseModel):
    answer: str
    backend: Literal["bm25", "vector"]
    sources: list[str]
    citations: list[str]
    retrieved: list[RetrievalResult]
    structured_context: StructuredContext | None = None
    retrieval_ms: float
    generation_ms: float
    model: str
    prompt_version: str
    token_usage: TokenUsage


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    backend: Literal["bm25", "vector"]
    index_loaded: bool
    corpus_fingerprint: str | None = None
