from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


FALLBACK_ANSWER = "I cannot confirm from the knowledge base."


class DocumentUnit(BaseModel):
    id: str
    source_path: str
    heading: str
    anchor: str
    citation: str
    text: str
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


class ChatResponse(BaseModel):
    answer: str
    backend: Literal["bm25", "vector"]
    sources: list[str]
    citations: list[str]
    retrieved: list[RetrievalResult]
    structured_context: StructuredContext | None = None
    retrieval_ms: float
    generation_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    backend: Literal["bm25", "vector"]
    index_loaded: bool
    corpus_fingerprint: str | None = None
