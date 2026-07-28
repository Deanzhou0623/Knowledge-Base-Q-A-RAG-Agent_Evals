from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kbqa.embeddings import OPENAI_EMBEDDING_MODEL
from kbqa.models import ANSWER_MODEL, SHARED_TOP_K


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    openai_chat_model: Literal["gpt-5.6-sol"] = ANSWER_MODEL
    openai_embedding_model: Literal["text-embedding-3-small"] = OPENAI_EMBEDDING_MODEL
    retrieval_backend: Literal["bm25", "vector"] = "bm25"
    docs_path: Path = Path("docs")
    kb_path: Path = Path(".kb")
    transaction_fixture_path: Path = Path("fixtures/bookings.json")
    # Typed as int rather than Literal[3] so a string environment value such
    # as TOP_K=3 still parses; the shared K = 3 contract is enforced below.
    top_k: int = SHARED_TOP_K
    vector_chunk_words: int = Field(default=160, ge=20)
    vector_chunk_overlap: int = Field(default=30, ge=0)
    max_output_tokens: int = Field(default=500, ge=1)

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.vector_chunk_overlap >= self.vector_chunk_words:
            raise ValueError("VECTOR_CHUNK_OVERLAP must be smaller than VECTOR_CHUNK_WORDS")
        return self

    @model_validator(mode="after")
    def validate_top_k(self) -> "Settings":
        if self.top_k != SHARED_TOP_K:
            raise ValueError(f"The shared Q&A contract requires top_k={SHARED_TOP_K}")
        return self

    @property
    def bm25_index_path(self) -> Path:
        return self.kb_path / "index.json"

    @property
    def vector_index_path(self) -> Path:
        return self.kb_path / "faiss_index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
