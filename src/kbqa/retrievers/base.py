from pathlib import Path
from typing import Protocol

from kbqa.models import IndexSummary, RetrievalResult


class Retriever(Protocol):
    backend: str
    loaded: bool
    corpus_fingerprint: str | None

    def build(self, docs_path: Path) -> IndexSummary: ...

    def load(self) -> bool: ...

    def search(self, query: str, k: int = 3) -> list[RetrievalResult]: ...
