import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

from kbqa.io import atomic_write_json, corpus_fingerprint
from kbqa.models import DocumentUnit, IndexSummary, RetrievalResult
from kbqa.parsing import load_sections


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    backend = "bm25"
    schema_version = 1

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.units: list[DocumentUnit] = []
        self._index: BM25Okapi | None = None
        self.loaded = False
        self.corpus_fingerprint: str | None = None
        self.files_indexed = 0

    def _hydrate(self, units: list[DocumentUnit]) -> None:
        self.units = units
        corpus = [tokenize(f"{unit.heading} {unit.text}") for unit in units]
        self._index = BM25Okapi(corpus) if corpus else None
        self.loaded = bool(units)

    def build(self, docs_path: Path) -> IndexSummary:
        units, file_count = load_sections(docs_path)
        fingerprint = corpus_fingerprint(docs_path)
        created_at = datetime.now(timezone.utc)
        payload = {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "created_at": created_at.isoformat(),
            "corpus_fingerprint": fingerprint,
            "files_indexed": file_count,
            "units": [unit.model_dump() for unit in units],
        }
        atomic_write_json(self.index_path, payload)
        self.corpus_fingerprint = fingerprint
        self.files_indexed = file_count
        self._hydrate(units)
        return IndexSummary(
            backend="bm25",
            files_indexed=file_count,
            units_indexed=len(units),
            index_path=str(self.index_path),
            corpus_fingerprint=fingerprint,
            created_at=created_at,
        )

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != self.schema_version:
                return False
            if payload.get("backend") != self.backend:
                return False
            units = [DocumentUnit.model_validate(item) for item in payload["units"]]
            self.corpus_fingerprint = payload["corpus_fingerprint"]
            self.files_indexed = int(payload["files_indexed"])
            self._hydrate(units)
            return self.loaded
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.loaded = False
            return False

    def search(self, query: str, k: int = 3) -> list[RetrievalResult]:
        if not self.loaded or self._index is None:
            raise RuntimeError("BM25 index is not loaded")
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(range(len(self.units)), key=lambda i: (-float(scores[i]), self.units[i].id))
        return [
            RetrievalResult(**self.units[index].model_dump(), score=float(scores[index]), rank=rank)
            for rank, index in enumerate(ranked[:k], start=1)
        ]
