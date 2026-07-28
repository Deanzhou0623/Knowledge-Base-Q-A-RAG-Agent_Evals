import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from kbqa.io import atomic_write_json, corpus_fingerprint
from kbqa.models import DocumentUnit, IndexSummary, RetrievalResult
from kbqa.parsing import PARSER_VERSION, load_sections, slugify_heading, stable_unit_id


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
TOKENIZER_VERSION = "unicode-word-lower-v1"
BM25_PARAMETERS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    backend = "bm25"
    schema_version = 2

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.units: list[DocumentUnit] = []
        self._index: BM25Okapi | None = None
        self.loaded = False
        self.corpus_fingerprint: str | None = None
        self.files_indexed = 0

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "parser_version": PARSER_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "token_pattern": TOKEN_RE.pattern,
            "indexed_fields": ["heading", "text"],
            "bm25": {
                "implementation": "rank_bm25.BM25Okapi",
                **BM25_PARAMETERS,
            },
        }

    @staticmethod
    def _tokenized_corpus(units: list[DocumentUnit]) -> list[list[str]]:
        return [tokenize(f"{unit.heading} {unit.text}") for unit in units]

    @staticmethod
    def _make_index(tokenized_corpus: list[list[str]]) -> BM25Okapi:
        return BM25Okapi(tokenized_corpus, **BM25_PARAMETERS)

    def _clear(self) -> None:
        self.units = []
        self._index = None
        self.loaded = False
        self.corpus_fingerprint = None
        self.files_indexed = 0

    def _hydrate(
        self, units: list[DocumentUnit], tokenized_corpus: list[list[str]]
    ) -> None:
        self.units = units
        self._index = self._make_index(tokenized_corpus) if tokenized_corpus else None
        self.loaded = bool(units)

    def build(self, docs_path: Path) -> IndexSummary:
        units, file_count = load_sections(docs_path)
        if not units:
            raise ValueError("No Markdown heading sections found to index")
        tokenized_corpus = self._tokenized_corpus(units)
        index = self._make_index(tokenized_corpus)
        fingerprint = corpus_fingerprint(docs_path)
        created_at = datetime.now(timezone.utc)
        payload = {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "created_at": created_at.isoformat(),
            "corpus_fingerprint": fingerprint,
            "files_indexed": file_count,
            "configuration": self.configuration,
            "bm25_data": {"tokenized_corpus": tokenized_corpus},
            "units": [unit.model_dump() for unit in units],
        }
        atomic_write_json(self.index_path, payload)
        self.units = units
        self._index = index
        self.loaded = True
        self.corpus_fingerprint = fingerprint
        self.files_indexed = file_count
        return IndexSummary(
            backend="bm25",
            files_indexed=file_count,
            units_indexed=len(units),
            index_path=str(self.index_path),
            corpus_fingerprint=fingerprint,
            created_at=created_at,
        )

    @staticmethod
    def _validate_units(units: list[DocumentUnit]) -> int:
        if not units:
            raise ValueError("Index contains no sections")
        seen_ids: set[str] = set()
        duplicate_counts: dict[tuple[str, str], int] = {}
        hierarchy_by_source: dict[str, list[tuple[int, str]]] = {}
        source_paths: list[str] = []
        for unit in units:
            source = Path(unit.source_path)
            if (
                source.is_absolute()
                or ".." in source.parts
                or source.suffix.lower() != ".md"
            ):
                raise ValueError("Index contains an invalid source path")
            source_paths.append(unit.source_path)
            if unit.chunk_index is not None:
                raise ValueError("BM25 index contains a vector chunk")
            if not unit.heading:
                raise ValueError("Index contains an empty normalized heading")
            hierarchy = hierarchy_by_source.setdefault(unit.source_path, [])
            if unit.heading_level == 0:
                expected_path = ["Document"]
                if unit.heading != "Document" or hierarchy:
                    raise ValueError("Index contains an invalid document preamble")
            else:
                while hierarchy and hierarchy[-1][0] >= unit.heading_level:
                    hierarchy.pop()
                hierarchy.append((unit.heading_level, unit.heading))
                expected_path = [title for _, title in hierarchy]
            if unit.heading_path != expected_path:
                raise ValueError("Index contains an invalid heading hierarchy")
            base_anchor = slugify_heading(unit.heading)
            duplicate_key = (unit.source_path, base_anchor)
            duplicate_index = duplicate_counts.get(duplicate_key, 0)
            duplicate_counts[duplicate_key] = duplicate_index + 1
            expected_anchor = (
                base_anchor
                if duplicate_index == 0
                else f"{base_anchor}-{duplicate_index}"
            )
            if unit.anchor != expected_anchor:
                raise ValueError("Index contains an invalid heading anchor")
            if unit.citation != f"{unit.source_path}#{unit.anchor}":
                raise ValueError("Index contains an invalid citation")
            if unit.id != stable_unit_id(unit.source_path, unit.anchor):
                raise ValueError("Index contains an invalid section ID")
            if unit.id in seen_ids:
                raise ValueError("Index contains duplicate section IDs")
            seen_ids.add(unit.id)
        if source_paths != sorted(source_paths):
            raise ValueError("Index sections are not in deterministic file order")
        return len(set(source_paths))

    def load(self) -> bool:
        self._clear()
        if not self.index_path.exists():
            return False
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != self.schema_version:
                return False
            if payload.get("backend") != self.backend:
                return False
            if payload.get("configuration") != self.configuration:
                return False
            units = [DocumentUnit.model_validate(item) for item in payload["units"]]
            source_count = self._validate_units(units)
            tokenized_corpus = payload["bm25_data"]["tokenized_corpus"]
            if tokenized_corpus != self._tokenized_corpus(units):
                return False
            fingerprint = payload["corpus_fingerprint"]
            if (
                not isinstance(fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
            ):
                return False
            files_indexed = payload["files_indexed"]
            if (
                not isinstance(files_indexed, int)
                or isinstance(files_indexed, bool)
                or files_indexed < source_count
            ):
                return False
            created_at = datetime.fromisoformat(payload["created_at"])
            if created_at.tzinfo is None:
                return False
            self.corpus_fingerprint = fingerprint
            self.files_indexed = files_indexed
            self._hydrate(units, tokenized_corpus)
            return self.loaded
        except (
            KeyError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            self._clear()
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
