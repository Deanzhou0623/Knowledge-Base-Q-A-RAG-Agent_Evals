import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from kbqa.embeddings import EmbeddingProvider
from kbqa.io import atomic_write_json, corpus_fingerprint, sha256_file
from kbqa.models import DocumentUnit, IndexSummary, RetrievalResult
from kbqa.parsing import chunk_sections, load_sections


class VectorRetriever:
    backend = "vector"
    schema_version = 1

    def __init__(
        self,
        index_path: Path,
        embeddings: EmbeddingProvider,
        chunk_words: int = 160,
        overlap: int = 30,
    ) -> None:
        self.index_path = index_path
        self.embeddings = embeddings
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.units: list[DocumentUnit] = []
        self._index: faiss.Index | None = None
        self.loaded = False
        self.corpus_fingerprint: str | None = None
        self.files_indexed = 0

    @property
    def faiss_path(self) -> Path:
        if self.metadata_path.exists():
            try:
                index_file = json.loads(
                    self.metadata_path.read_text(encoding="utf-8")
                ).get("index_file")
                if isinstance(index_file, str) and Path(index_file).name == index_file:
                    return self.index_path / index_file
            except (OSError, json.JSONDecodeError):
                pass
        return self.index_path / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_path / "metadata.json"

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        faiss.normalize_L2(matrix)
        return matrix

    def build(self, docs_path: Path) -> IndexSummary:
        sections, file_count = load_sections(docs_path)
        units = chunk_sections(sections, self.chunk_words, self.overlap)
        if not units:
            raise ValueError("No Markdown content found to index")
        vectors = self._normalize(self.embeddings.embed_documents([unit.text for unit in units]))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.index_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.index_path, delete=False) as handle:
            temporary_index = Path(handle.name)
        try:
            faiss.write_index(index, str(temporary_index))
            index_hash = sha256_file(temporary_index)
            final_index = self.index_path / f"index-{index_hash[:16]}.faiss"
            os.replace(temporary_index, final_index)
        finally:
            if temporary_index.exists():
                temporary_index.unlink()

        fingerprint = corpus_fingerprint(docs_path)
        created_at = datetime.now(timezone.utc)
        metadata = {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "created_at": created_at.isoformat(),
            "corpus_fingerprint": fingerprint,
            "files_indexed": file_count,
            "embedding_model": self.embeddings.model,
            "dimension": int(vectors.shape[1]),
            "distance_metric": "inner_product_on_l2_normalized_vectors",
            "index_type": "IndexFlatIP",
            "chunk_words": self.chunk_words,
            "overlap": self.overlap,
            "faiss_sha256": index_hash,
            "index_file": final_index.name,
            "units": [unit.model_dump() for unit in units],
        }
        atomic_write_json(self.metadata_path, metadata)
        self._index = index
        self.units = units
        self.loaded = True
        self.corpus_fingerprint = fingerprint
        self.files_indexed = file_count
        return IndexSummary(
            backend="vector",
            files_indexed=file_count,
            units_indexed=len(units),
            index_path=str(self.index_path),
            corpus_fingerprint=fingerprint,
            created_at=created_at,
        )

    def load(self) -> bool:
        if not self.faiss_path.exists() or not self.metadata_path.exists():
            return False
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            compatible = (
                metadata.get("schema_version") == self.schema_version
                and metadata.get("backend") == self.backend
                and metadata.get("embedding_model") == self.embeddings.model
                and metadata.get("chunk_words") == self.chunk_words
                and metadata.get("overlap") == self.overlap
                and metadata.get("faiss_sha256") == sha256_file(self.faiss_path)
            )
            if not compatible:
                return False
            units = [DocumentUnit.model_validate(item) for item in metadata["units"]]
            index = faiss.read_index(str(self.faiss_path))
            if index.ntotal != len(units) or index.d != int(metadata["dimension"]):
                return False
            self.units = units
            self._index = index
            self.loaded = bool(units)
            self.corpus_fingerprint = metadata["corpus_fingerprint"]
            self.files_indexed = int(metadata["files_indexed"])
            return self.loaded
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
            self.loaded = False
            return False

    def search(self, query: str, k: int = 3) -> list[RetrievalResult]:
        if not self.loaded or self._index is None:
            raise RuntimeError("Vector index is not loaded")
        query_vector = self._normalize(self.embeddings.embed_query(query))
        scores, indices = self._index.search(query_vector, min(k, len(self.units)))
        results: list[RetrievalResult] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if index < 0:
                continue
            results.append(
                RetrievalResult(
                    **self.units[int(index)].model_dump(), score=float(score), rank=rank
                )
            )
        return results
