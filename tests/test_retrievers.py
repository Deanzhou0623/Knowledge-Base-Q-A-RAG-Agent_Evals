import json

import numpy as np
import pytest

from kbqa.retrievers.markdown_kb import BM25Retriever
from kbqa.retrievers.vector_rag import VectorRetriever

from conftest import FakeEmbeddings


def test_bm25_build_search_and_reload(corpus, tmp_path):
    index_path = tmp_path / ".kb" / "index.json"
    retriever = BM25Retriever(index_path)
    summary = retriever.build(corpus)

    results = retriever.search("refund business days", 3)
    restored = BM25Retriever(index_path)

    assert summary.units_indexed == 2
    assert results[0].citation == "policy.md#refund-policy"
    assert restored.load() is True
    assert [item.citation for item in restored.search("refund business days", 3)] == [
        item.citation for item in results
    ]


def test_vector_build_search_reload_and_checksum_validation(corpus, tmp_path):
    index_path = tmp_path / ".kb" / "faiss_index"
    embeddings = FakeEmbeddings()
    retriever = VectorRetriever(index_path, embeddings, chunk_words=20, overlap=5)
    summary = retriever.build(corpus)

    assert summary.units_indexed == 2
    assert len(retriever.search("refund business days", 3)) == 2

    restored = VectorRetriever(index_path, embeddings, chunk_words=20, overlap=5)
    assert restored.load() is True

    with restored.faiss_path.open("ab") as handle:
        handle.write(b"tampered")
    rejected = VectorRetriever(index_path, embeddings, chunk_words=20, overlap=5)
    assert rejected.load() is False


class CountingEmbeddings(FakeEmbeddings):
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return super().embed_documents(texts)

    def embed_query(self, text):
        self.query_calls += 1
        return super().embed_query(text)


def test_vector_metadata_records_complete_configuration_and_row_mapping(
    corpus, tmp_path
):
    index_path = tmp_path / ".kb" / "faiss_index"
    retriever = VectorRetriever(
        index_path, FakeEmbeddings(), chunk_words=20, overlap=5
    )
    retriever.build(corpus)

    metadata = json.loads(retriever.metadata_path.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == 2
    assert metadata["embedding_model"] == "fake-embeddings-v1"
    assert metadata["dimension"] == 16
    assert metadata["normalization"] == "l2"
    assert metadata["distance_metric"] == "inner_product"
    assert metadata["index_type"] == "IndexFlatIP"
    assert metadata["chunk_words"] == 20
    assert metadata["overlap"] == 5
    assert len(metadata["units"]) == retriever._index.ntotal
    assert all(
        unit["citation"] == f"{unit['source_path']}#{unit['anchor']}"
        and unit["source_path"].endswith(".md")
        and unit["text"]
        for unit in metadata["units"]
    )


def test_vector_restart_matches_results_without_document_embedding(corpus, tmp_path):
    index_path = tmp_path / ".kb" / "faiss_index"
    initial_embeddings = CountingEmbeddings()
    initial = VectorRetriever(
        index_path, initial_embeddings, chunk_words=20, overlap=5
    )
    initial.build(corpus)
    expected = initial.search("refund business days", 3)

    restored_embeddings = CountingEmbeddings()
    restored = VectorRetriever(
        index_path, restored_embeddings, chunk_words=20, overlap=5
    )

    assert restored.load() is True
    assert restored_embeddings.document_calls == 0
    actual = restored.search("refund business days", 3)
    assert restored_embeddings.document_calls == 0
    assert restored_embeddings.query_calls == 1
    assert [item.model_dump() for item in actual] == [
        item.model_dump() for item in expected
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 999),
        ("embedding_model", "different-model"),
        ("created_at", None),
        ("corpus_fingerprint", "not-a-sha256"),
        ("normalization", "none"),
        ("distance_metric", "l2"),
        ("index_type", "IndexFlatL2"),
        ("chunk_words", 21),
        ("overlap", 4),
    ],
)
def test_vector_load_rejects_incompatible_metadata_and_clears_state(
    field, value, corpus, tmp_path
):
    index_path = tmp_path / ".kb" / "faiss_index"
    retriever = VectorRetriever(
        index_path, FakeEmbeddings(), chunk_words=20, overlap=5
    )
    retriever.build(corpus)
    metadata = json.loads(retriever.metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    retriever.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert retriever.load() is False
    assert retriever.loaded is False
    assert retriever.units == []
    assert retriever.corpus_fingerprint is None


def test_vector_load_rejects_invalid_row_metadata(corpus, tmp_path):
    index_path = tmp_path / ".kb" / "faiss_index"
    retriever = VectorRetriever(
        index_path, FakeEmbeddings(), chunk_words=20, overlap=5
    )
    retriever.build(corpus)
    metadata = json.loads(retriever.metadata_path.read_text(encoding="utf-8"))
    metadata["units"][0]["citation"] = "other.md#invented"
    retriever.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    restored = VectorRetriever(
        index_path, FakeEmbeddings(), chunk_words=20, overlap=5
    )
    assert restored.load() is False


class MalformedDocumentEmbeddings(FakeEmbeddings):
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_documents(self, texts):
        return self.vectors


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((1, 16), dtype="float32"),
        np.ones((2, 0), dtype="float32"),
        np.zeros((2, 16), dtype="float32"),
        np.full((2, 16), np.nan, dtype="float32"),
    ],
)
def test_vector_build_rejects_invalid_document_embeddings(
    vectors, corpus, tmp_path
):
    retriever = VectorRetriever(
        tmp_path / ".kb" / "faiss_index",
        MalformedDocumentEmbeddings(vectors),
        chunk_words=20,
        overlap=5,
    )

    with pytest.raises(ValueError, match="document embeddings"):
        retriever.build(corpus)

    assert retriever.loaded is False
    assert not retriever.metadata_path.exists()


class WrongDimensionQueryEmbeddings(FakeEmbeddings):
    def embed_query(self, text):
        return np.ones(3, dtype="float32")


def test_vector_search_rejects_query_dimension_mismatch(corpus, tmp_path):
    retriever = VectorRetriever(
        tmp_path / ".kb" / "faiss_index",
        WrongDimensionQueryEmbeddings(),
        chunk_words=20,
        overlap=5,
    )
    retriever.build(corpus)

    with pytest.raises(ValueError, match="dimension"):
        retriever.search("refund", 3)


class ConstantEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts):
        return np.ones((len(texts), 4), dtype="float32")

    def embed_query(self, text):
        return np.ones(4, dtype="float32")


def test_vector_search_resolves_score_ties_by_stable_chunk_id(corpus, tmp_path):
    retriever = VectorRetriever(
        tmp_path / ".kb" / "faiss_index",
        ConstantEmbeddings(),
        chunk_words=20,
        overlap=5,
    )
    retriever.build(corpus)

    results = retriever.search("anything", 3)

    assert [result.id for result in results] == sorted(
        unit.id for unit in retriever.units
    )
    assert [result.rank for result in results] == [1, 2]
    assert all(result.score == pytest.approx(1.0) for result in results)
