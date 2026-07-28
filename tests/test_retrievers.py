import json

import pytest

from kbqa.io import corpus_fingerprint
from kbqa.retrievers.markdown_kb import BM25Retriever
from kbqa.retrievers.markdown_kb.bm25 import BM25_PARAMETERS, TOKENIZER_VERSION
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


def test_bm25_index_is_inspectable_and_rebuild_is_stable(corpus, tmp_path):
    index_path = tmp_path / ".kb" / "index.json"
    retriever = BM25Retriever(index_path)
    first_summary = retriever.build(corpus)
    first_payload = json.loads(index_path.read_text(encoding="utf-8"))
    second_summary = retriever.build(corpus)
    second_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert first_payload["schema_version"] == 2
    assert first_payload["backend"] == "bm25"
    assert first_payload["configuration"]["tokenizer_version"] == TOKENIZER_VERSION
    assert first_payload["configuration"]["bm25"] == {
        "implementation": "rank_bm25.BM25Okapi",
        **BM25_PARAMETERS,
    }
    assert first_payload["bm25_data"]["tokenized_corpus"]
    assert first_payload["units"][0]["heading_path"] == ["Refund Policy"]
    assert first_payload["units"] == second_payload["units"]
    assert first_payload["bm25_data"] == second_payload["bm25_data"]
    assert first_summary.corpus_fingerprint == second_summary.corpus_fingerprint
    assert first_summary.corpus_fingerprint == corpus_fingerprint(corpus)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(schema_version=999),
        lambda payload: payload.pop("bm25_data"),
        lambda payload: payload["configuration"].update(tokenizer_version="other"),
        lambda payload: payload["bm25_data"]["tokenized_corpus"][0].append("tampered"),
        lambda payload: payload["units"][0].update(citation="invented.md#heading"),
        lambda payload: payload["units"][0].update(source_path="../policy.md"),
        lambda payload: payload["units"][1].update(heading_path=["Wrong", "Cancellation"]),
        lambda payload: payload.update(corpus_fingerprint="not-a-fingerprint"),
        lambda payload: payload.update(created_at="2026-01-01T00:00:00"),
    ],
    ids=[
        "schema",
        "missing-data",
        "configuration",
        "tokens",
        "citation",
        "source-path",
        "hierarchy",
        "fingerprint",
        "timestamp",
    ],
)
def test_bm25_load_rejects_incompatible_or_incomplete_indexes(
    corpus, tmp_path, mutate
):
    index_path = tmp_path / ".kb" / "index.json"
    retriever = BM25Retriever(index_path)
    retriever.build(corpus)
    assert retriever.loaded is True
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    mutate(payload)
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    assert retriever.load() is False
    assert retriever.loaded is False
    assert retriever.units == []
    assert retriever.corpus_fingerprint is None


def test_bm25_load_rejects_corrupt_json(corpus, tmp_path):
    index_path = tmp_path / ".kb" / "index.json"
    retriever = BM25Retriever(index_path)
    retriever.build(corpus)
    index_path.write_text("{", encoding="utf-8")

    assert retriever.load() is False
    assert retriever.loaded is False


def test_bm25_failed_empty_rebuild_preserves_valid_index(corpus, tmp_path):
    index_path = tmp_path / ".kb" / "index.json"
    retriever = BM25Retriever(index_path)
    retriever.build(corpus)
    valid_index = index_path.read_bytes()
    empty_docs = tmp_path / "empty-docs"
    empty_docs.mkdir()

    with pytest.raises(ValueError, match="No Markdown heading sections"):
        retriever.build(empty_docs)

    assert index_path.read_bytes() == valid_index
    assert BM25Retriever(index_path).load() is True


def test_bm25_returns_three_ranked_raw_scores_with_deterministic_ties(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ["c.md", "a.md", "b.md", "d.md"]:
        (docs / name).write_text("# Policy\nSame terms.", encoding="utf-8")
    retriever = BM25Retriever(tmp_path / ".kb" / "index.json")
    retriever.build(docs)

    first = retriever.search("unmatched", 3)
    second = retriever.search("unmatched", 3)

    assert len(first) == 3
    assert [item.rank for item in first] == [1, 2, 3]
    assert all(item.score == 0.0 for item in first)
    assert [item.id for item in first] == sorted(item.id for item in first)
    assert [item.id for item in first] == [item.id for item in second]


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
