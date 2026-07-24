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
