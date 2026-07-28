import pytest
from fastapi.testclient import TestClient

from kbqa.api import create_app
from kbqa.config import Settings
from kbqa.models import FALLBACK_ANSWER
from kbqa.retrievers.markdown_kb import BM25Retriever
from kbqa.retrievers.vector_rag import VectorRetriever
from kbqa.service import QAService
from kbqa.transactions import TransactionStore

from conftest import FakeEmbeddings, FakeGenerator


@pytest.mark.parametrize("backend", ["vector", "bm25"])
def test_ui_smoke_uses_the_same_shared_api_for_every_backend(
    backend, corpus, booking_fixture, tmp_path
):
    if backend == "vector":
        retriever = VectorRetriever(
            tmp_path / ".kb" / "faiss_index",
            FakeEmbeddings(),
            chunk_words=20,
            overlap=5,
        )
    else:
        retriever = BM25Retriever(tmp_path / ".kb" / "index.json")
    service = QAService(
        retriever,
        FakeGenerator(FALLBACK_ANSWER),
        TransactionStore(booking_fixture),
        corpus,
        top_k=3,
    )
    settings = Settings(
        retrieval_backend=backend,
        docs_path=corpus,
        kb_path=tmp_path / ".kb",
        transaction_fixture_path=booking_fixture,
    )

    with TestClient(create_app(settings, service)) as client:
        ui = client.get("/ui/")
        before = client.get("/health")
        indexed = client.post("/index")
        answer = client.post("/chat", json={"query": "How long does a refund take?"})

    assert ui.status_code == 200
    assert 'id="question-form"' in ui.text
    assert before.json()["backend"] == backend
    assert before.json()["index_loaded"] is False
    assert indexed.json()["backend"] == backend
    assert answer.status_code == 200
    assert answer.json()["backend"] == backend


def test_root_redirects_to_ui_and_static_assets_are_served(
    corpus, booking_fixture, tmp_path
):
    service = QAService(
        BM25Retriever(tmp_path / ".kb" / "index.json"),
        FakeGenerator(),
        TransactionStore(booking_fixture),
        corpus,
        top_k=3,
    )
    settings = Settings(
        docs_path=corpus,
        kb_path=tmp_path / ".kb",
        transaction_fixture_path=booking_fixture,
    )

    with TestClient(create_app(settings, service), follow_redirects=False) as client:
        root = client.get("/")
        script = client.get("/ui/app.js")
        styles = client.get("/ui/styles.css")

    assert root.status_code == 307
    assert root.headers["location"] == "/ui/"
    assert script.status_code == 200
    assert styles.status_code == 200
    assert "createController" in script.text
    assert "innerHTML" not in script.text
