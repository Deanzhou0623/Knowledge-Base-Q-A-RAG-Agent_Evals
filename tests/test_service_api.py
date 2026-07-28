import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from kbqa.api import create_app
from kbqa.config import Settings
from kbqa.models import FALLBACK_ANSWER, ChatRequest
from kbqa.retrievers.markdown_kb import BM25Retriever
from kbqa.retrievers.vector_rag import VectorRetriever
from kbqa.service import QAService
from kbqa.transactions import TransactionStore

from conftest import FakeEmbeddings, FakeGenerator


def make_service(
    corpus,
    booking_fixture,
    tmp_path,
    *,
    backend="bm25",
    answer=FALLBACK_ANSWER,
):
    if backend == "bm25":
        retriever = BM25Retriever(tmp_path / ".kb" / "index.json")
    else:
        retriever = VectorRetriever(
            tmp_path / ".kb" / "faiss_index",
            FakeEmbeddings(),
            chunk_words=20,
            overlap=5,
        )
    generator = FakeGenerator(answer)
    service = QAService(
        retriever,
        generator,
        TransactionStore(booking_fixture),
        corpus,
        top_k=3,
    )
    return service, generator


@pytest.fixture(params=["bm25", "vector"])
def backend(request):
    return request.param


def settings_for(backend, corpus, booking_fixture):
    return Settings(
        retrieval_backend=backend,
        docs_path=corpus,
        transaction_fixture_path=booking_fixture,
    )


def test_chat_rejects_unbuilt_index_without_calling_model(
    backend, corpus, booking_fixture, tmp_path
):
    service, generator = make_service(
        corpus, booking_fixture, tmp_path, backend=backend
    )
    settings = settings_for(backend, corpus, booking_fixture)

    with TestClient(create_app(settings, service)) as client:
        response = client.post("/chat", json={"query": "Refund timing?"})

    assert response.status_code == 409
    assert generator.grounded_calls == 0


def test_grounded_chat_accepts_only_supplied_citations(
    backend, corpus, booking_fixture, tmp_path
):
    valid = "Refunds take 7 to 11 business days. [policy.md#refund-policy]"
    service, _ = make_service(
        corpus, booking_fixture, tmp_path, backend=backend, answer=valid
    )
    service.build_index()

    response = service.chat(ChatRequest(query="How long does a refund take?"))
    assert response.answer == valid
    assert response.citations == ["policy.md#refund-policy"]
    assert response.model == "fake-answer-model"
    assert response.prompt_version == "grounded-v1"
    assert response.token_usage.total_tokens == 14

    service.generator.answer = "Tomorrow. [invented.md#answer]"
    response = service.chat(ChatRequest(query="How long does a refund take?"))
    assert response.answer == FALLBACK_ANSWER
    assert response.citations == []


def test_transaction_only_chat_does_not_require_an_index(
    backend, corpus, booking_fixture, tmp_path
):
    answer = "The booking is confirmed. [booking:BK-1#status]"
    service, _ = make_service(
        corpus, booking_fixture, tmp_path, backend=backend, answer=answer
    )

    response = service.chat(
        ChatRequest(
            query="What is my status?",
            booking_id="BK-1",
            expected_fixture_version="v1",
            requires_document_retrieval=False,
        )
    )

    assert response.retrieved == []
    assert response.citations == ["booking:BK-1#status"]


def test_startup_load_rejects_an_index_for_changed_documents(
    backend, corpus, booking_fixture, tmp_path
):
    service, _ = make_service(
        corpus, booking_fixture, tmp_path, backend=backend
    )
    service.build_index()
    (corpus / "policy.md").write_text("# Changed\nNew policy", encoding="utf-8")

    restarted, _ = make_service(
        corpus, booking_fixture, tmp_path, backend=backend
    )
    assert restarted.load() is False
    assert restarted.health().index_loaded is False


def test_health_and_index_endpoints_share_the_contract(
    backend, corpus, booking_fixture, tmp_path
):
    service, _ = make_service(
        corpus, booking_fixture, tmp_path, backend=backend
    )
    settings = settings_for(backend, corpus, booking_fixture)

    with TestClient(create_app(settings, service)) as client:
        health_before = client.get("/health")
        index_response = client.post("/index")
        health_after = client.get("/health")

    assert health_before.status_code == 200
    assert health_before.json()["backend"] == backend
    assert health_before.json()["index_loaded"] is False
    assert index_response.status_code == 200
    assert index_response.json()["backend"] == backend
    assert index_response.json()["files_indexed"] == 1
    assert index_response.json()["units_indexed"] == 2
    assert index_response.json()["index_path"]
    assert index_response.json()["corpus_fingerprint"]
    assert health_after.json()["index_loaded"] is True


def test_valid_unanswerable_query_returns_exact_fallback(
    backend, corpus, booking_fixture, tmp_path
):
    service, generator = make_service(
        corpus, booking_fixture, tmp_path, backend=backend
    )
    service.build_index()
    settings = settings_for(backend, corpus, booking_fixture)

    with TestClient(create_app(settings, service)) as client:
        response = client.post("/chat", json={"query": "Is there a shop on Mars?"})

    assert response.status_code == 200
    assert response.json()["answer"] == FALLBACK_ANSWER
    assert generator.grounded_calls == 1


def test_empty_and_whitespace_queries_are_validation_errors(
    backend, corpus, booking_fixture, tmp_path
):
    service, _ = make_service(
        corpus, booking_fixture, tmp_path, backend=backend
    )
    settings = settings_for(backend, corpus, booking_fixture)

    with TestClient(create_app(settings, service)) as client:
        empty = client.post("/chat", json={"query": ""})
        whitespace = client.post("/chat", json={"query": "   "})

    assert empty.status_code == 422
    assert whitespace.status_code == 422


def test_chat_requests_exactly_top_three_units(
    backend, corpus, booking_fixture, tmp_path
):
    (corpus / "more.md").write_text(
        "# Shipping\nShipping policy.\n"
        "# Accounts\nAccount policy.\n"
        "# Payments\nPayment policy.\n",
        encoding="utf-8",
    )
    service, _ = make_service(
        corpus, booking_fixture, tmp_path, backend=backend
    )
    service.build_index()

    response = service.chat(ChatRequest(query="What are the policies?"))

    assert len(response.retrieved) == 3
    assert [result.rank for result in response.retrieved] == [1, 2, 3]


def test_shared_contract_rejects_non_three_top_k(corpus, booking_fixture, tmp_path):
    retriever = BM25Retriever(tmp_path / ".kb" / "index.json")

    with pytest.raises(ValueError, match="top_k=3"):
        QAService(
            retriever,
            FakeGenerator(),
            TransactionStore(booking_fixture),
            corpus,
            top_k=2,
        )


def test_answer_model_is_pinned():
    with pytest.raises(ValidationError):
        Settings(openai_chat_model="another-model")


def test_embedding_model_is_pinned_for_production_configuration():
    with pytest.raises(ValidationError):
        Settings(openai_embedding_model="another-embedding-model")
