from fastapi.testclient import TestClient

from kbqa.api import create_app
from kbqa.config import Settings
from kbqa.models import FALLBACK_ANSWER, ChatRequest
from kbqa.retrievers.markdown_kb import BM25Retriever
from kbqa.service import QAService
from kbqa.transactions import TransactionStore

from conftest import FakeGenerator


def make_service(corpus, booking_fixture, tmp_path, answer=FALLBACK_ANSWER):
    retriever = BM25Retriever(tmp_path / ".kb" / "index.json")
    generator = FakeGenerator(answer)
    service = QAService(
        retriever,
        generator,
        TransactionStore(booking_fixture),
        corpus,
        top_k=3,
    )
    return service, generator


def test_chat_rejects_unbuilt_index_without_calling_model(corpus, booking_fixture, tmp_path):
    service, generator = make_service(corpus, booking_fixture, tmp_path)
    settings = Settings(docs_path=corpus, transaction_fixture_path=booking_fixture)

    with TestClient(create_app(settings, service)) as client:
        response = client.post("/chat", json={"query": "Refund timing?"})

    assert response.status_code == 409
    assert generator.grounded_calls == 0


def test_grounded_chat_accepts_only_supplied_citations(corpus, booking_fixture, tmp_path):
    valid = "Refunds take 7 to 11 business days. [policy.md#refund-policy]"
    service, _ = make_service(corpus, booking_fixture, tmp_path, valid)
    service.build_index()

    response = service.chat(ChatRequest(query="How long does a refund take?"))
    assert response.answer == valid
    assert response.citations == ["policy.md#refund-policy"]

    service.generator.answer = "Tomorrow. [invented.md#answer]"
    response = service.chat(ChatRequest(query="How long does a refund take?"))
    assert response.answer == FALLBACK_ANSWER
    assert response.citations == []


def test_transaction_only_chat_does_not_require_an_index(corpus, booking_fixture, tmp_path):
    answer = "The booking is confirmed. [booking:BK-1#status]"
    service, _ = make_service(corpus, booking_fixture, tmp_path, answer)

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
    corpus, booking_fixture, tmp_path
):
    service, _ = make_service(corpus, booking_fixture, tmp_path)
    service.build_index()
    (corpus / "policy.md").write_text("# Changed\nNew policy", encoding="utf-8")

    restarted, _ = make_service(corpus, booking_fixture, tmp_path)
    assert restarted.load() is False
    assert restarted.health().index_loaded is False
