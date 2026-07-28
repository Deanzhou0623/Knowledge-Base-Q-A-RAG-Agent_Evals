import hashlib

import numpy as np
import pytest

from kbqa.models import FALLBACK_ANSWER
from kbqa.llm import GeneratedAnswer
from kbqa.models import TokenUsage


class FakeEmbeddings:
    model = "fake-embeddings-v1"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vector = np.zeros(16, dtype="float32")
        for token in text.lower().split():
            bucket = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % len(vector)
            vector[bucket] += 1
        if not vector.any():
            vector[0] = 1
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class FakeGenerator:
    model = "fake-answer-model"

    def __init__(self, answer: str = FALLBACK_ANSWER) -> None:
        self.answer = answer
        self.grounded_calls = 0
        self.closed_book_calls = 0

    def grounded(self, prompt_input: str) -> GeneratedAnswer:
        self.grounded_calls += 1
        return GeneratedAnswer(
            text=self.answer,
            token_usage=TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14),
        )

    def closed_book(self, question: str) -> GeneratedAnswer:
        self.closed_book_calls += 1
        return GeneratedAnswer(
            text=self.answer,
            token_usage=TokenUsage(input_tokens=5, output_tokens=4, total_tokens=9),
        )


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text(
        "# Refund Policy\n\nApproved refunds take 7 to 11 business days.\n\n"
        "## Cancellation\n\nCancellation is free before the deadline.\n",
        encoding="utf-8",
    )
    return docs


@pytest.fixture
def booking_fixture(tmp_path):
    path = tmp_path / "bookings.json"
    path.write_text(
        '{"version":"v1","bookings":[{"id":"BK-1","status":"confirmed",'
        '"property_id":"P-1","check_in":"2026-09-01",'
        '"free_cancellation_until":"2026-08-10T18:00:00+00:00"}]}',
        encoding="utf-8",
    )
    return path
