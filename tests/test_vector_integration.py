import os

import pytest

from kbqa.embeddings import OPENAI_EMBEDDING_MODEL, OpenAIEmbeddingProvider
from kbqa.retrievers.vector_rag import VectorRetriever


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set RUN_OPENAI_INTEGRATION=1 and OPENAI_API_KEY to run OpenAI integration",
)


def test_real_embedding_build_and_search_is_explicitly_opt_in(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text(
        "# Refund Timeline\nApproved refunds take seven to eleven business days.",
        encoding="utf-8",
    )
    retriever = VectorRetriever(
        tmp_path / ".kb" / "faiss_index",
        OpenAIEmbeddingProvider(
            OPENAI_EMBEDDING_MODEL,
            api_key=os.environ["OPENAI_API_KEY"],
        ),
        chunk_words=20,
        overlap=5,
    )

    retriever.build(docs)
    results = retriever.search("How long do approved refunds take?", 3)

    assert results[0].citation == "policy.md#refund-timeline"
