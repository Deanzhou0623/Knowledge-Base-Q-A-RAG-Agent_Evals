from kbqa.config import Settings
from kbqa.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from kbqa.llm import AnswerGenerator, OpenAIAnswerGenerator
from kbqa.retrievers.markdown_kb import BM25Retriever
from kbqa.retrievers.vector_rag import VectorRetriever
from kbqa.service import QAService
from kbqa.transactions import TransactionStore


def build_service(
    settings: Settings,
    *,
    generator: AnswerGenerator | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> QAService:
    answer_generator = generator or OpenAIAnswerGenerator(
        settings.openai_chat_model,
        api_key=settings.openai_api_key,
        max_output_tokens=settings.max_output_tokens,
    )
    if settings.retrieval_backend == "bm25":
        retriever = BM25Retriever(settings.bm25_index_path)
    else:
        embedding_provider = embeddings or OpenAIEmbeddingProvider(
            settings.openai_embedding_model, api_key=settings.openai_api_key
        )
        retriever = VectorRetriever(
            settings.vector_index_path,
            embedding_provider,
            chunk_words=settings.vector_chunk_words,
            overlap=settings.vector_chunk_overlap,
        )
    return QAService(
        retriever,
        answer_generator,
        TransactionStore(settings.transaction_fixture_path),
        settings.docs_path,
        top_k=settings.top_k,
    )
