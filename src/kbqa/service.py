import re
from pathlib import Path
from time import perf_counter

from kbqa.llm import AnswerGenerator
from kbqa.io import corpus_fingerprint
from kbqa.models import (
    FALLBACK_ANSWER,
    SHARED_TOP_K,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexSummary,
)
from kbqa.prompts import PROMPT_VERSION, build_grounded_input
from kbqa.retrievers.base import Retriever
from kbqa.transactions import TransactionStore


CITATION_RE = re.compile(r"\[([^\[\]\n]+)\]")


class IndexNotReady(RuntimeError):
    pass


def extract_citations(answer: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(answer)))


class QAService:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        transaction_store: TransactionStore,
        docs_path: Path,
        top_k: int = SHARED_TOP_K,
    ) -> None:
        if top_k != SHARED_TOP_K:
            raise ValueError(
                f"The shared Q&A contract requires top_k={SHARED_TOP_K}"
            )
        self.retriever = retriever
        self.generator = generator
        self.transaction_store = transaction_store
        self.docs_path = docs_path
        self.top_k = top_k

    def load(self) -> bool:
        loaded = self.retriever.load()
        if loaded and self.retriever.corpus_fingerprint != corpus_fingerprint(self.docs_path):
            self.retriever.loaded = False
            return False
        return loaded

    def health(self) -> HealthResponse:
        return HealthResponse(
            backend=self.retriever.backend,
            index_loaded=self.retriever.loaded,
            corpus_fingerprint=self.retriever.corpus_fingerprint,
        )

    def build_index(self) -> IndexSummary:
        return self.retriever.build(self.docs_path)

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.requires_document_retrieval and not self.retriever.loaded:
            raise IndexNotReady("The knowledge base has not been indexed yet.")

        retrieval_started = perf_counter()
        retrieved = (
            self.retriever.search(request.query, self.top_k)
            if request.requires_document_retrieval
            else []
        )
        structured = None
        if request.booking_id:
            structured = self.transaction_store.lookup(
                request.booking_id,
                as_of=request.as_of,
                expected_version=request.expected_fixture_version,
            )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        prompt_input = build_grounded_input(request.query, retrieved, structured)
        generation_started = perf_counter()
        generation = self.generator.grounded(prompt_input)
        answer = generation.text.strip()
        generation_ms = (perf_counter() - generation_started) * 1000

        allowed = {result.citation for result in retrieved}
        if structured is not None:
            allowed.update(structured.references)
        citations = extract_citations(answer)
        if answer != FALLBACK_ANSWER and (not citations or any(c not in allowed for c in citations)):
            answer = FALLBACK_ANSWER
            citations = []

        return ChatResponse(
            answer=answer,
            backend=self.retriever.backend,
            sources=sorted(allowed),
            citations=citations,
            retrieved=retrieved,
            structured_context=structured,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            model=self.generator.model,
            prompt_version=PROMPT_VERSION,
            token_usage=generation.token_usage,
        )
