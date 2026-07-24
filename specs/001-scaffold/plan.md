# Implementation Plan: Runnable RAG Evaluation Scaffold

## Stack

- Python 3.10+
- FastAPI with lifespan startup loading
- OpenAI Responses and Embeddings APIs behind injectable protocols
- rank-bm25 and FAISS CPU
- Pydantic settings and models
- pytest and FastAPI TestClient

## Structure

```text
src/kbqa/          application and retrieval package
src/kbqa/evals/    evaluation dataset, metrics, and runner
docs/              fictional e-commerce policy corpus
fixtures/          synthetic transaction records
evals/             versioned evaluation cases
tests/             offline unit and API tests
```

## Constitution check

- No live data: pass; fixtures are synthetic.
- Reproducible persistence: atomic writes plus metadata/fingerprints.
- Fair retrieval comparison: shared result model and K.
- Offline tests: provider protocols and deterministic fakes.
