# Feature Specification: Runnable RAG Evaluation Scaffold

> Historical artifact: this specification records the initial all-backends
> scaffold. New implementation work follows
> [`../002-phased-delivery/`](../002-phased-delivery/).

## Goal

Create a runnable Python scaffold implementing the contracts in `prompt.md`
without presenting the scaffold as a completed production benchmark.

## User stories

1. A developer can select BM25 or Vector retrieval behind one FastAPI API.
2. A developer can build and reload either index across restarts.
3. A developer can ask grounded questions and receive auditable citations.
4. An evaluator can load versioned cases and compare retrieval outputs offline.
5. Tests can replace OpenAI and embedding calls with deterministic fakes.

## Acceptance scenarios

- `/chat` before indexing returns an explicit not-indexed response.
- `/index` persists the selected backend in its specified location.
- A restarted service loads a compatible index without rebuilding.
- Unsupported or invalidly cited generated answers use the exact fallback.
- Synthetic booking fields resolve through stable structured references.
- Unit and API tests run without an API key.
