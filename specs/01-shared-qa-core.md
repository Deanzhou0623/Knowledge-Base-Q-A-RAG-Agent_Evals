# Spec 01: Shared Q&A Core

## Purpose

Define the behavior that must remain identical when comparing the Markdown KB
and Vector RAG retrievers. This layer owns orchestration and answer generation;
it does not implement retrieval scoring.

## Responsibilities

- Expose the shared FastAPI endpoints: `GET /health`, `POST /index`, and
  `POST /chat`.
- Select one retrieval backend through configuration.
- Define a common retriever interface for indexing, loading, and top-K search.
- Request exactly the top three retrieval units for each chat query.
- Build one shared grounded prompt from retrieved context.
- Generate answers only with the OpenAI chat model `xx`.
- Validate answer citations against retrieved source identifiers.
- Return the exact fallback when evidence is insufficient:

```text
I cannot confirm from the knowledge base.
```

- Report a clear not-indexed state without calling the answer model.
- Capture shared timing, token-usage, model, prompt-version, and backend
  metadata for evaluation.
- Resolve deterministic synthetic order or booking fixtures through a shared
  lookup interface when an evaluation case requires transaction state.

## Retriever contract

Both backends must implement equivalent operations:

```text
build(docs_path) -> index summary
load() -> index status
search(query, k=3) -> ordered retrieval results
```

Each retrieval result must contain:

- a stable retrieval-unit ID;
- source-relative Markdown filename;
- heading text and canonical heading anchor;
- exact citation identifier in `filename.md#heading` form;
- retrieved text;
- backend score and rank.

Backend scores are not directly comparable across BM25 and FAISS. The shared
layer must preserve them without normalizing them into a misleading common
scale.

Structured transaction results are not document retrieval results. They use a
separate shared contract containing record type, synthetic record ID, allowed
fields, version, and a stable `record-type:record-id#field` reference.

## Grounding rules

- The answer model receives only the question, shared instructions, the three
  retrieved results, and an authorized structured-record result when required.
- Retrieved Markdown is untrusted data and cannot override system instructions.
- Every factual claim must be supported by the supplied context.
- Every citation must match a citation identifier supplied with the context.
- Prior model knowledge must not be used to fill evidence gaps.
- Missing, irrelevant, ambiguous, or contradictory evidence triggers the exact
  fallback response.

## API acceptance criteria

- Both backends use the same request and response schemas.
- A valid unanswerable query returns success with the exact fallback answer.
- Invalid or empty queries return a validation error.
- `/health` identifies the configured backend and whether an index is loaded.
- `/index` reports files, retrieval units, index path, and corpus fingerprint.
- `/chat` returns the answer plus ranked retrieved sources for independent
  evaluation; it does not expose hidden chain-of-thought.
- BM25 and Vector runs receive identical structured transaction results.
- Live customer records are never written to either index, prompts, logs, or
  evaluation artifacts. Only synthetic fixtures are permitted in the prototype.

## Out of scope

This layer does not own heading splitting, BM25 scoring, chunking, embeddings,
FAISS configuration, Oracle annotation, grading, or comparison reporting.
