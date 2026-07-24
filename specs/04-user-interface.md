# Spec 04: Backend-Neutral User Interface

## Purpose

Provide a small interface for manually exercising the Q&A system after the
Vector RAG vertical slice works and before Markdown KB is implemented. The UI is
an API client, not a retrieval or evaluation layer.

## User capabilities

- Ask a natural-language question.
- View the grounded answer or exact fallback.
- Inspect retrieved source identifiers, headings, ranks, and backend scores.
- See the active backend and index-ready state.
- Trigger an index rebuild with clear progress and error feedback.
- Switch between configured backends when both are available without changing
  the request or response contract.

## Architecture constraints

- Call only the shared `GET /health`, `POST /index`, and `POST /chat` API.
- Do not import retriever, FAISS, BM25, embedding, prompt, or evaluation code.
- Do not construct, modify, or repair citations in the browser.
- Do not add hidden conversation context, query rewriting, or client-side
  retrieval.
- Render source text as untrusted content and escape it appropriately.
- Keep the interface functional when only Vector RAG is initially available.
  Adding Markdown KB later must require no backend-specific UI flow.

## Evaluation boundary

The UI is for development and demonstration. Controlled evaluations call the
shared application service or API directly and never scrape the UI. UI latency,
rendering, and user interaction time are excluded from retrieval and generation
metrics.

## Testing requirements

- Contract tests use mocked API responses for healthy, unindexed, answerable,
  unsupported, and server-error states.
- A smoke test verifies the UI can query Vector RAG through the shared API.
- After Markdown KB is implemented, the same smoke test runs with that backend
  without changing UI code.
- Citation/source rendering must not execute HTML or Markdown-provided scripts.

## Acceptance criteria

- A user can index, ask, and inspect sources without using command-line tools.
- The displayed backend and readiness state match `/health`.
- Exact fallback text is displayed without client-side rewriting.
- The UI remains unchanged when the configured backend changes.
- No retrieval or grading decision is made in UI code.
