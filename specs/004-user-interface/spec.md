# Feature Specification: Backend-Neutral Browser UI

## Goal

Let a developer index and manually query the configured Q&A backend in a browser
while preserving the shared API as the sole source of backend and answer state.

## Clarified requirements

- FastAPI serves one static interface at `/ui/`; `/` redirects there.
- The client calls only `GET /health`, `POST /index`, and `POST /chat`.
- The configured server chooses the backend. The browser does not send a
  backend name or contain backend-specific branches.
- Health, indexing, answering, and error operations expose visible progress.
- The answer is displayed byte-for-byte as returned, including the exact
  fallback.
- Retrieved rank, heading, citation, score, and text are displayed as untrusted
  text using DOM text nodes.
- A `409` chat response is presented as an unindexed state with a rebuild hint.
- The UI is excluded from evaluation execution and timing.

## Acceptance scenarios

1. A healthy ready or unindexed response displays its backend and readiness.
2. Rebuild progress is visible and a successful rebuild refreshes health.
3. An answerable response shows the answer and ordered source metadata.
4. An unsupported response shows the exact fallback without rewriting it.
5. Server and connectivity errors remain visible and do not become answers.
6. Vector and BM25 smoke tests use the same UI assets and shared endpoints.
7. Malicious source markup is rendered as inert text.

## Out of scope

Conversation history, streaming, backend configuration changes, query rewriting,
client-side retrieval, citation repair, grading, and evaluation execution.
