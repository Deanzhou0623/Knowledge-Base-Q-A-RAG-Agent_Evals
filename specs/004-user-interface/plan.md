# Implementation Plan: Backend-Neutral Browser UI

## Approach

- Mount package-owned static assets under `/ui/` in the existing FastAPI app.
- Keep HTTP calls, orchestration, and DOM rendering in a dependency-free ES
  module with injectable API and view adapters.
- Render all API-controlled values with `textContent`; do not parse Markdown or
  assign HTML.
- Use the existing response schemas without adding or changing API fields.

## Interfaces consumed

- `GET /health`: backend and index readiness.
- `POST /index`: build summary and progress completion.
- `POST /chat`: answer, backend, timing, model, and ranked retrieval records.

## Test strategy

- Node built-in tests inject mocked API and view adapters for healthy,
  unindexed, answerable, unsupported, rebuild, and server-error behavior.
- A fake DOM verifies source data is assigned only as text.
- FastAPI smoke tests serve the real static assets and run the unchanged UI/API
  route flow against both configured backends with fake providers.
- Run the complete Python suite to protect shared API contracts.

## Constitution check

- Backend comparison remains controlled because the UI sends no backend choice
  and changes no query or response.
- Grounding remains server-owned; the client does not repair answers or
  citations.
- Evaluation bypasses the browser.
- No dependencies, credentials, or backend internals are added to the client.
