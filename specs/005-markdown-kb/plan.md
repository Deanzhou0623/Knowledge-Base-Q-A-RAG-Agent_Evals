# Implementation Plan: Markdown KB Baseline

## Interfaces consumed

- `DocumentUnit`, `RetrievalResult`, `IndexSummary`, and the Spec 01 retriever
  protocol.
- Shared `QAService`, FastAPI endpoints, grounded prompt, citation validation,
  exact fallback, transaction lookup, and fixed `K = 3`.

## Design

- Parse Markdown deterministically in sorted source-relative file order.
- Emit one section per heading plus substantive pre-heading content, retaining
  heading level and ancestor path.
- Generate canonical per-document anchors and IDs from source path plus anchor.
- Tokenize lowercase Unicode word sequences in both indexed sections and
  queries; construct unmodified `rank_bm25.BM25Okapi` with recorded defaults.
- Persist all restoration data to `.kb/index.json` via atomic replacement.
- Validate schema, configuration, metadata, section integrity, and persisted
  tokens before hydrating an index.

## Test strategy

- Parser unit tests cover preambles, nesting, duplicates, empty sections,
  Setext syntax, fenced code, anchors, source paths, and stable ordering.
- Retriever tests cover ranking, raw scores, top three, inspectability,
  serialization, restart equivalence, stable rebuilds, empty builds, and
  corrupted/incompatible index rejection.
- The shared parameterized service/API tests remain the contract evidence for
  backend neutrality, persistence, citations, fallback, and top-three behavior.

## Constitution check

- The implementation remains a simple BM25 baseline.
- It changes no dataset, answer model, prompt, `K`, citation format, or grader.
- Persistence is inspectable and failures are explicit.
- Tests run offline without OpenAI or embedding calls.
