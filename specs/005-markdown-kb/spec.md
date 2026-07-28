# Feature Specification: Markdown KB Baseline

## Goal

Complete the transparent Markdown-heading and BM25 retrieval baseline described
by `specs/05-markdown-kb.md` while preserving the shared Spec 01 contracts and
without tuning against Vector outputs or seed questions.

## User stories

1. A developer can inspect each Markdown heading section, its hierarchy, stable
   identifier, source-relative path, canonical anchor, citation, and raw text.
2. A developer can build an inspectable JSON BM25 index and restore it after a
   restart without re-indexing.
3. The shared service can retrieve the same fixed top three contract through
   BM25 without any Markdown-specific API or UI behavior.
4. A corrupted, incomplete, stale, or incompatible index remains unavailable
   instead of being silently rebuilt or partially loaded.

## Acceptance scenarios

- Pre-heading content, nested headings, duplicate headings, empty sections,
  Setext headings, and headings appearing inside fenced code are deterministic.
- Duplicate headings receive GitHub-style numeric anchor suffixes and stable
  section IDs when the corpus is unchanged.
- Index JSON records schema, parser/tokenizer configuration, BM25 parameters,
  tokenized corpus data, corpus fingerprint, timestamp, and complete sections.
- Query and document tokenization use the same transparent tokenizer and raw
  BM25 scores and ranks are returned.
- A restart restores the same ordered retrieval results without rebuilding.
- Invalid metadata, units, token data, citations, paths, timestamps, or
  fingerprints are rejected and clear any prior in-memory loaded state.
- The existing backend-neutral contract suite passes unchanged for BM25 and
  Vector.

## Non-goals

- Semantic expansion, embeddings, heading boosts, hybrid search, reranking, or
  changes to frozen evaluation questions and evidence.
- Backend-specific API, prompt, answer-generation, transaction, or UI behavior.
