# Spec 02: Markdown KB Retriever

## Purpose

Provide the simple, transparent retrieval baseline: Markdown heading sections
retrieved with BM25 keyword search.

## Pipeline

```text
docs/**/*.md
  -> Markdown heading sections
  -> inspectable section records
  -> BM25 index
  -> top 3 sections
```

## Indexing requirements

- Discover Markdown files deterministically and store source-relative paths.
- Split documents at Markdown headings while preserving heading hierarchy.
- Handle content before the first heading, nested headings, duplicate headings,
  and empty sections deterministically.
- Generate stable GitHub-style heading anchors, including deterministic suffixes
  for duplicates.
- Keep raw section Markdown available for prompt construction.
- Store stable section IDs, filename, heading, anchor, text, and BM25 data.
- Fingerprint the corpus and record index schema/configuration versions.

## Retrieval requirements

- Tokenize and score queries consistently with the indexed sections.
- Return the three highest-ranked sections in deterministic order.
- Preserve raw BM25 score and rank in every result.
- Return the shared retriever result schema defined in Spec 01.
- Do not add embeddings, semantic query expansion, LLM reranking, or hybrid
  retrieval to the baseline.

## Persistence

Persist an inspectable index at:

```text
.kb/index.json
```

Write the index atomically. Load a compatible index automatically at startup.
Reject corrupted, incompatible, or incomplete index data and expose an
unavailable-index state instead of silently rebuilding during chat.

## Expected evaluation failures

- `synonym_miss`
- `keyword_false_positive`
- `weak_retrieval`

These are outcomes to record, not defects that must be optimized away before
the baseline evaluation.

## Acceptance criteria

- The JSON index can be inspected without backend-specific tooling.
- Section citations resolve to real Markdown headings.
- Restarting the service restores retrieval without re-indexing.
- Re-indexing an unchanged corpus produces stable section IDs and ordering.
- Unit tests cover heading and anchor edge cases plus serialization round trips.
- Retrieved section citations can be matched directly against labeled Oracle
  document sources during evaluation.
