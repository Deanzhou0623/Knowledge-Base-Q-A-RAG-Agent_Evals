# Spec 03: Vector RAG Retriever

## Purpose

Provide the semantic retrieval comparison: deterministic Markdown chunks
retrieved through embeddings and a FAISS similarity index. This is the first
complete retrieval vertical slice.

## Delivery constraints

- Implement only after the shared API and frozen seed dataset contracts are
  reviewed.
- Build against backend-neutral interfaces; do not place FAISS, embedding, or
  chunk-specific logic in the shared API layer.
- Add chunking, embedding-adapter, search, persistence, restart, API-contract,
  and citation tests during this phase.
- Use the seed dataset for regression checks, but do not tune chunking or
  retrieval settings to maximize seed scores. Record failures as observations.
- Record the initial chunking and embedding configuration before implementing
  Markdown KB so later comparison changes remain auditable.

## Pipeline

```text
docs/**/*.md
  -> deterministic chunks
  -> embeddings
  -> FAISS index
  -> top 3 chunks
```

## Chunking requirements

- Discover the same Markdown corpus used by the BM25 backend.
- Use deterministic fixed-size or token-aware chunking with recorded chunk size
  and overlap.
- Preserve source-relative filename and the nearest applicable heading for every
  chunk.
- Do not create a chunk whose citation cannot be mapped unambiguously to a real
  `filename.md#heading` source.
- Generate stable chunk IDs when corpus and chunking configuration are unchanged.

## Embedding and retrieval requirements

- Use one pinned embedding model for indexing and query embedding.
- Record embedding model, vector dimension, normalization, distance metric, and
  FAISS index type.
- Return the three nearest chunks in deterministic rank order where ties permit.
- Preserve raw similarity/distance score and rank in every result.
- Return the shared retriever result schema defined in Spec 01.
- The embedding model retrieves context only; it does not generate answers.

## Persistence

Persist the FAISS index and restoration metadata under:

```text
.kb/faiss_index/
```

Metadata must map every FAISS row to its chunk text and canonical source
citation. Write index artifacts safely, validate compatibility on load, and
load automatically at startup without recomputing document embeddings.

## Expected evaluation failures

- `semantic_false_positive`
- `chunk_boundary_failure`
- `missing_exact_term`

## Acceptance criteria

- Restored retrieval matches pre-restart retrieval for a fixed index and query.
- Every vector row has valid metadata and a resolvable source citation.
- No document embedding call is made during startup restoration.
- Chunking, metadata preservation, persistence, and search are covered by tests.
- Embedding calls are mocked in unit tests and opt-in for integration tests.
- Retrieved chunk citations can be matched against labeled Oracle document
  sources without weakening source identity to filename-only matching.
