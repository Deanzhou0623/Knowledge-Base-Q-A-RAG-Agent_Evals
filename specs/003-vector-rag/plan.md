# Implementation Plan: Verified Vector RAG Retriever

## Scope and invariants

- Preserve the Spec 01 retriever result, API, prompt, citation, fallback, model,
  and `K = 3` contracts.
- Preserve the frozen Spec 02 cases; no question or annotation changes.
- Keep vector-specific validation and persistence inside `VectorRetriever`.
- Keep embeddings injectable so all phase tests run offline.

## Design

- Use whitespace-word chunks with 160-word size and 30-word overlap by default.
- Skip empty Markdown sections because they contain no retrievable evidence.
- L2-normalize document and query vectors and rank with FAISS `IndexFlatIP`.
- Persist a content-addressed FAISS file plus atomically replaced JSON metadata.
- Validate the complete persistence configuration and row mapping before
  publishing loaded state.
- Resolve score ties by stable chunk ID after retrieving all indexed rows.

## Test strategy

- Unit-test chunk boundaries, overlap, metadata, stable IDs, and invalid config.
- Use counting and malformed fake embedding providers for adapter validation.
- Test persistence metadata, restart equivalence, no re-embedding, corruption,
  incompatibility, and stale-state clearing.
- Retain the shared API-contract suite for Vector and BM25.
