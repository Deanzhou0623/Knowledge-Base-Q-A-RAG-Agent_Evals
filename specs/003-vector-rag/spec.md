# Feature Specification: Verified Vector RAG Retriever

## Goal

Complete the Vector RAG vertical slice defined by `specs/03-vector-rag.md`
without changing the shared Q&A contract or tuning against seed-case outcomes.

## User stories

1. A developer can build deterministic, heading-aware chunks from the shared
   Markdown corpus and retrieve exactly the nearest three through FAISS.
2. A restarted service can restore the saved FAISS index without embedding the
   documents again.
3. An evaluator can audit the embedding, chunking, normalization, distance, and
   index configuration used for a retrieval result.
4. A corrupted or incompatible index is unavailable rather than partly loaded.

## Acceptance scenarios

- Unchanged corpus and chunk settings produce stable chunk IDs and ordering.
- Every indexed row retains non-empty text and an exact canonical
  `filename.md#heading` citation.
- Invalid embedding counts, dimensions, non-finite values, and zero vectors fail
  explicitly.
- Search returns deterministic ranks and raw inner-product scores.
- Metadata and FAISS artifacts restore identical results without document
  embedding calls.
- Hash, schema, model, chunking, normalization, metric, index type, row count,
  dimension, and row metadata incompatibilities are rejected.
