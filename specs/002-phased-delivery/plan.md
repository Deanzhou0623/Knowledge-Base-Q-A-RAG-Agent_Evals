# Implementation Plan: Phased Vector-First Delivery

## Documentation changes

- Renumber responsibility specs to match delivery order.
- Add a seed-dataset specification and a backend-neutral UI specification.
- Preserve existing grounding, persistence, citation, fallback, and evaluation
  requirements.
- Update repository documentation and Spec Kit workflow links.

## Implementation phases

1. Reconcile existing shared contracts and offline test harness.
2. Validate and freeze seed dataset version 1.
3. Complete Vector RAG against shared contracts and tests.
4. Build the backend-neutral UI and UI contract tests.
5. Complete the plain BM25 Markdown KB and shared contract tests.
6. Complete the evaluation runner and graders.
7. Expand the dataset blind, freeze its version, and run final comparisons.

## Constitution check

- Fair comparison: preserved through pre-implementation seed freeze.
- Transparent BM25: preserved; no reactive optimization is allowed.
- Testability: strengthened by per-phase quality gates.
- UI separation: explicit; controlled evaluation bypasses the UI.
- Traceability: this feature owns the delivery-order documentation change.
