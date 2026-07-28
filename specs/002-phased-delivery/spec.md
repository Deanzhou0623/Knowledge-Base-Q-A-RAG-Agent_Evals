# Feature Specification: Phased Vector-First Delivery

## Goal

Reorder implementation so the team can learn through a complete Vector RAG
vertical slice and UI before building the Markdown KB baseline, without biasing
the eventual evaluation.

## User stories

1. A developer can implement and verify shared contracts before choosing
   backend details.
2. An evaluator can freeze seed questions and gold evidence before retrieval
   outputs exist.
3. A developer can build Vector RAG first and exercise it through a thin UI.
4. A developer can add Markdown KB later through the same API and UI.
5. The final evaluation can demonstrate that both arms used controlled inputs.

## Acceptance scenarios

- The seed dataset validates and resolves evidence before backend execution.
- Vector RAG passes component, persistence, and API-contract tests before UI
  work is accepted.
- UI code contains no retrieval or grading logic.
- Markdown KB passes the same shared contract suite without changing seed cases.
- Expanded evaluation cases are annotated blind and frozen before final runs.
- No testing phase is deferred until after all implementation work.
