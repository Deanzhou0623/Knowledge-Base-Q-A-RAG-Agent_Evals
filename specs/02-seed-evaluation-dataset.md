# Spec 02: Frozen Seed Evaluation Dataset

## Purpose

Define a small, versioned evaluation set before implementing either retrieval
backend. The seed set verifies contracts and exposes obvious regressions without
becoming a target for retrieval tuning.

## Position in delivery

```text
shared contracts
  -> frozen seed dataset
  -> Vector RAG
  -> backend-neutral UI
  -> Markdown KB
  -> evaluation runner and final dataset expansion
```

The seed dataset is required before Vector RAG implementation begins. The final
benchmark may contain more cases, but it must preserve the seed cases and their
versioned annotation history.

## Dataset requirements

- Store cases outside `docs/` so evaluation questions are never indexed.
- Include at least one case for each category:
  `company_specific`, `generic_ecommerce`, `user_specific`, and `unsupported`.
- Include at least one paraphrase pair for an answerable policy question.
- Distinguish transaction-only and transaction-plus-document user-specific
  cases through `requires_document_retrieval`.
- Record stable case ID, question, answerability, expected facts, acceptable
  citations, minimal evidence spans, and Oracle sources.
- Pin the corpus fingerprint, transaction-fixture version, and `as_of` clock
  where required.
- Record dataset version and annotation date.

## Annotation protocol

- Author questions from the intended product requirements and corpus, not from
  retrieved outputs.
- Select minimal gold evidence and Oracle sources before running any evaluation
  arm.
- Do not rewrite a question because BM25 or Vector retrieves it poorly.
- Correct factual or annotation errors only through a versioned change with a
  written reason.
- Do not use the seed set as a retrieval-optimization leaderboard.

## Test-fixture boundary

The seed set supports smoke and regression checks, but unit tests should use
small deterministic fixtures tailored to the behavior under test. Passing the
seed set does not establish benchmark quality or replace component tests.

## Acceptance criteria

- The dataset validates without calling an LLM or retriever.
- Every answerable document-backed case resolves to a real source and evidence
  span.
- Every answerable case has minimal Oracle evidence chosen before model output
  inspection.
- Unsupported cases contain no gold evidence and expect the documented refusal
  contract.
- The dataset version and corpus fingerprint are recorded.
- Both backends consume the identical immutable seed cases.
