# Feature Specification: Frozen Seed Evaluation Dataset

## Goal

Freeze a small pre-retrieval evaluation dataset whose questions, evidence, and
Oracle labels can be audited and validated without calling a retriever or LLM.
The seed is a regression fixture, not a retrieval-tuning leaderboard.

## User stories

1. An evaluator can validate the seed version, case-file hash, corpus
   fingerprint, and transaction-fixture version with one offline command.
2. An annotator receives a precise error when a document evidence span or
   structured-record field no longer resolves.
3. A backend runner consumes the same frozen cases through the manifest rather
   than selecting or rewriting cases for a particular retriever.

## Acceptance scenarios

- The seed contains all four categories and an answerable policy paraphrase
  pair.
- It contains both transaction-only and transaction-plus-document cases.
- Every answerable case has expected facts, acceptable evidence, and Oracle
  sources selected from that evidence.
- Document evidence resolves at the source-heading and minimal-span level.
- Structured evidence resolves against the pinned synthetic fixture and frozen
  `as_of` clock.
- Unsupported cases have no gold evidence and require the exact shared
  fallback.
- Modifying the cases, corpus, or fixture version without updating the
  versioned manifest makes validation fail.

## Clarifications

- Version metadata lives in a JSON manifest and cases remain JSONL so each case
  stays independently inspectable.
- Immutability is enforced by the cases SHA-256 stored in the manifest.
- The corpus fingerprint uses the same deterministic function as index
  metadata.
- Automated validation proves reference resolution and declared annotation
  invariants; human judgment remains responsible for factual sufficiency and
  minimality.
