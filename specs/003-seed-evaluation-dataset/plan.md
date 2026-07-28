# Implementation Plan: Frozen Seed Evaluation Dataset

## Data contract

- Add a frozen manifest containing schema version, dataset identity/version,
  annotation date, case-file hash, corpus fingerprint, fixture version, freeze
  status, and the pre-output Oracle annotation attestation.
- Strengthen case models for category, answerability, expected facts, fallback,
  transaction-kind, citation, and evidence-span invariants.
- Preserve the existing JSONL case format and runner compatibility.

## Validation

- Load Markdown sections without a retrieval backend.
- Resolve every document citation to a canonical heading section and verify its
  evidence text or offsets.
- Resolve structured citations through the deterministic synthetic transaction
  store at the case's pinned fixture version and `as_of` time.
- Reject missing categories, absent paraphrase pairs, duplicate IDs, stale
  fingerprints, fixture mismatches, and modified frozen case files.

## Integration

- Make the evaluation CLI default to the frozen manifest while retaining raw
  JSONL support for small unit fixtures.
- Attach the dataset version to records created from a manifest.
- Document a model-free validation command.

## Constitution check

- No network or model call is needed.
- Gold evidence is outside `docs/` and cannot be indexed.
- Frozen hashes and fingerprints make unreviewed drift explicit.
- No backend-specific retrieval behavior or tuning is introduced.
