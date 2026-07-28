# Implementation Plan: Controlled Evaluation Runner

## Interfaces

- `EvaluationDataset` combines JSONL cases with a versioned JSON manifest.
- `validate_dataset` resolves document spans and structured references before
  execution.
- `run` accepts injected providers for offline use and requires `allow_live`
  before constructing OpenAI providers.
- Retrieval and answer graders return JSON-serializable, versioned metrics.
- Reporting recomputes cells and scoped improvements from persisted records.

## Controls

- Assert corpus fingerprint, transaction version, grader version, `K = 3`,
  arm matrix, category minimums, and trial count.
- Use one answer generator and one vector embedding adapter per run.
- Build both indexes from the same validated corpus and record their distinct
  configurations without comparing backend scores.
- Keep Oracle evidence minimal and bypass retriever construction in Oracle-only
  execution.

## Test strategy

- Validate malformed manifests, evidence, categories, and controlled inputs.
- Grade full versus section-only hits, citations, refusals, unsupported
  retrieval, transaction-only cases, and paraphrase groups offline.
- Execute the full matrix with fakes and verify record/report contracts.
- Preserve provider and indexing errors as auditable records.
- Skip the real OpenAI integration test unless credentials and a dedicated
  opt-in environment flag are both present.
