# Spec 04: Evaluation Runner

## Purpose

Run a controlled, reproducible comparison of Markdown KB and Vector RAG while
keeping all non-retrieval inputs fixed.

The runner contains two smaller tracks:

1. [Retrieval evaluation](evals/04a-retrieval-evaluation.md)
2. [Answer and citation evaluation](evals/04b-answer-evaluation.md)

## Controlled inputs

Both backend runs must share:

- corpus and corpus fingerprint;
- evaluation questions and reference criteria;
- `K = 3`;
- exact OpenAI answer model `xx`;
- grounded prompt version;
- generation settings;
- citation format and exact fallback text;
- graders, grader versions, and scoring thresholds.

Retrieval-specific configuration must be recorded but must not alter the shared
answering or grading path.

## Dataset contract

Keep evaluation data outside the indexed `docs/` directory. Each case must have:

- stable question ID and question text;
- whether it is answerable from the corpus;
- expected facts or grading criteria;
- acceptable source citations;
- optional paraphrase-group ID;
- optional expected failure label.

## Runner workflow

```text
load versioned evaluation cases
  -> run all cases against BM25
  -> run all cases against Vector RAG
  -> grade retrieval results
  -> grade answers and citations
  -> attach failure labels
  -> write per-case results
  -> aggregate a side-by-side report
```

Run cases in a way that makes latency measurements meaningful. Record cold/warm
state, concurrency, retries, and failures. Do not silently drop errored cases.

## Result contract

Every result must record:

- run ID, question ID, timestamp, backend, and status;
- corpus fingerprint and index configuration;
- retrieved IDs, citations, ranks, and raw scores;
- answer and parsed citations;
- answerability and expected sources;
- retrieval and answer grades;
- failure labels;
- index, retrieval, and generation latency as applicable;
- embedding and answer token usage and estimated cost;
- exact model IDs, prompt version, generation settings, and grader version.

Produce machine-readable per-case output and a human-readable aggregate report.

## Acceptance criteria

- One command can run the same suite against both backends.
- Inputs held constant are asserted, not merely assumed.
- Per-case results retain enough evidence to audit aggregate scores.
- Retrieval and answer metrics can be recomputed from stored results.
- A backend failure is visible and cannot be mistaken for a low-quality answer.
- The report compares both systems and lists categorized failure examples.
