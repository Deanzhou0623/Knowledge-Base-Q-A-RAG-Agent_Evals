# Feature Specification: Controlled Evaluation Runner

## Goal

Execute and audit the same frozen evaluation suite across `llm_only`, `bm25`,
`vector`, and optional `oracle` while preserving the controls in Spec 06 and its
retrieval and answer-evaluation sub-specifications.

## Acceptance scenarios

1. A frozen manifest validates corpus and transaction versions, blind
   annotation, category minimums, trial count, evidence spans, and Oracle
   references before providers or retrievers run.
2. One command executes all main arms; live OpenAI access requires an explicit
   opt-in and offline tests inject deterministic fakes.
3. Per-case JSONL records inputs, outputs, raw retrievals, grades, usage, cost,
   latency, configuration, status, and failures without dropping errored cases.
4. Retrieval grading distinguishes source-heading overlap from evidence-span
   hits and reports transaction-only, LLM-only, and Oracle retrieval as `N/A`.
5. Answer grading uses the same versioned deterministic rules for every arm,
   validates citations against supplied context, and applies each arm's refusal
   contract.
6. The summary reports every arm-by-category cell before scoped improvements,
   keeps user-specific capability separate, states the prompt confound, and
   links failures to per-case records.

## Deliberate limitations

- The seed manifest permits one trial and therefore produces point estimates.
- Lexical fact coverage is an auditable deterministic grader, not semantic
  equivalence. A future LLM grader requires its own pinned, versioned contract.
- Provider pricing defaults to zero and must be explicitly configured from
  current pricing before interpreting cost fields.
- A live final comparison and inter-annotator agreement remain human-owned
  benchmark activities.
