# GitHub Spec Kit Workflow

## Purpose

Use this project to practice AI-native software development through
specification-driven implementation and refactoring. The AI agent may draft
requirements, plans, tasks, code, and tests, but reviewed specifications and
observable evaluation evidence remain the source of truth.

This workflow follows the official
[GitHub Spec Kit](https://github.com/github/spec-kit) approach. It documents the
methodology expected by this repository; it does not claim that the Spec Kit CLI
has already been initialized.

## Project constitution

Before implementation, establish governing principles that apply to all six
major project parts:

- source-grounded answers over convenience or fluency;
- identical non-retrieval conditions for both backends;
- reproducible and inspectable experiments;
- explicit failure instead of silent corruption or unsupported answers;
- tests for behavior, contracts, persistence, and evaluation metrics;
- simple BM25 baseline without optimization that conceals its failure modes;
- requirement and task traceability for AI-generated changes;
- seed questions and gold evidence frozen before retriever implementation;
- tests delivered with every implementation phase rather than postponed;
- a backend-neutral UI kept outside the controlled evaluation path;
- no secrets, credentials, or hidden test data committed to the repository.

Record these principles in the Spec Kit constitution after initialization.

## Artifact hierarchy

The repository-level files define product intent:

```text
README.md
prompt.md
specs/01-shared-qa-core.md
specs/02-seed-evaluation-dataset.md
specs/03-vector-rag.md
specs/04-user-interface.md
specs/05-markdown-kb.md
specs/06-evaluation-runner.md
specs/evals/06a-retrieval-evaluation.md
specs/evals/06b-answer-evaluation.md
```

Spec Kit feature artifacts refine that intent into implementation work:

```text
feature spec.md
  -> clarified requirements
  -> plan.md and design artifacts
  -> tasks.md
  -> code and tests
  -> evaluation evidence
```

The feature artifacts may add implementation detail, but they must not weaken
the shared comparison, grounding, citation, persistence, or fallback rules.

## Workflow

### 1. Constitution

Establish or update project-wide principles before feature work. Review later
plans and tasks against those principles.

### 2. Specify

Select one bounded capability from the six responsibility specs. Describe what
users or evaluators need and why, including observable acceptance scenarios.
Avoid making unnecessary technology decisions in the feature specification.

Examples of bounded features include:

- shared `/chat` request and response contract;
- frozen seed-case validation and evidence resolution;
- deterministic vector chunking and FAISS restoration;
- backend-neutral source and citation rendering;
- Markdown heading parser and persistent BM25 index;
- Recall@3 computation and per-case result serialization.

### 3. Clarify

Resolve ambiguity that could materially change behavior, data contracts,
evaluation fairness, persistence, or test expectations. Record decisions in the
feature artifacts rather than leaving them only in an AI conversation.

### 4. Plan

Choose implementation details, interfaces, data models, dependencies, and test
strategy. Include constitution checks and explicitly identify which shared
contracts the feature consumes or changes.

### 5. Tasks

Break the plan into small, dependency-ordered tasks. Every task should name its
target files and acceptance check. Keep backend-specific work separate when it
can be completed independently, then integrate through the shared contract.

### 6. Analyze

Check the specification, plan, and tasks for omissions, contradictions,
duplicate responsibilities, untestable requirements, and violations of the
controlled-comparison rules. Resolve material findings before implementation.

### 7. Implement or refactor

Implement one traceable task group at a time. For refactoring, first capture
current behavior in tests and identify which requirements must remain invariant.
Do not mix unrelated retrieval tuning into structural refactors.

### 8. Validate and converge

Run unit, contract, integration, persistence, and evaluation checks appropriate
to the change. Compare observed behavior with the feature specification and add
remaining gaps back to the task list. A feature is complete only when its
acceptance scenarios have evidence.

## Delivery sequence

```text
shared contracts and offline test harness
  -> frozen seed evaluation dataset
  -> Vector RAG vertical slice and tests
  -> backend-neutral UI and tests
  -> Markdown KB baseline and tests
  -> evaluation runner
  -> blind dataset expansion and version freeze
  -> final controlled evaluation
```

Spec numbers follow this delivery sequence. The final product still compares
both retrievers symmetrically; implementing Vector first does not permit
Vector-specific changes to shared behavior or later question selection.

## Refactoring rule

Before an AI agent refactors code, require it to provide:

1. the requirement and task being addressed;
2. the behavior that must remain unchanged;
3. the files and interfaces expected to change;
4. the tests and evaluation cases that prove equivalence or intended change;
5. any spec, plan, or task updates discovered during implementation.

Refactoring is not complete because code compiles or looks cleaner. It is
complete when contract tests pass and the relevant BM25-versus-Vector evaluation
results remain comparable.

## Review checklist

- Does the code change trace to a reviewed requirement and task?
- Do the specification, plan, tasks, tests, and implementation agree?
- Are shared behavior and backend-specific behavior still separated?
- Were model IDs, prompt versions, corpus fingerprints, and settings recorded?
- Do citations and the exact fallback remain valid?
- Can indexes still be restored after restart?
- Can evaluation metrics be reproduced from stored per-case results?
- Were changes in answer quality or retrieval behavior explained rather than
  hidden by aggregate scores?

## Tooling note

When the team is ready to use the official tooling, install or run a pinned
release from the official `github/spec-kit` repository and initialize it for the
chosen coding-agent integration. Commit the generated constitution, feature
specifications, plans, tasks, and relevant design artifacts. Do not commit local
credentials or machine-specific configuration.
