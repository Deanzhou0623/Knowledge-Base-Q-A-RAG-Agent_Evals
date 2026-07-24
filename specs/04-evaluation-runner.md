# Spec 04: Evaluation Runner

## Purpose

Run a controlled, reproducible evaluation of Markdown KB versus Vector RAG and
both e-commerce RAG systems versus an LLM-only control while keeping all other
inputs fixed.

The runner contains two smaller tracks:

1. [Retrieval evaluation](evals/04a-retrieval-evaluation.md)
2. [Answer and citation evaluation](evals/04b-answer-evaluation.md)

## Controlled inputs

All evaluation arms must share:

- corpus and corpus fingerprint;
- evaluation questions and reference criteria;
- exact OpenAI answer model `gpt-5.6-sol`;
- generation settings;
- graders, grader versions, and scoring thresholds;
- the transaction-fixture version and, for time-sensitive cases, the frozen
  `as_of` evaluation clock, so every arm resolves the same transaction state.

Both RAG arms additionally share `K = 3`, the grounded prompt version, the
document and structured-record citation formats, and exact knowledge-base
fallback. The LLM-only arm uses one fixed,
versioned control prompt without retrieved context or citation requirements.
Prompt differences must be limited to those required by the experimental arm.

Retrieval-specific configuration must be recorded but must not alter the shared
answer model or grading criteria.

## Evaluation matrix

Every main run must include these arms:

| Arm | Retrieval | Model context |
| --- | --- | --- |
| `llm_only` | None | Question only |
| `bm25` | Markdown KB | Top three sections |
| `vector` | Vector RAG | Top three chunks |

An optional `oracle` arm supplies manually verified correct evidence as an upper
bound on answer generation given perfect retrieval. It is not an absolute
ceiling, must not be included in average production performance, and runs only
on answerable cases: there is no correct oracle context for `unsupported`
questions.

Report every arm across `company_specific`, `generic_ecommerce`,
`user_specific`, and `unsupported` question categories. Retrieval and citation
metrics are `N/A` for `llm_only`, not failed scores.

`user_specific` questions are answerable only through the shared synthetic
transaction lookup, which the `llm_only` control cannot access. They are a
RAG-only capability: exclude them from improvement-over-control scoring and
report them separately. Split them into two sub-kinds, marked by the
`requires_document_retrieval` field, because they exercise the retrieval
comparison differently:

- **transaction-only** (`requires_document_retrieval: false`): answerable purely
  from the transaction lookup, e.g. "What is the status of booking BK-10023?".
  These do not exercise document retrieval — both RAG arms receive the identical
  transaction result and no relevant document exists — so they are **not** a
  BM25-versus-Vector comparison. Report them on their own and mark their
  document-retrieval metrics `N/A`.
- **transaction + document** (`requires_document_retrieval: true`): need the
  transaction result *and* a retrieved policy section, e.g. "My order BK-10023
  arrived late — am I owed a refund?". With the transaction result held
  identical across arms, these remain a valid BM25-versus-Vector comparison of
  the document-retrieval portion.

The `llm_only` arm necessarily uses a different prompt from the RAG arms (prior
knowledge allowed, an uncertainty statement instead of the strict knowledge-base
fallback, no citations). Any RAG-minus-`llm_only` improvement therefore measures
grounded prompt plus retrieved context against control prompt with no context,
not retrieved context in isolation. Record this confound with the results, and
compute improvement scores on answerable categories only (primarily
`company_specific`); do not compute them on `unsupported`, where the arms are
graded under different contracts, nor on `user_specific`, which the control
cannot access.

## Dataset contract

Keep evaluation data outside the indexed `docs/` directory. Each case must have:

- stable question ID and question text;
- category: `company_specific`, `generic_ecommerce`, `user_specific`, or
  `unsupported`;
- for `user_specific` cases, `requires_document_retrieval` to distinguish
  transaction-only from transaction-plus-document questions;
- whether it is answerable from the corpus;
- expected facts or grading criteria;
- acceptable source citations, each document gold reference carrying the minimal
  evidence span (quoted text or character offsets) so chunk-level hits can be
  distinguished from mere heading overlap;
- minimal sufficient `oracle_sources` selected before model execution for each
  answerable case;
- for time-sensitive cases (e.g. booking or cancellation windows), an `as_of`
  timestamp and the transaction-fixture version, so a frozen evaluation clock
  makes the correct answer deterministic and reproducible;
- optional paraphrase-group ID;
- optional expected failure label.

Define and record a minimum number of questions per arm-by-category cell so
per-cell differences are not dominated by noise; state the chosen N in the
dataset version. Because the answer model is not fully deterministic even at
temperature 0, either run each question multiple times per arm and report mean
and variance, or explicitly document that reported metrics are single-run point
estimates. Record the number of trials per question with the results.

## Runner workflow

```text
load versioned evaluation cases
  -> stratify by question category
  -> run all cases with LLM-only
  -> run all cases against BM25
  -> run all cases against Vector RAG
  -> optionally run with oracle context
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

- run ID, question ID, timestamp, evaluation arm, question category, and status;
- corpus fingerprint and index configuration;
- transaction-fixture version and, for time-sensitive cases, the `as_of`
  evaluation clock used to resolve transaction state;
- retrieved IDs, citations, ranks, and raw scores, or explicit `N/A` values for
  arms without retrieval;
- answer and parsed citations;
- answerability and expected sources;
- retrieval and answer grades;
- failure labels;
- index, retrieval, and generation latency as applicable;
- embedding and answer token usage and estimated cost;
- exact model IDs, prompt version, generation settings, and grader version.

Produce machine-readable per-case output and a human-readable aggregate report.
The report must show each metric for every arm-by-category cell before any
overall aggregate, plus BM25 and Vector improvements over the LLM-only control.

## Corpus controls

Use a fictional company corpus as the primary benchmark so the LLM-only control
cannot rely on memorized public policies. Give it realistic but unique facts.

Real-platform policy snapshots may form a separately reported robustness tier.
Record platform, official source URL, retrieval date, locale, and content hash.
Do not mix synthetic and public-policy results because the latter may contain
pretraining contamination and policies may change over time.

## Oracle evidence contract

Oracle is an evaluation-only context-selection procedure. It uses the same
OpenAI answer model and applicable grounded prompt as the RAG arms but replaces
retrieval output with human-labeled, versioned gold evidence.

Valid Oracle evidence includes:

- platform policy Markdown sections;
- merchant, product, or property policy Markdown sections;
- allowed fields from deterministic synthetic order or booking fixtures.

Every Oracle reference must resolve to content available to the system.
Document evidence uses `filename.md#heading`; synthetic structured evidence uses
`record-type:record-id#field`. Store enough source data to audit the annotation
without storing hidden reasoning.

Annotators must label evidence before inspecting any arm's answers. Reference
answers and grader explanations are prohibited from Oracle context. Report
inter-annotator agreement or adjudication for a final benchmark when practical.

## Transaction-data control

Policy questions use document retrieval. User-specific order or booking
questions use a deterministic synthetic transaction lookup shared by both RAG
arms. The transaction result is a controlled input, not part of the BM25-versus-
Vector treatment. Live customer data is prohibited from indexes and evaluation
artifacts.

## Acceptance criteria

- One command can run the same suite against both backends.
- All question categories run against every main evaluation arm.
- The suite rejects missing, unknown, or inconsistent category labels.
- Each arm-by-category cell meets the documented minimum question count, and the
  trial count per question is recorded so point estimates and variance are
  distinguishable.
- Improvement-over-control scores are computed only on answerable categories and
  are never reported for `unsupported` or `user_specific`; `user_specific`
  results are reported separately as a RAG-only capability.
- Inputs held constant are asserted, not merely assumed.
- Per-case results retain enough evidence to audit aggregate scores.
- Every answerable Oracle case has resolvable gold evidence.
- Oracle execution never invokes BM25 or FAISS.
- Retrieval and answer metrics can be recomputed from stored results.
- A backend failure is visible and cannot be mistaken for a low-quality answer.
- The report compares both systems and lists categorized failure examples.
- The report cannot hide company-specific, user-specific, or unsupported
  behavior inside an overall score dominated by generic e-commerce questions.
