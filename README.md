# Knowledge Base Q&A RAG Agent

A small, source-grounded e-commerce question-answering system over Markdown
documents. The project compares a transparent keyword-retrieval baseline with a
semantic vector retrieval pipeline while keeping the rest of the experiment
consistent.

> [!NOTE]
> The `scaffold` branch contains a runnable, offline-tested implementation
> foundation. A real OpenAI evaluation run and a human-readable comparison
> report remain explicit follow-up work; this scaffold is not benchmark evidence.

### Current status and next phase

The initial scaffold already contains both retrieval adapters because it was
created before the delivery sequence was revised. The specifications now govern
subsequent implementation and refactoring in this order:

```text
shared contracts + seed dataset
  -> finish and verify Vector RAG
  -> build the backend-neutral UI
  -> reconcile and verify Markdown KB
  -> complete the evaluation runner
  -> expand and freeze the final dataset
```

The Vector phase has deterministic chunking and ranking, complete persisted
configuration validation, restart restoration without document re-embedding,
and phase-owned offline tests.

The Markdown-specific reconciliation is complete and documented under
[`specs/005-markdown-kb/`](specs/005-markdown-kb/): heading-hierarchy parsing,
GitHub-style anchors with deterministic duplicate suffixes, an inspectable JSON
index with load-time validation, and phase-owned parser, retrieval, persistence,
and shared-contract tests.

## Retrieval strategies

| System | Document unit | Retrieval |
| --- | --- | --- |
| **Markdown KB** | Sections split at Markdown headings | BM25 keyword search |
| **Vector RAG** | Fixed-size or token-aware chunks | Embeddings and FAISS similarity search |

Both systems must use the same documents, questions, answer model, grounded
prompt, citation format, and graders. Only the retrieval strategy should differ.

## Evaluation comparisons

The experiment answers two primary questions:

1. **Retrieval framework:** How does Markdown KB with BM25 compare with Vector
   RAG using embeddings and FAISS?
2. **Specialist versus general model:** How much does either e-commerce RAG
   system improve the same OpenAI model over an LLM-only, closed-book control?

The production bot has two retrieval backends. The evaluation runner adds a
third, non-production control arm that bypasses retrieval:

| Evaluation arm | Context supplied to the OpenAI model | Purpose |
| --- | --- | --- |
| LLM-only control | Question only | General-model baseline |
| Markdown KB RAG | Top 3 BM25 heading sections | Keyword-retrieval treatment |
| Vector RAG | Top 3 FAISS chunks | Semantic-retrieval treatment |
| Oracle context, optional | Manually verified correct sections | Answer-generation upper bound |

All main arms use the same pinned OpenAI answer model, questions, generation
settings, reference answers, and graders. The main benchmark uses the same
e-commerce questions across arms; it does not compare unrelated question sets.
The optional oracle arm runs only on answerable cases; there is no correct
oracle context for `unsupported` questions, and it bounds answer generation
given perfect retrieval rather than serving as an absolute ceiling.

> **Confound to report, not hide.** The LLM-only arm necessarily uses a
> different prompt from the RAG arms (it may use prior knowledge, states
> uncertainty instead of the strict knowledge-base fallback, and cites nothing).
> So any RAG-minus-control "improvement" measures *grounded prompt plus
> retrieved context* against *control prompt with no context*, not retrieved
> context in isolation. Report this as a known limitation rather than claiming a
> clean context-only ablation.

Use four question categories:

- `company_specific`: store policies and facts that a general model should not
  reliably know;
- `generic_ecommerce`: general e-commerce concepts that the base model may know;
- `user_specific`: a specific user's order or booking, answerable only through
  the shared synthetic transaction lookup;
- `unsupported`: facts not established by the corpus, used to measure
  hallucination and refusal calibration.

`user_specific` questions are a RAG-only capability: the `llm_only` control has
no transaction access and cannot answer them by construction, so they are
excluded from the improvement-over-control (lift) metric and reported separately
as a capability the RAG arms have and the control does not. They split in two:
**transaction-only** questions (e.g. "What is the status of booking BK-10023?")
need no document retrieval and are not a BM25-versus-Vector comparison, while
**transaction-plus-document** questions (e.g. "My order arrived late — am I owed
a refund?") do retrieve a policy section and, with the transaction result held
identical across arms, remain a valid BM25-versus-Vector comparison of the
document-retrieval portion.

For a clean causal comparison, the primary corpus should describe a fictional
company with realistic but unique policies. A secondary robustness corpus may
use dated snapshots from official public Amazon, Walmart, eBay, or other
platform documentation. Real-platform results must remain separate because the
base model may have memorized public policies.

> **Caution:** capturing and especially redistributing platform policy pages can
> violate their terms of service, and is a separate concern from using a private
> snapshot for local evaluation. Review each platform's terms before committing
> any captured content to this repository; prefer referencing snapshots by
> source URL, retrieval date, locale, and content hash over checking in the
> verbatim text.

### What Oracle means

Oracle is not another model and is not a production retriever. Before running
the systems, a human annotator records the minimal correct evidence for each
answerable evaluation question. The Oracle arm bypasses BM25 and FAISS and gives
that verified evidence directly to the same OpenAI answer model.

For an e-commerce or booking support agent, Oracle evidence can come from:

| Question type | Evidence source | Example reference |
| --- | --- | --- |
| Platform-general | Platform policy Markdown | `platform/refunds.md#refund-timeline` |
| Merchant, product, or property-specific | Seller, product, or hotel policy Markdown | `properties/hotel-101.md#pet-policy` |
| User order or booking-specific | Deterministic synthetic transaction lookup | `booking:BK-10023#status` |

```text
Question
├── LLM-only: question only
├── BM25: top 3 retrieved sections
├── Vector: top 3 retrieved chunks
└── Oracle: manually labeled correct evidence
```

Oracle diagnoses where quality is lost:

- Oracle succeeds but RAG fails: retrieval is the likely failure.
- RAG retrieves the gold evidence but answers incorrectly: answer generation or
  prompt handling is the likely failure.
- Oracle fails: inspect the model, prompt, annotation, or reference answer.

Annotators must select Oracle evidence before inspecting model outputs. They
must never write an ideal answer and pass that answer back as Oracle context.

### Policy and transaction data

Static RAG is appropriate for platform and merchant/property policies. Live
customer order status must come from an authenticated transaction API, not the
Markdown or FAISS index. The prototype uses deterministic synthetic order or
booking fixtures through a shared lookup interface; production customer data is
out of scope.

BM25 and Vector must receive identical transaction lookup results so their
comparison still changes only document retrieval. Knowledge-base documents use
`filename.md#heading` citations. Synthetic structured records use
`record-type:record-id#field` references and must not expose private data.

## Project specifications

This repository is one comparison project composed of six major parts. Each
part has its own specification so it can be implemented and verified without
mixing retrieval-specific behavior into shared code.

| Part | Responsibility | Specification |
| --- | --- | --- |
| 1. Shared Q&A core | Common API, retrieval contract, grounded prompt, citations, fallback, and OpenAI answer generation | [`specs/01-shared-qa-core.md`](specs/01-shared-qa-core.md) |
| 2. Seed evaluation dataset | Frozen pre-implementation questions, gold evidence, and Oracle labels | [`specs/02-seed-evaluation-dataset.md`](specs/02-seed-evaluation-dataset.md) |
| 3. Vector RAG | First retrieval vertical slice: chunks, embeddings, FAISS, persistence, and tests | [`specs/03-vector-rag.md`](specs/03-vector-rag.md) |
| 4. User interface | Backend-neutral API client for indexing, questions, answers, and source inspection | [`specs/04-user-interface.md`](specs/04-user-interface.md) |
| 5. Markdown KB | Heading parsing, plain BM25 retrieval, JSON persistence, and tests | [`specs/05-markdown-kb.md`](specs/05-markdown-kb.md) |
| 6. Evaluation runner | Controlled execution, blind dataset expansion, metrics, and comparison reports | [`specs/06-evaluation-runner.md`](specs/06-evaluation-runner.md) |

The evaluation runner contains two smaller evaluation tracks:

1. **Retrieval evaluation** — Recall@K, retrieval failure labels, paraphrase
   robustness, and retrieval latency. See
   [`specs/evals/06a-retrieval-evaluation.md`](specs/evals/06a-retrieval-evaluation.md).
2. **Answer and citation evaluation** — correctness, citation accuracy,
   hallucination, fallback behavior, answer latency, and model cost. See
   [`specs/evals/06b-answer-evaluation.md`](specs/evals/06b-answer-evaluation.md).

```text
Shared core + frozen seed dataset
  -> Vector RAG + tests
  -> backend-neutral UI + tests
  -> Markdown KB + tests
  -> evaluation runner
  -> blind dataset expansion and freeze
  -> final controlled evaluation
```

This is a delivery order, not an experimental preference. Vector is implemented
first for learning, while the final comparison still holds non-retrieval inputs
constant and treats both backends symmetrically.

## Development methodology

This repository is also a practice project for AI-native, spec-driven
development. Implementation and later refactoring should follow the
[GitHub Spec Kit](https://github.com/github/spec-kit) methodology: establish a
project constitution, specify behavior, clarify ambiguity, create a technical
plan, break the plan into reviewable tasks, analyze artifact consistency, and
only then implement and validate the code.

The specifications above define the current product boundaries and acceptance
criteria. When Spec Kit is initialized, use them as inputs to feature-level
`spec.md`, `plan.md`, and `tasks.md` artifacts rather than asking an AI coding
agent to generate the entire system from one unconstrained prompt.

The intended practice loop is:

```text
constitution
  -> specify
  -> clarify
  -> plan
  -> tasks
  -> analyze
  -> implement or refactor
  -> tests and evaluation evidence
```

Each implementation or refactoring change should be traceable to a specification
requirement and task, preserve the controlled-comparison constraints, and finish
with automated tests plus evaluation evidence. Human review owns scope and
acceptance; the AI agent helps turn those decisions into versioned artifacts and
code. See [`SPEC_KIT_WORKFLOW.md`](SPEC_KIT_WORKFLOW.md) for the project-specific
workflow.

## System flow

```text
Question
  -> retrieve the top 3 sections or chunks
  -> look up a synthetic transaction record when required
  -> build a prompt from the retrieved context
  -> generate an answer with the OpenAI chat model
  -> return the answer with auditable source references
```

The model must answer only from the retrieved context. Every supported claim
must cite its source. Knowledge-base evidence uses:

```text
filename.md#heading
```

Synthetic structured transaction evidence uses:

```text
record-type:record-id#field
```

If the retrieved context does not contain enough information, the response must
be exactly:

```text
I cannot confirm from the knowledge base.
```

## Model acknowledgement

The only answer-generating LLM used by this project is the OpenAI chat model
`gpt-5.6-sol`. No other LLM provider or answer model is used. The Vector RAG
pipeline also requires an embedding model for retrieval; embeddings are
retrieval features and do not generate answers.

The default embedding model is `text-embedding-3-small`. Both retrieval systems
use the same pinned answer model; the embedding model is used only by Vector RAG
to create retrieval features.

The initial Vector RAG configuration is deliberately fixed and auditable:
whitespace-word chunks contain 160 words with 30 words of overlap, document and
query vectors are L2-normalized, and FAISS `IndexFlatIP` ranks their inner
product (cosine similarity after normalization). The persisted metadata records
this configuration, the returned vector dimension, and the embedding model.

## Run the scaffold

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, select `RETRIEVAL_BACKEND=bm25` or `vector`,
then start the API:

```bash
.venv/bin/uvicorn kbqa.api:app --reload
curl -X POST http://127.0.0.1:8000/index
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"How long does a NovaStay refund take?"}'
```

Run offline tests without an OpenAI key:

```bash
.venv/bin/pytest
```

Validate the frozen seed dataset without calling an LLM or retriever:

```bash
.venv/bin/python -m kbqa.evals.dataset \
  --manifest evals/seed-v1.manifest.json \
  --docs docs \
  --transactions fixtures/bookings.json
```

The manifest pins seed version `seed-v1`, its annotation date, the exact
`cases.jsonl` SHA-256, the corpus fingerprint, and transaction fixture
`bookings-v1`. A factual annotation correction requires a new dataset version,
an updated `change_reason`, and reviewed hashes; cases must not be changed in
response to backend outputs.

The real Vector embedding smoke test is deliberately opt-in:

```bash
RUN_OPENAI_INTEGRATION=1 OPENAI_API_KEY="sk-..." \
  .venv/bin/pytest tests/test_vector_integration.py
```

Run the machine-readable comparison after configuring an OpenAI key:

```bash
.venv/bin/kbqa-eval \
  --dataset evals/seed-v1.manifest.json \
  --output results/eval.jsonl \
  --arms llm_only bm25 vector oracle
```

These commands exercise the current API scaffold. UI setup and commands will be
added during Spec 04 implementation; the README must not advertise an
unimplemented browser interface.

## API

Both retrieval implementations expose the same FastAPI interface.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Check server health |
| `POST` | `/index` | Index all Markdown files in `docs/` |
| `POST` | `/chat` | Retrieve sources and generate a grounded answer |

Keeping the interface identical allows the evaluation harness to switch between
retrievers without changing test questions or grading logic.

## Persistence

Indexes are stored locally and must load automatically when the server restarts.

| System | Index location |
| --- | --- |
| Markdown KB | `.kb/index.json` |
| Vector RAG | `.kb/faiss_index/` |

Calling `/index` rebuilds the relevant index from the Markdown files in `docs/`.

The Markdown KB index is intentionally inspectable JSON. It records complete
heading-section records (including heading level and hierarchy), stable IDs,
canonical citations, the corpus fingerprint, schema/parser/tokenizer versions,
the unmodified `rank_bm25.BM25Okapi` parameters, and the tokenized corpus needed
for deterministic restoration. Incompatible or incomplete index files are
reported as unavailable and are never silently rebuilt during chat.

## Evaluation focus

The goal is to measure retrieval-framework differences and the improvement over
an LLM-only control, not to optimize either retriever until its characteristic
failure modes disappear.

Evaluate both systems on:

- answer correctness
- retrieval Recall@K
- citation accuracy
- hallucination rate
- correct fallback behavior
- paraphrase robustness
- latency
- cost

Report metrics by evaluation arm and question category. Do not hide
company-specific, generic e-commerce, user-specific, or unsupported behavior
inside one aggregate score.

Retrieval and citation metrics apply only to the RAG arms. Record them as `N/A`,
not zero, for the LLM-only control. Compare answer quality across all arms, then
calculate the treatment improvements on **answerable categories only**
(primarily `company_specific`, where the base model cannot rely on memorized
facts):

```text
BM25 improvement   = BM25 metric   - LLM-only metric   (company_specific)
Vector improvement = Vector metric - LLM-only metric   (company_specific)
```

Do not compute an improvement score on `unsupported` cases: the correct answer
there is refusal, not stated facts, and the control and RAG arms are graded
under different contracts, so subtracting them is not meaningful. Likewise
exclude `user_specific` cases: the control cannot access the transaction lookup,
so any "lift" there is trivial and is instead reported as a RAG-only capability.
On `generic_ecommerce`, expect a smaller lift because the base model may already
know the answer; report it separately rather than folding it into the headline.

BM25 remains intentionally simple and transparent. Synonym misses and keyword
false positives should be recorded as evaluation outcomes rather than hidden by
retrieval-specific tuning.

### Suggested failure labels

| Markdown KB / BM25 | Vector RAG / FAISS |
| --- | --- |
| `synonym_miss` | `semantic_false_positive` |
| `keyword_false_positive` | `chunk_boundary_failure` |
| `weak_retrieval` | `missing_exact_term` |

## Fair-comparison requirements

For a valid comparison, hold these inputs constant:

- the Markdown document corpus and its version
- the evaluation questions and reference answers
- the same labeled balance of `company_specific`, `generic_ecommerce`,
  `user_specific`, and `unsupported` questions
- `K = 3`
- the OpenAI answer model (`gpt-5.6-sol`)
- the grounded-answer prompt and fallback text
- the citation formats (document `filename.md#heading` and structured-record
  `record-type:record-id#field`)
- generation settings such as temperature and maximum output tokens
- graders and scoring thresholds

Record corpus version, index time, retrieval latency, answer latency, token
usage, and model cost for each run. Version the evaluation set separately from
the indexed documents to avoid accidental test leakage.

## Reference and inspiration

This project was inspired in part by Andrej Karpathy's
[LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f),
which describes a Markdown-centered, persistent knowledge base maintained and
queried with an LLM.

This repository has a narrower purpose: it is a read-only, source-grounded Q&A
evaluation comparing BM25 retrieval with embeddings plus FAISS. It does not
claim to implement the gist's full incremental wiki-maintenance workflow.
