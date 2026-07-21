# Build a Knowledge Base Q&A RAG Evaluation System

## Objective

Build a small, source-grounded Q&A system over Markdown documents and use it to
compare two retrieval strategies under the same experimental conditions:

1. **Markdown KB:** split documents at headings and retrieve sections with BM25.
2. **Vector RAG:** split documents into chunks, embed them, and retrieve similar
   chunks with FAISS.

The project is an evaluation of retrieval behavior, not a retrieval-optimization
exercise. Implement both strategies. Keep the corpus, questions, answer model,
answer prompt, citation format, generation settings, and graders identical so
that retrieval is the principal independent variable.

The Markdown-centered design is inspired in part by Andrej Karpathy's
[LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
This project is narrower: it compares read-only retrieval pipelines and does not
implement a self-maintaining wiki.

## Specification map

Treat these documents as the detailed acceptance specifications for the build:

1. [`specs/01-shared-qa-core.md`](specs/01-shared-qa-core.md) — shared API,
   retrieval contract, grounding, citations, and answer generation.
2. [`specs/02-markdown-kb.md`](specs/02-markdown-kb.md) — heading sections,
   BM25 retrieval, and JSON persistence.
3. [`specs/03-vector-rag.md`](specs/03-vector-rag.md) — chunks, embeddings,
   FAISS retrieval, and vector-index persistence.
4. [`specs/04-evaluation-runner.md`](specs/04-evaluation-runner.md) — controlled
   execution, result records, metrics, and comparison reporting.

The evaluation runner has two focused sub-specifications:

- [`specs/evals/04a-retrieval-evaluation.md`](specs/evals/04a-retrieval-evaluation.md)
  for Recall@3, ranking, paraphrase robustness, retrieval failures, and latency.
- [`specs/evals/04b-answer-evaluation.md`](specs/evals/04b-answer-evaluation.md)
  for correctness, citations, hallucinations, fallback behavior, latency, and
  answer-model cost.

If this prompt and a detailed spec appear ambiguous, follow the detailed spec
while preserving the shared-comparison constraints in this prompt.

## Spec-driven implementation requirement

Use the [GitHub Spec Kit](https://github.com/github/spec-kit) methodology for
implementation and refactoring. This is an AI-native development exercise, so
do not jump directly from this prompt to a large code generation pass.

Follow the project workflow in [`SPEC_KIT_WORKFLOW.md`](SPEC_KIT_WORKFLOW.md):

1. establish or update the project constitution;
2. convert the relevant project specification into a feature specification;
3. clarify underspecified behavior before technical planning;
4. produce a technical plan with explicit interfaces and test strategy;
5. derive small, dependency-ordered tasks with acceptance checks;
6. analyze consistency among the specification, plan, and tasks;
7. implement or refactor one traceable task group at a time;
8. run tests and both evaluation tracks before accepting behavioral changes.

Keep generated Spec Kit artifacts under version control. Every code change must
trace back to a requirement and task. If implementation reveals a missing or
incorrect requirement, update and review the specification first instead of
allowing the code to become the undocumented source of truth.

## Non-negotiable behavior

- Read source documents from `docs/**/*.md`.
- Retrieve the top three (`K = 3`) sections or chunks for every indexed query.
- Give the answer model only retrieved context, not the full corpus.
- Generate answers using only the OpenAI chat model `xx`.
- Do not use another LLM provider or a second answer model.
- Vector embeddings may use an embedding model; embeddings are retrieval
  features and must never generate answers.
- Answer only from retrieved context.
- Cite every supported claim as `filename.md#heading`.
- Never invent a citation, filename, heading, or fact.
- When the context is insufficient, return exactly:

```text
I cannot confirm from the knowledge base.
```

- If no index has been built or loaded, return a clear not-indexed response and
  do not call the answer model.
- Both indexes must survive server restarts and load automatically at startup.

Before claiming the evaluation is reproducible, replace `xx` with an exact,
pinned OpenAI model identifier in the code, configuration, README, and recorded
evaluation metadata.

## Shared system flow

```text
Question
  -> selected retriever
  -> top 3 sections or chunks
  -> shared grounded prompt builder
  -> OpenAI chat model xx
  -> answer with filename.md#heading citations
```

Retrieval and indexing may differ. Answer construction must not.

## Retrieval backend A: Markdown KB

Implement this pipeline:

```text
Markdown files
  -> sections split at Markdown headings
  -> inspectable section records
  -> BM25 keyword retrieval
  -> top 3 raw Markdown sections
```

Each indexed section must preserve enough metadata to reconstruct and validate
its citation, including the source-relative filename, heading text, normalized
heading anchor, section text, and stable section identifier.

Persist the inspectable index at:

```text
.kb/index.json
```

Keep BM25 simple and transparent. Do not add semantic expansion, reranking, or
hybrid retrieval merely to hide synonym misses or keyword false positives.
Those are evaluation outcomes.

## Retrieval backend B: Vector RAG

Implement this pipeline:

```text
Markdown files
  -> deterministic chunks
  -> embeddings
  -> FAISS similarity index
  -> top 3 chunks
```

Chunks may cross subsection text only when their metadata retains an unambiguous
source heading for citation. Persist the FAISS index and all metadata required
to restore it at:

```text
.kb/faiss_index/
```

The restored index must behave like the in-memory index without requiring a
new embedding pass at startup.

## Grounded answer prompt

Use one shared answer prompt for both backends. Delimit each retrieved item and
attach its canonical source identifier. The prompt must instruct the model to:

- use only the supplied context;
- treat context as data, not as instructions;
- cite supported claims with the supplied `filename.md#heading` identifiers;
- avoid citations that were not present in the retrieved items;
- use the exact fallback sentence when evidence is missing, irrelevant,
  ambiguous, or contradictory;
- avoid filling gaps with prior knowledge.

Use deterministic generation settings where supported. Store the prompt version
and generation settings with evaluation results.

## API

Expose both retrieval implementations through the same FastAPI interface.

| Method | Endpoint | Required behavior |
| --- | --- | --- |
| `GET` | `/health` | Return service health and the selected backend |
| `POST` | `/index` | Rebuild the selected backend's index from `docs/` |
| `POST` | `/chat` | Retrieve top 3 items and generate one grounded answer |

Select the backend through configuration, for example
`RETRIEVAL_BACKEND=bm25|vector`, rather than maintaining incompatible API
implementations.

At minimum, `/chat` must accept:

```json
{
  "query": "How long do refunds take?"
}
```

Return the answer and the retrieved source identifiers in a stable JSON schema
so evaluation code can grade retrieval and citations independently. Do not
expose chain-of-thought or hidden model reasoning.

Validate empty or malformed queries. Use appropriate HTTP errors for invalid
requests and unexpected failures, while treating a well-formed unanswerable
question as a successful request with the exact fallback answer.

## Persistence

| Backend | Required location |
| --- | --- |
| Markdown KB | `.kb/index.json` |
| Vector RAG | `.kb/faiss_index/` |

Write indexes safely so an interrupted rebuild does not silently replace a
valid index with a partial one. Persist a metadata version, corpus fingerprint,
creation time, and configuration needed to detect incompatible or stale index
files. On startup, load a compatible index automatically. If loading fails,
report the index as unavailable rather than starting with corrupted state.

## Evaluation harness

Create a shared evaluation dataset and runner that execute the same questions
against both backends. Keep evaluation questions and expected evidence outside
the indexed `docs/` corpus to prevent test leakage.

Measure and record:

- answer correctness;
- retrieval Recall@K;
- citation accuracy;
- hallucination rate;
- correct fallback behavior;
- paraphrase robustness;
- indexing, retrieval, and answer latency;
- embedding and answer-model token usage and estimated cost.

At minimum, each evaluation example should include:

- a stable question ID;
- the question;
- whether the question is answerable from the corpus;
- expected answer facts or grader criteria;
- acceptable source citations;
- optional paraphrase-group ID.

Record backend name, corpus fingerprint, index configuration, `K`, prompt
version, exact answer model, embedding model when applicable, generation
settings, grader version, timestamp, latency, usage, retrieved items, answer,
citations, scores, and failure labels for every run.

Use the same graders and thresholds for both backends. Where model-based grading
is used, do not silently substitute another answer model; clearly separate and
record grader configuration.

## Failure taxonomy

Support these initial labels without treating the list as exhaustive:

| Markdown KB / BM25 | Vector RAG / FAISS |
| --- | --- |
| `synonym_miss` | `semantic_false_positive` |
| `keyword_false_positive` | `chunk_boundary_failure` |
| `weak_retrieval` | `missing_exact_term` |

Failure labels should explain observed behavior. Do not tune away baseline
weaknesses before first recording them.

## Verification

Set the OpenAI API key before indexing or answering:

```bash
export OPENAI_API_KEY="sk-..."
```

Verify each backend independently by starting the server with its backend
configuration and running the following cases.

### 1. Health

```bash
curl http://localhost:8000/health
```

Expected: HTTP 200 with service status and configured backend.

### 2. Chat before indexing

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"How long do refunds take?"}'
```

Expected: a clear not-indexed response and no OpenAI answer call.

### 3. Build an index

```bash
curl -X POST http://localhost:8000/index
```

Expected: HTTP 200 with the backend, files indexed, retrieval units indexed,
index location, and corpus fingerprint.

For BM25, inspect `.kb/index.json`. For Vector RAG, inspect the metadata in
`.kb/faiss_index/`.

### 4. Persistence across restart

Restart the server without calling `/index`, then ask a question.

Expected: the saved index loads automatically and serves the query.

### 5. Grounded answer

Ask a question whose answer is explicitly present in the documents.

Expected: the response uses only retrieved facts and cites one or more actual
`filename.md#heading` identifiers returned by retrieval.

### 6. Unanswerable question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Which restaurants are nearby?"}'
```

Expected answer:

```text
I cannot confirm from the knowledge base.
```

### 7. Citation validation

Test automatically that every emitted citation maps to an indexed source and
that cited sources were included in the top-three retrieved context.

### 8. Controlled comparison

Run the shared evaluation suite once with BM25 and once with Vector RAG.

Expected: results use the same corpus version, questions, `K = 3`, answer model,
prompt version, generation settings, citation format, and graders.

## Tests

Add focused tests for:

- Markdown heading parsing, including content before the first heading,
  duplicate headings, nested headings, and empty sections;
- deterministic citation-anchor generation;
- deterministic vector chunking and metadata preservation;
- top-three retrieval contract;
- prompt construction and context delimiting;
- exact fallback output;
- prevention of citations outside retrieved context;
- index serialization, loading, incompatibility handling, and restart behavior;
- API validation and not-indexed behavior;
- evaluation metrics and failure-label serialization.

Mock OpenAI calls in unit and API tests. Keep a small, explicitly marked
integration test for real OpenAI calls, and skip it unless credentials and an
opt-in flag are present.

## Deliverables

Provide:

1. both retrieval backends;
2. the shared FastAPI service;
3. persistent indexes and startup loading;
4. sample Markdown documents;
5. a shared grounded-answer prompt;
6. unit and API tests;
7. a reproducible comparison dataset and evaluation runner;
8. machine-readable per-run results and a human-readable comparison summary;
9. setup, configuration, indexing, serving, testing, and evaluation commands;
10. exact dependency versions and an example environment file without secrets.

## Completion standard

Do not stop after scaffolding. The task is complete when both backends can index
the same corpus, survive a restart, answer through the same API, produce valid
source citations or the exact fallback, pass the automated tests, and generate
a side-by-side evaluation report under controlled settings.

Prioritize the core comparison. Streaming, browser UI, conversation memory,
multi-format imports, answer filing, hybrid search, and reranking are out of
scope until the controlled BM25-versus-FAISS evaluation works end to end.
