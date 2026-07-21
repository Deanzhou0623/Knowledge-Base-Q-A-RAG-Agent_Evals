# Knowledge Base Q&A RAG Agent

A small, source-grounded question-answering system over Markdown documents. The
project compares a transparent keyword-retrieval baseline with a semantic vector
retrieval pipeline while keeping the rest of the experiment consistent.

> [!NOTE]
> This repository currently contains the project specification only. The service
> implementation and runnable configuration still need to be added.

## Retrieval strategies

| System | Document unit | Retrieval |
| --- | --- | --- |
| **Markdown KB** | Sections split at Markdown headings | BM25 keyword search |
| **Vector RAG** | Fixed-size or token-aware chunks | Embeddings and FAISS similarity search |

Both systems must use the same documents, questions, answer model, grounded
prompt, citation format, and graders. Only the retrieval strategy should differ.

## Project specifications

This repository is one comparison project composed of four major parts. Each
part has its own specification so it can be implemented and verified without
mixing retrieval-specific behavior into shared code.

| Part | Responsibility | Specification |
| --- | --- | --- |
| 1. Shared Q&A core | Common API, retrieval contract, grounded prompt, citations, fallback, and OpenAI answer generation | [`specs/01-shared-qa-core.md`](specs/01-shared-qa-core.md) |
| 2. Markdown KB | Heading parsing, BM25 retrieval, and `.kb/index.json` persistence | [`specs/02-markdown-kb.md`](specs/02-markdown-kb.md) |
| 3. Vector RAG | Chunking, embeddings, FAISS retrieval, and `.kb/faiss_index/` persistence | [`specs/03-vector-rag.md`](specs/03-vector-rag.md) |
| 4. Evaluation runner | Controlled execution of both backends, metrics, failure labels, and comparison reports | [`specs/04-evaluation-runner.md`](specs/04-evaluation-runner.md) |

The evaluation runner contains two smaller evaluation tracks:

1. **Retrieval evaluation** — Recall@K, retrieval failure labels, paraphrase
   robustness, and retrieval latency. See
   [`specs/evals/04a-retrieval-evaluation.md`](specs/evals/04a-retrieval-evaluation.md).
2. **Answer and citation evaluation** — correctness, citation accuracy,
   hallucination, fallback behavior, answer latency, and model cost. See
   [`specs/evals/04b-answer-evaluation.md`](specs/evals/04b-answer-evaluation.md).

```text
Shared Q&A core
├── Markdown KB retriever
├── Vector RAG retriever
└── Evaluation runner
    ├── Retrieval evaluation
    └── Answer and citation evaluation
```

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
  -> build a prompt from the retrieved context
  -> generate an answer with the OpenAI chat model
  -> return the answer with filename.md#heading citations
```

The model must answer only from the retrieved context. Every supported claim
must cite its source using this format:

```text
filename.md#heading
```

If the retrieved context does not contain enough information, the response must
be exactly:

```text
I cannot confirm from the knowledge base.
```

## Model acknowledgement

The only answer-generating LLM used by this project is the OpenAI chat model
`xx`. No other LLM provider or answer model is used. The Vector RAG pipeline
also requires an embedding model for retrieval; embeddings are retrieval
features and do not generate answers.

Before running a reproducible evaluation, replace `xx` with the exact pinned
OpenAI model identifier in both the implementation and this README. Both
retrieval systems must use that same pinned answer model.

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

## Evaluation focus

The goal is to compare retrieval strategies, not to optimize either retriever
until its characteristic failure modes disappear.

Evaluate both systems on:

- answer correctness
- retrieval Recall@K
- citation accuracy
- hallucination rate
- correct fallback behavior
- paraphrase robustness
- latency
- cost

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
- `K = 3`
- the OpenAI answer model (`xx`)
- the grounded-answer prompt and fallback text
- the citation format
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
