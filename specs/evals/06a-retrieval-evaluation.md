# Spec 06A: Retrieval Evaluation

## Purpose

Evaluate whether each backend retrieves the evidence required to answer a
question, independently of answer-model behavior.

## Inputs

- question and stable question ID;
- category: `company_specific`, `generic_ecommerce`, `user_specific`, or
  `unsupported`;
- answerability label;
- acceptable source citation set;
- labeled Oracle document-source set for answerable cases;
- ordered top-three retrieval results;
- backend name and retrieval configuration;
- optional paraphrase-group ID.

## Metrics

- **Recall@3:** fraction of a question's required or acceptable sources that
  appear in the top three.
- **Hit rate:** whether at least one acceptable source appears in the top three
  (binary any-hit).
- **Rank of first relevant result:** position of the first acceptable source.
- **Paraphrase robustness:** consistency of relevant retrieval across equivalent
  question phrasings.
- **Retrieval latency:** measured separately from answer generation.
- **Unsupported retrieval rate:** percentage of `unsupported` cases for which
  retrieval returns topically-plausible or above-threshold context despite the
  corpus containing no supporting source. Judge this at the retrieval level
  (score threshold or relevance judgment on the returned units), independently
  of the answer model, to preserve the answer-model-free grading requirement
  below.

Do not compare raw BM25 and FAISS scores as if they shared a scale. Compare
ranked relevance outcomes instead.

## Failure labels

Apply labels only when supported by retrieved results and reference evidence.

Markdown KB:

- `synonym_miss`
- `keyword_false_positive`
- `weak_retrieval`

Vector RAG:

- `semantic_false_positive`
- `chunk_boundary_failure`
- `missing_exact_term`

Unknown or cross-cutting failures may use a documented additional label rather
than forcing an incorrect predefined category.

## Acceptance criteria

- Retrieval grading can run without calling the answer model.
- Retrieval metrics apply to `bm25` and `vector`; they are `N/A` for the
  `llm_only` control and optional `oracle` arm.
- Citation aliases are normalized before matching but source identity is not
  weakened to arbitrary filename-only matching.
- Unanswerable questions are not incorrectly rewarded for retrieving any source.
- Unsupported cases are evaluated for irrelevant or misleading
  retrieval rather than Recall@3 against a nonexistent relevant source.
- Metrics are reported by backend and question category, not only in aggregate.
- Every metric can be reproduced from stored question and retrieval records.
- Oracle document references provide gold relevance labels for Recall@3.
  Structured transaction references are excluded because they come from the
  shared lookup rather than BM25 or FAISS.
- Match retrieved units to gold at the source-heading level so BM25 sections and
  Vector chunks are comparable against the same `filename.md#heading` labels:
  each retrieved chunk is mapped to its source heading before matching. Because a
  heading can span several chunks and only some contain the answer, a
  heading-only label cannot identify which chunk holds the evidence, so each gold
  reference must also carry the minimal evidence span (quoted text or character
  offsets). A retrieved chunk counts as a full hit only when it contains that
  span; a chunk that shares the heading but omits the span is recorded as a
  weaker section-level match, not a full hit. Report both granularities so Vector
  recall is not over-credited by heading overlap alone.
