# Spec 04A: Retrieval Evaluation

## Purpose

Evaluate whether each backend retrieves the evidence required to answer a
question, independently of answer-model behavior.

## Inputs

- question and stable question ID;
- answerability label;
- acceptable source citation set;
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
- Citation aliases are normalized before matching but source identity is not
  weakened to arbitrary filename-only matching.
- Unanswerable questions are not incorrectly rewarded for retrieving any source.
- Every metric can be reproduced from stored question and retrieval records.
