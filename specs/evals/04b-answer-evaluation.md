# Spec 04B: Answer and Citation Evaluation

## Purpose

Evaluate final grounded-answer quality after retrieval, including correct refusal
when the knowledge base does not support an answer.

## Inputs

- question, answerability label, and expected facts;
- acceptable source citations;
- exact top-three retrieved context;
- generated answer and parsed citations;
- shared answer model and prompt configuration.

## Metrics

- **Answer correctness:** supported expected facts are stated accurately.
- **Citation accuracy:** cited sources support the associated claims.
- **Citation validity:** every citation exists and came from retrieved context.
- **Hallucination rate:** claims unsupported by retrieved context.
- **Fallback behavior:** exact fallback on insufficient evidence and no fallback
  when sufficient evidence supports an answer.
- **Answer latency:** model-generation time separated from retrieval time.
- **Cost:** answer token usage and configured model pricing; embedding cost is
  reported separately for Vector RAG.

## Grading rules

- Use the same deterministic graders and thresholds for both backends.
- Prefer exact checks for fallback text, citation existence, and citation
  membership in retrieved context.
- Keep reference-based correctness grading separate from context-groundedness.
- If an LLM grader is added, pin and record its OpenAI model, prompt, and
  settings. Do not treat its judgment as an unversioned ground truth.
- Store grader explanations intended for auditing, but never request or expose
  hidden chain-of-thought.

## Acceptance criteria

- The exact fallback is checked byte-for-byte after documented whitespace
  normalization.
- Fabricated and non-retrieved citations always fail citation validity.
- A correct-looking answer with unsupported facts fails groundedness.
- An answer can be graded independently from retrieval Recall@3.
- Every aggregate metric links back to auditable per-question results.
