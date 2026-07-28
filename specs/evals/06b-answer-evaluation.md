# Spec 06B: Answer and Citation Evaluation

## Purpose

Evaluate answer quality across LLM-only, RAG, and optional oracle-context arms,
including groundedness and correct refusal where context is supplied.

## Inputs

- question, answerability label, and expected facts;
- category: `company_specific`, `generic_ecommerce`, `user_specific`, or
  `unsupported`;
- evaluation arm: `llm_only`, `bm25`, `vector`, or optional `oracle`;
- acceptable source citations when applicable;
- minimal sufficient Oracle evidence for answerable cases;
- exact retrieved or oracle context when applicable;
- generated answer and parsed citations;
- shared answer model and prompt configuration.

## Metrics

- **Answer correctness:** supported expected facts are stated accurately.
- **Citation accuracy:** RAG citations support the associated claims.
- **Citation validity:** every RAG citation exists and came from the supplied
  context — document citations from the top-three retrieved items, and
  structured-record citations from the shared transaction result provided in the
  same context.
- **Unsupported-claim rate:** claims not supported by reference evidence and,
  for contextual arms, by the supplied context.
- **Fallback behavior:** exact fallback on insufficient evidence and no fallback
  when sufficient evidence supports an answer.
- **Company-specific lift:** answer-quality difference between each RAG arm and
  `llm_only` on `company_specific` cases.
- **Refusal calibration:** appropriate uncertainty or fallback behavior on
  `unsupported` cases under the arm-specific prompt contract.
- **Answer latency:** model-generation time separated from retrieval time.
- **Cost:** answer token usage and configured model pricing; embedding cost is
  reported separately for Vector RAG.

## Grading rules

- Use the same deterministic graders and thresholds for both backends.
- Grade and report each evaluation-arm-by-category cell separately.
- Use the same pinned OpenAI answer model and generation settings in every arm.
- The LLM-only control receives no retrieved context and no knowledge-base
  citations. Citation and retrieval metrics are `N/A`, not zero.
- The Oracle arm receives only manually labeled gold evidence and never invokes
  BM25 or FAISS.
- BM25 and Vector use the same strict grounded prompt and exact knowledge-base
  fallback. Their citation requirements remain identical.
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
- Every `company_specific` case contains auditable corpus evidence and is graded
  for correctness across all arms and for citations in RAG arms.
- Unsupported RAG cases expect the exact knowledge-base fallback. The LLM-only
  control is graded against its separate uncertainty contract.
- Generic e-commerce results are reported separately because prior model
  knowledge may reduce the measured RAG improvement.
- Fabricated and non-retrieved citations always fail citation validity.
- A correct-looking answer with unsupported facts fails groundedness.
- An answer can be graded independently from retrieval Recall@3.
- Every aggregate metric links back to auditable per-question results.
- Oracle success with RAG failure is classified as likely retrieval failure;
  failure after gold evidence was retrieved is classified as likely generation
  or prompt failure; Oracle failure triggers annotation/model/prompt review.
