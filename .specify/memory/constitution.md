# Project Constitution

## Principles

1. Answers are source-grounded; unsupported claims fail closed.
2. BM25 and Vector RAG share API, prompt, model, K, and graders.
3. Evaluation artifacts are versioned and reproducible.
4. BM25 remains a transparent baseline without semantic enhancements.
5. Live customer data never enters indexes, prompts, logs, or fixtures.
6. AI-authored changes trace to a reviewed spec, plan, task, and test.

## Quality gates

- Unit tests do not require network access or OpenAI credentials.
- Persistent artifacts are written atomically and validated before loading.
- Every citation resolves to context supplied to that request.
- Evaluation reports preserve per-arm and per-category results.
