# Live evaluation review

- Run ID: `61f28d3f-7134-422d-bbde-718344a265d6`
- Dataset: `seed-v1.1`
- Records: 23 successful, 0 failed
- Grader: deterministic lexical `deterministic-v2`
- Trials: 1 per question
- Answer model: `gpt-5.6-sol`
- Embedding model: `text-embedding-3-small`

## Main observations

- BM25 and Vector achieved full answer correctness on every answerable RAG case.
- Both achieved Recall@3 of 1.0 on the company-specific refund paraphrases and
  the generic chargeback case.
- Vector retrieved the gold hotel cancellation policy for `booking-002` at rank
  1. BM25 did not retrieve that policy in its top three.
- Both RAG arms still answered `booking-002` correctly because the shared
  structured transaction result directly supplied confirmation and free
  cancellation availability.
- Both retrievers returned topically related but non-answering hotel sections
  for the unsupported restaurant question. The grounded answer model still
  returned the exact fallback.
- The Oracle arm unexpectedly returned the fallback for `booking-002` despite
  receiving the labeled policy evidence and structured fields. Review the
  Oracle prompt/context contract and repeat before using Oracle as an upper
  bound.
- The LLM-only control answered the generic chargeback question correctly but
  did not establish the company-specific refund facts or booking state.

## Limitations

- This is a six-question smoke dataset with one trial per question, not a
  statistically meaningful final benchmark.
- Pricing variables were zero during the run. Token usage is recorded, but all
  estimated dollar costs are zero and must not be interpreted as actual cost.
- The deterministic lexical grader is auditable but is not a semantic judge.
- The `booking-002` expected facts can be answered from structured transaction
  context even when document retrieval misses the policy. Refine that case if
  it is intended to isolate document-retrieval quality.
