# Controlled RAG Evaluation Report

- Run ID: `61f28d3f-7134-422d-bbde-718344a265d6`
- Dataset: `seed-v1.1`
- Corpus tier: `primary_controlled`
- Trials per question: 1 (single_run_point_estimate)
- Minimum questions per category: 1

> Experimental limitation: RAG-minus-control compares a strict grounded prompt plus retrieved context against a closed-book prompt; it is not a context-only ablation.

## Arm-by-category results

| Arm / category | Success | Correctness | Recall@3 | Citation accuracy | Fallback | Retrieval ms | Generation ms | Answer cost | Embedding cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only:company_specific | 2/2 | 0.0000 | N/A | N/A | 1.0000 | N/A | 3904.1012 | 0.0000 | N/A |
| llm_only:generic_ecommerce | 1/1 | 1.0000 | N/A | N/A | 1.0000 | N/A | 3820.3352 | 0.0000 | N/A |
| llm_only:user_specific | 2/2 | 0.0000 | N/A | N/A | 1.0000 | N/A | 2388.0496 | 0.0000 | N/A |
| llm_only:unsupported | 1/1 | N/A | N/A | N/A | 0.0000 | N/A | 1930.9951 | 0.0000 | N/A |
| bm25:company_specific | 2/2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2418 | 1460.7249 | 0.0000 | N/A |
| bm25:generic_ecommerce | 1/1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2968 | 1298.4443 | 0.0000 | N/A |
| bm25:user_specific | 2/2 | 1.0000 | 0.0000 | 0.7500 | 1.0000 | 0.6312 | 3020.2803 | 0.0000 | N/A |
| bm25:unsupported | 1/1 | N/A | N/A | 1.0000 | 1.0000 | 0.4188 | 1203.0348 | 0.0000 | N/A |
| vector:company_specific | 2/2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 475.6329 | 1928.2400 | 0.0000 | 0.0000 |
| vector:generic_ecommerce | 1/1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 828.5511 | 1578.8592 | 0.0000 | 0.0000 |
| vector:user_specific | 2/2 | 1.0000 | 1.0000 | 0.7500 | 1.0000 | 113.6655 | 1540.3088 | 0.0000 | 0.0000 |
| vector:unsupported | 1/1 | N/A | N/A | 1.0000 | 1.0000 | 243.7237 | 1538.0230 | 0.0000 | 0.0000 |
| oracle:company_specific | 2/2 | 1.0000 | N/A | 1.0000 | 1.0000 | N/A | 1621.2469 | 0.0000 | N/A |
| oracle:generic_ecommerce | 1/1 | 1.0000 | N/A | 1.0000 | 1.0000 | N/A | 2129.0398 | 0.0000 | N/A |
| oracle:user_specific | 2/2 | 0.5000 | N/A | 0.5000 | 0.5000 | N/A | 3508.4295 | 0.0000 | N/A |

## Index runs

| Backend | Index ms | Embedding tokens | Embedding cost (USD) | Error |
| --- | ---: | ---: | ---: | --- |
| bm25 | 2.1431 | N/A | N/A |  |
| vector | 2095.8777 | 453 | 0.0000 |  |

## Improvement over LLM-only

Improvement is intentionally omitted for `unsupported` and `user_specific` cases.

| Arm / category / metric | Difference |
| --- | ---: |
| bm25:company_specific:correctness | 1.0000 |
| bm25:company_specific:unsupported_claim_rate | -1.0000 |
| bm25:company_specific:fallback_correct | 0.0000 |
| vector:company_specific:correctness | 1.0000 |
| vector:company_specific:unsupported_claim_rate | -1.0000 |
| vector:company_specific:fallback_correct | 0.0000 |
| bm25:generic_ecommerce:correctness | 0.0000 |
| bm25:generic_ecommerce:unsupported_claim_rate | 0.0000 |
| bm25:generic_ecommerce:fallback_correct | 0.0000 |
| vector:generic_ecommerce:correctness | 0.0000 |
| vector:generic_ecommerce:unsupported_claim_rate | 0.0000 |
| vector:generic_ecommerce:fallback_correct | 0.0000 |

## Paraphrase robustness

- `bm25:refund-timeline`: hit rate 1.0000, consistency 1.0000, 2 members
- `vector:refund-timeline`: hit rate 1.0000, consistency 1.0000, 2 members

## Failure examples

- `bm25:booking-002` — oracle_annotation_model_prompt_review
- `vector:booking-002` — oracle_annotation_model_prompt_review
- `bm25:unsupported-001` — keyword_false_positive
- `vector:unsupported-001` — semantic_false_positive
