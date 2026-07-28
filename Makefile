.PHONY: install test test-ui serve index eval

install:
	python3 -m pip install -e '.[dev]'

test:
	python3 -m pytest

test-ui:
	node --test tests/ui/*.test.mjs

serve:
	python3 -m uvicorn kbqa.api:app --reload

index:
	curl -X POST http://127.0.0.1:8000/index

eval:
	python3 -m kbqa.evals.runner --dataset evals/cases.jsonl --output results/eval.jsonl
