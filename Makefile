.PHONY: install test test-ui serve index eval validate-dataset

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
	python3 -m kbqa.evals.runner --live --dataset evals/cases.jsonl \
	  --manifest evals/manifest.json --output results/eval.jsonl

validate-dataset:
	python3 -m kbqa.evals.dataset --manifest evals/manifest.json \
	  --docs docs --transactions fixtures/bookings.json
