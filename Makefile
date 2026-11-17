.DEFAULT_GOAL := help
RUN := uv run --env-file .env
.PHONY: help up down reset db-check data seed test lint eval eval-test demo run process

.env:
	cp .env.example .env

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

up: .env  ## start Postgres (localhost:5434) and wait until it's healthy
	docker compose up -d --wait db

down:  ## stop containers (data kept)
	docker compose down

reset:  ## DELETE the local database volume, then start fresh (dev data only)
	docker compose down -v
	$(MAKE) up

db-check:  ## confirm pg_search, vector and the app tables exist
	@docker compose exec -T db psql -U app -d app -tAc \
	  "SELECT count(*) FROM pg_extension WHERE extname IN ('pg_search','vector')" | grep -qx 2 \
	  && docker compose exec -T db psql -U app -d app -tAc \
	  "SELECT count(to_regclass(t)) FROM unnest(array['llm_calls','documents','audit_log']) t" \
	  | grep -qx 3 \
	  && echo "db OK: pg_search, vector, llm_calls, documents, audit_log" \
	  || (echo "db check FAILED"; exit 1)

data:  ## generate the 200 invoices, ground truth and ERP data (same seed, same files)
	uv run python -m data.generate

seed: .env  ## load products and purchase orders into Postgres (safe to repeat)
	$(RUN) python -m app.cli seed

test: .env  ## unit + database tests (database tests need `make up`)
	$(RUN) pytest -q

lint:  ## ruff
	uv run ruff check .

eval: .env seed  ## dev split, baseline vs pipeline -> evals/results-dev.md (calls the API)
	$(RUN) python -m evals.run --split dev

eval-test: .env seed  ## test split -> evals/results-test.md; only once you've stopped tuning
	$(RUN) python -m evals.run --split test

demo: .env seed  ## offline walkthrough in ORACLE mode (no API key, no model)
	$(RUN) python -m app.demo

run: .env  ## API + review page on http://localhost:8002/review (EXTRACTOR=oracle for offline)
	$(RUN) uvicorn app.main:app --reload --port 8002

process: .env  ## process a PDF or folder: make process FILE=path
	$(RUN) python -m app.cli process $(FILE)
