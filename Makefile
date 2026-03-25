PYTHON ?= python
PYTEST ?= pytest
UVICORN ?= uvicorn
HOST ?= 0.0.0.0
PORT ?= 8000
QA_FILE ?=
PIPELINE ?= build_index
RUN_ID ?=

.PHONY: help compile check test-unit test-retrieval test-generation test-chat-routes test-health test-architecture test-build-index test-run-manifest test-chunk-contract test-official-docs-manifest test-html-manual test-kg infra-up infra-down build-index-sample load-sample-embeddings sample-setup build-graph download-official-docs fetch-tesla-service-manual run-api run-frontend eval-ragas health health-ready rollback-artifact handoff codex-tmux codex-resume

help:
	@printf "Available targets:\n"
	@printf "  make check                  # compile + lightweight tests\n"
	@printf "  make sample-setup           # build sample BM25 data + load sample embeddings\n"
	@printf "  make infra-up               # start Milvus + Neo4j\n"
	@printf "  make run-api                # run FastAPI app\n"
	@printf "  make run-frontend           # run Streamlit MVP\n"
	@printf "  make eval-ragas             # run RAGAS against a running API\n"
	@printf "  make download-official-docs # download official Tesla/BYD docs\n"
	@printf "  make fetch-tesla-service-manual # crawl Tesla HTML service manual pages\n"
	@printf "  make health-ready           # check readiness endpoint\n"
	@printf "  make rollback-artifact      # rollback build_index/build_graph artifact\n"
	@printf "  make handoff                # write handoff snapshot to data/artifacts/handoff/latest.md\n"
	@printf "  make codex-tmux             # start or attach Codex inside tmux\n"
	@printf "  make codex-resume           # resume last Codex session inside tmux\n"

compile:
	$(PYTHON) -m compileall src scripts tests

test-retrieval:
	$(PYTEST) tests/test_retrieval.py -q

test-generation:
	$(PYTEST) tests/test_generation.py -q

test-chat-routes:
	$(PYTEST) tests/test_chat_routes.py -q

test-health:
	$(PYTEST) tests/test_health_routes.py -q

test-architecture:
	$(PYTEST) tests/test_architecture.py -q

test-build-index:
	$(PYTEST) tests/test_build_index.py -q

test-run-manifest:
	$(PYTEST) tests/test_run_manifest.py -q

test-chunk-contract:
	$(PYTEST) tests/test_chunk_contract.py -q

test-official-docs-manifest:
	$(PYTEST) tests/test_official_docs_manifest.py -q

test-html-manual:
	$(PYTEST) tests/test_html_manual_parser.py -q

test-kg:
	$(PYTEST) tests/test_entity_extractor.py tests/test_relation_builder.py -q

test-unit: test-retrieval test-generation test-chat-routes test-health test-architecture test-build-index test-run-manifest test-chunk-contract test-official-docs-manifest test-html-manual test-kg

check: compile test-unit

infra-up:
	docker compose up -d

infra-down:
	docker compose down

build-index-sample:
	$(PYTHON) scripts/build_index.py --input data/sample/ --skip-embedding

load-sample-embeddings:
	$(PYTHON) scripts/load_embeddings.py --input data/sample/sample_chunks.json

sample-setup: build-index-sample load-sample-embeddings

build-graph:
	$(PYTHON) scripts/build_graph.py --chunks-file data/processed/chunks.json

download-official-docs:
	$(PYTHON) scripts/download_official_docs.py

fetch-tesla-service-manual:
	$(PYTHON) scripts/fetch_tesla_service_manual.py

run-api:
	$(UVICORN) src.api.main:app --host $(HOST) --port $(PORT)

run-frontend:
	streamlit run src/frontend/streamlit_app.py

eval-ragas:
ifeq ($(strip $(QA_FILE)),)
	$(PYTHON) tests/eval_ragas.py
else
	$(PYTHON) tests/eval_ragas.py --qa-file $(QA_FILE)
endif

health:
	curl http://localhost:$(PORT)/health

health-ready:
	curl http://localhost:$(PORT)/ready

rollback-artifact:
ifeq ($(strip $(RUN_ID)),)
	$(PYTHON) scripts/rollback_artifact.py --pipeline $(PIPELINE) --previous
else
	$(PYTHON) scripts/rollback_artifact.py --pipeline $(PIPELINE) --run-id $(RUN_ID)
endif

handoff:
	$(PYTHON) scripts/handoff_snapshot.py

codex-tmux:
	./scripts/start_codex_tmux.sh

codex-resume:
	./scripts/start_codex_tmux.sh resume
