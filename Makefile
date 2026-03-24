PYTHON ?= python
PYTEST ?= pytest
UVICORN ?= uvicorn
HOST ?= 0.0.0.0
PORT ?= 8000
QA_FILE ?=

.PHONY: help compile check test-unit test-retrieval test-generation test-chat-routes test-architecture test-build-index infra-up infra-down build-index-sample load-sample-embeddings sample-setup build-graph run-api run-frontend eval-ragas health codex-tmux codex-resume

help:
	@printf "Available targets:\n"
	@printf "  make check                  # compile + lightweight tests\n"
	@printf "  make sample-setup           # build sample BM25 data + load sample embeddings\n"
	@printf "  make infra-up               # start Milvus + Neo4j\n"
	@printf "  make run-api                # run FastAPI app\n"
	@printf "  make run-frontend           # run Streamlit MVP\n"
	@printf "  make eval-ragas             # run RAGAS against a running API\n"
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

test-architecture:
	$(PYTEST) tests/test_architecture.py -q

test-build-index:
	$(PYTEST) tests/test_build_index.py -q

test-unit: test-retrieval test-generation test-chat-routes test-architecture test-build-index

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

codex-tmux:
	./scripts/start_codex_tmux.sh

codex-resume:
	./scripts/start_codex_tmux.sh resume
