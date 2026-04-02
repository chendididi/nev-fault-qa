PYTHON ?= python
PYTEST ?= pytest
UVICORN ?= uvicorn
HOST ?= 0.0.0.0
PORT ?= 8000
QA_FILE ?=
PIPELINE ?= build_index
RUN_ID ?=

.PHONY: help compile check test-unit test-retrieval test-generation test-chat-routes test-health test-architecture test-build-index test-run-manifest test-chunk-contract test-official-docs-manifest test-html-manual test-fetch-manual test-download-nhtsa test-git-health test-eval-dataset test-handoff test-kg test-auto-pipeline test-config-loader test-validate-runtime test-eval-retrieval test-eval-generation-baseline infra-up infra-down build-index-sample load-sample-embeddings sample-setup build-graph download-official-docs download-nhtsa-data fetch-tesla-service-manual auto-pipeline auto-pipeline-cpu git-health run-api run-frontend eval-ragas eval-ragas-fixed eval-retrieval eval-retrieval-baseline eval-generation-baseline validate-runtime validate-runtime-spawn validate-runtime-smoke health health-ready rollback-artifact rollback-milvus rollback-graph handoff codex-tmux codex-resume

help:
	@printf "Available targets:\n"
	@printf "  make check                  # compile + lightweight tests\n"
	@printf "  make sample-setup           # build sample BM25 data + load sample embeddings\n"
	@printf "  make infra-up               # start Milvus + Neo4j\n"
	@printf "  make run-api                # run FastAPI app\n"
	@printf "  make run-frontend           # run Streamlit MVP\n"
	@printf "  make eval-ragas             # run RAGAS against a running API\n"
	@printf "  make eval-ragas-fixed       # run RAGAS against fixed eval set\n"
	@printf "  make download-official-docs # download official Tesla/BYD docs\n"
	@printf "  make download-nhtsa-data    # download NHTSA manufacturer communications + chunks\n"
	@printf "  make fetch-tesla-service-manual # crawl Tesla HTML service manual pages\n"
	@printf "  make auto-pipeline          # run end-to-end offline automation pipeline\n"
	@printf "  make auto-pipeline-cpu      # run heavy pipeline using config/config.cpu.yaml\n"
	@printf "  make git-health             # check .git writability + branch/remote/dirty status\n"
	@printf "  make validate-runtime       # runtime checks against already running API\n"
	@printf "  make validate-runtime-smoke # spawn API (CPU config), check /health + /ready\n"
	@printf "  make validate-runtime-spawn # spawn API (CPU config), full contract including /chat\n"
	@printf "  make eval-retrieval         # fixed retrieval baseline (Recall@K / MRR)\n"
	@printf "  make eval-retrieval-baseline # qrels-based retrieval baseline with thresholds\n"
	@printf "  make eval-generation-baseline # fixed RAGAS baseline with thresholds\n"
	@printf "  make health-ready           # check readiness endpoint\n"
	@printf "  make rollback-artifact      # rollback build_index/build_graph artifact\n"
	@printf "  make rollback-milvus        # rebuild Milvus from previous build_index chunks\n"
	@printf "  make rollback-graph         # rebuild Neo4j from previous build_graph entities\n"
	@printf "  make handoff                # write versioned handoff snapshot + latest pointers\n"
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

test-fetch-manual:
	$(PYTEST) tests/test_fetch_tesla_service_manual.py -q

test-download-nhtsa:
	$(PYTEST) tests/test_download_nhtsa_data.py -q

test-git-health:
	$(PYTEST) tests/test_git_health.py -q

test-eval-dataset:
	$(PYTEST) tests/test_eval_dataset.py -q

test-handoff:
	$(PYTEST) tests/test_handoff_snapshot.py -q

test-kg:
	$(PYTEST) tests/test_entity_extractor.py tests/test_relation_builder.py -q

test-auto-pipeline:
	$(PYTEST) tests/test_auto_pipeline.py -q

test-config-loader:
	$(PYTEST) tests/test_config_loader.py -q

test-validate-runtime:
	$(PYTEST) tests/test_validate_runtime.py -q

test-eval-retrieval:
	$(PYTEST) tests/test_eval_retrieval_baseline.py -q

test-eval-generation-baseline:
	$(PYTEST) tests/test_eval_generation_baseline.py -q

test-unit: test-retrieval test-generation test-chat-routes test-health test-architecture test-build-index test-run-manifest test-chunk-contract test-official-docs-manifest test-html-manual test-fetch-manual test-download-nhtsa test-git-health test-eval-dataset test-handoff test-kg test-auto-pipeline test-config-loader test-validate-runtime test-eval-retrieval test-eval-generation-baseline

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

download-nhtsa-data:
	$(PYTHON) scripts/download_nhtsa_data.py

fetch-tesla-service-manual:
	$(PYTHON) scripts/fetch_tesla_service_manual.py

auto-pipeline:
	$(PYTHON) scripts/auto_pipeline.py

auto-pipeline-cpu:
	$(PYTHON) scripts/auto_pipeline.py --config config/config.cpu.yaml --with-nhtsa --nhtsa-make TESLA --nhtsa-make BYD --with-embedding --with-graph --entities-file data/processed/chunks_with_entities.json --require-services --skip-check

git-health:
	$(PYTHON) scripts/git_health.py

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

eval-ragas-fixed:
	$(PYTHON) tests/eval_ragas.py --qa-file tests/eval_qa_set.json

eval-retrieval:
	$(PYTHON) scripts/eval_retrieval.py --qa-file tests/eval_qa_set.json

eval-retrieval-baseline:
	$(PYTHON) scripts/eval_retrieval.py --qa-file tests/eval_retrieval_set.json --qrels-file tests/eval_retrieval_qrels.json --mode auto --config config/config.cpu.yaml --thresholds-file tests/eval_retrieval_thresholds.json --fail-on-threshold

eval-generation-baseline:
	$(PYTHON) scripts/eval_generation_baseline.py --spawn-api --config config/config.cpu.yaml --qa-file tests/eval_generation_set.json --api-timeout 240 --ragas-timeout 900 --ragas-max-workers 2 --llm-model gpt-5.4 --llm-base-url https://cmdme.cn --thresholds-file tests/eval_ragas_thresholds.json --fail-on-threshold

validate-runtime:
	$(PYTHON) scripts/validate_runtime.py

validate-runtime-smoke:
	$(PYTHON) scripts/validate_runtime.py --spawn-api --config config/config.cpu.yaml --skip-chat --startup-timeout 300

validate-runtime-spawn:
	$(PYTHON) scripts/validate_runtime.py --spawn-api --config config/config.cpu.yaml --startup-timeout 300

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

rollback-milvus:
ifeq ($(strip $(RUN_ID)),)
	$(PYTHON) scripts/rollback_milvus.py --previous
else
	$(PYTHON) scripts/rollback_milvus.py --run-id $(RUN_ID)
endif

rollback-graph:
ifeq ($(strip $(RUN_ID)),)
	$(PYTHON) scripts/rollback_graph.py --previous
else
	$(PYTHON) scripts/rollback_graph.py --run-id $(RUN_ID)
endif

handoff:
	$(PYTHON) scripts/handoff_snapshot.py

codex-tmux:
	./scripts/start_codex_tmux.sh

codex-resume:
	./scripts/start_codex_tmux.sh resume
