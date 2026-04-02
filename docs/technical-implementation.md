# 技术实现与链路说明（大白话版）

更新时间：2026-03-29

这份文档用尽量通俗的方式说明：系统怎么跑、关键链路走哪几个文件、以及仓库里每个文件是干什么的。

## 一句话理解这个项目

把维修手册做成“能问的知识库”：先把手册切成小段并做索引，用户提问时先检索，再用大模型总结成答案，最后把引用来源标出来。

## 在线链路（用户提问到答案输出）

链式路径（按实际执行顺序）：

1. `src/api/main.py`  
   这里是后端入口。启动时加载配置、初始化模型、检索器和数据库连接。
2. `src/api/routes/chat.py`  
   处理 `/api/v1/chat` 和 `/api/v1/chat/stream`。把用户问题丢进 RAG 流水线，再把答案流式吐回去。
3. `src/llm/query_rewriter.py`  
   先判断是故障码、故障现象还是操作步骤；如果是现象描述，就改写成更标准的检索用语。
4. `src/retrieval/bm25_retriever.py`  
   用关键词做“精确搜”。对故障码特别友好。
5. `src/retrieval/embedding_retriever.py`  
   用向量做“语义搜”。对口语化描述更友好。
6. `src/retrieval/hybrid_retriever.py`  
   把 BM25 和向量检索结果用 RRF 算法融合。
7. `src/retrieval/reranker.py`  
   用重排序模型把候选片段再精排，保留最相关的 Top-K。
8. `src/knowledge_graph/neo4j_client.py`  
   如果问题里有故障码或症状，就去图谱里查零部件/子系统/症状补充信息。
9. `src/llm/answer_generator.py`  
   把“检索到的片段 + 图谱信息”组装成提示词。
10. `src/llm/qwen_client.py`  
    调 Qwen2‑VL 生成答案，并支持流式输出。
11. `src/frontend/streamlit_app.py`  
    前端通过 SSE 接收流式答案，并把引用来源展示给用户。

## 离线链路（把资料变成可检索的知识库）

### 1. 构建索引（PDF/HTML/JSON -> chunks -> 向量库）

链式路径：

1. `scripts/build_index.py`  
   这是总入口，负责发现输入、合并 chunks、校验、去重并落盘。
2. `src/data_processing/pdf_parser.py`  
   把 PDF 按 chunk_size 切成小段。
3. `src/data_processing/html_manual_parser.py`  
   把 Tesla HTML 手册页面提取文本并切成小段。
4. `src/data_processing/table_extractor.py`  
   从表格里抽故障码、描述和原因，补进索引。
5. `src/data_processing/circuit_ocr.py`  
   针对电路图页面做 OCR 和文字描述（可选步骤）。
6. `src/data_processing/chunk_contract.py`  
   校验每个 chunk 必须有的字段和长度，避免脏数据。
7. `data/processed/chunks.json`  
   这是 BM25 和后续流程的“事实来源文件”。
8. `src/retrieval/embedding_retriever.py`  
   把 chunks 写入 Milvus，完成向量索引。

### 2. 构建图谱（chunks -> 实体抽取 -> Neo4j）

链式路径：

1. `scripts/build_graph.py`  
   读取 chunks，执行实体抽取并写入 Neo4j。
2. `src/knowledge_graph/entity_extractor.py`  
   调 Qwen2‑VL 抽取 FaultCode/Component/Symptom 等实体。
3. `src/knowledge_graph/relation_builder.py`  
   把实体变成关系（INDICATES、HAS_SYMPTOM 等）。
4. `src/knowledge_graph/neo4j_client.py`  
   负责写入 Neo4j 和查询。

## 评估链路（质量指标）

链式路径：

1. `tests/eval_ragas.py`  
   调用 API 拿答案和上下文，计算 RAGAS 指标。
2. `tests/eval_qa_set.json`  
   固定评测集。
3. `tests/ragas_results.json`  
   评估结果输出文件。

## 运维与可追溯

链式路径：

1. `src/ops/run_manifest.py`  
   每次离线脚本运行都会记录 run manifest。
2. `src/ops/artifact_store.py`  
   管理“当前激活版本”的产物，支持回滚。
3. `src/ops/logging_setup.py`  
   统一日志格式，自动带上 trace 字段。
4. `scripts/rollback_*.py`  
   回滚 chunks、Milvus 与 Neo4j。
5. `scripts/handoff_snapshot.py`  
   汇总运行状态和评估结果，生成交接快照。

## 全仓库文件说明（逐文件，大白话）

### 顶层文件

| 文件 | 大白话说明 |
| --- | --- |
| `AGENTS.md` | 给自动化工具看的“仓库规则说明书”。 |
| `CLAUDE.md` | 兼容 Claude 的说明和约束。 |
| `Makefile` | 常用命令入口的快捷方式。 |
| `README.md` | 项目说明书和启动方式。 |
| `docker-compose.yml` | 一键起 Milvus 和 Neo4j。 |
| `requirements.txt` | Python 依赖列表。 |
| `启动清单.md` | 运行项目的一步一步操作清单。 |

### 配置

| 文件 | 大白话说明 |
| --- | --- |
| `config/config.yaml` | 所有核心配置的“总开关”。 |

### docs 文档

| 文件 | 大白话说明 |
| --- | --- |
| `docs/README.md` | 文档索引，从这里开始读。 |
| `docs/architecture.md` | 系统架构图与流程说明。 |
| `docs/codex-standards.md` | 项目采用的规范与约束。 |
| `docs/harness.md` | 怎么验证、怎么排查问题。 |
| `docs/official-sources.md` | 官方数据来源边界说明。 |
| `docs/repo-map.md` | 每个模块负责啥，改哪里。 |
| `docs/project-progress.md` | 当前项目进度总结。 |
| `docs/technical-implementation.md` | 本文件，技术细节说明。 |

### data 数据

| 文件 | 大白话说明 |
| --- | --- |
| `data/raw/official/manifest.json` | 官方资料清单，决定能下载哪些文档。 |
| `data/sample/sample_chunks.json` | 示例数据，用来跑通流程。 |

### scripts 脚本

| 文件 | 大白话说明 |
| --- | --- |
| `scripts/build_index.py` | 解析资料、生成 chunks、写入 Milvus。 |
| `scripts/build_graph.py` | 实体抽取并构建 Neo4j 图谱。 |
| `scripts/download_models.py` | 一键下载 BGE 与 Qwen 模型。 |
| `scripts/download_official_docs.py` | 下载官方 PDF 和入口页。 |
| `scripts/fetch_tesla_service_manual.py` | 抓 Tesla HTML 手册页面。 |
| `scripts/handoff_snapshot.py` | 生成交接快照和总结。 |
| `scripts/load_embeddings.py` | 已有 chunks 时补建向量索引。 |
| `scripts/rollback_artifact.py` | 回滚 chunks 类文件产物。 |
| `scripts/rollback_graph.py` | 回滚 Neo4j 图谱。 |
| `scripts/rollback_milvus.py` | 回滚 Milvus 向量索引。 |
| `scripts/start_codex_tmux.sh` | tmux 里启动会话的脚本。 |

### src 代码

| 文件 | 大白话说明 |
| --- | --- |
| `src/__init__.py` | 标记这是一个 Python 包。 |
| `src/config_loader.py` | 读取配置并合并本地覆盖。 |
| `src/api/__init__.py` | 标记 API 包。 |
| `src/api/main.py` | FastAPI 入口与依赖初始化。 |
| `src/api/health.py` | `/health` 与 `/ready` 检查。 |
| `src/api/routes/__init__.py` | 路由包标记。 |
| `src/api/routes/chat.py` | 聊天接口与 SSE 流式输出。 |
| `src/api/routes/diagnosis.py` | 故障码/症状诊断专用接口。 |
| `src/data_processing/__init__.py` | 数据处理包标记。 |
| `src/data_processing/pdf_parser.py` | PDF 分块解析。 |
| `src/data_processing/html_manual_parser.py` | Tesla HTML 手册解析。 |
| `src/data_processing/table_extractor.py` | 故障码表格抽取。 |
| `src/data_processing/circuit_ocr.py` | 电路图 OCR 和描述生成。 |
| `src/data_processing/chunk_contract.py` | chunks 数据格式校验。 |
| `src/frontend/streamlit_app.py` | Streamlit 前端页面。 |
| `src/knowledge_graph/__init__.py` | 图谱包标记。 |
| `src/knowledge_graph/neo4j_client.py` | Neo4j 读写封装。 |
| `src/knowledge_graph/entity_extractor.py` | Qwen 抽取实体。 |
| `src/knowledge_graph/relation_builder.py` | 实体转关系并写图谱。 |
| `src/llm/__init__.py` | LLM 包标记。 |
| `src/llm/query_rewriter.py` | 意图识别与 query 改写。 |
| `src/llm/answer_generator.py` | 拼上下文并生成答案。 |
| `src/llm/qwen_client.py` | Qwen2‑VL 推理封装。 |
| `src/ops/__init__.py` | 运维包标记。 |
| `src/ops/logging_setup.py` | 统一日志配置。 |
| `src/ops/run_manifest.py` | 记录每次离线运行信息。 |
| `src/ops/artifact_store.py` | 管理产物版本与回滚。 |
| `src/retrieval/__init__.py` | 检索包标记。 |
| `src/retrieval/bm25_retriever.py` | 关键词检索。 |
| `src/retrieval/embedding_retriever.py` | 向量检索。 |
| `src/retrieval/hybrid_retriever.py` | BM25 + 向量融合。 |
| `src/retrieval/reranker.py` | 精排重排序。 |

### tests 测试

| 文件 | 大白话说明 |
| --- | --- |
| `tests/__init__.py` | 测试包标记。 |
| `tests/eval_qa_set.json` | 固定评测集题目。 |
| `tests/eval_ragas.py` | RAGAS 评估脚本。 |
| `tests/ragas_results.json` | RAGAS 评估输出结果。 |
| `tests/test_architecture.py` | 防止越层依赖的结构测试。 |
| `tests/test_build_index.py` | build_index 的单测。 |
| `tests/test_chat_routes.py` | 聊天 API 协议测试。 |
| `tests/test_chunk_contract.py` | chunks 格式校验测试。 |
| `tests/test_entity_extractor.py` | 实体抽取测试。 |
| `tests/test_eval_dataset.py` | 评测集格式测试。 |
| `tests/test_generation.py` | 答案生成逻辑测试。 |
| `tests/test_handoff_snapshot.py` | handoff 快照测试。 |
| `tests/test_health_routes.py` | health/ready 接口测试。 |
| `tests/test_html_manual_parser.py` | HTML 手册解析测试。 |
| `tests/test_official_docs_manifest.py` | 官方文档清单结构测试。 |
| `tests/test_relation_builder.py` | 图谱关系构建测试。 |
| `tests/test_retrieval.py` | 检索逻辑测试。 |
| `tests/test_run_manifest.py` | run manifest 结构测试。 |
