# AGENTS.md

仓库给 agent 的入口要尽量短。先看这里，再按任务跳转到更细的文档。

## 先读什么

1. `docs/README.md`：按任务找文档入口
2. `docs/codex-standards.md`：本仓库采用的 Codex 工作标准
3. `docs/repo-map.md`：模块边界、允许依赖方向、改动落点
4. `docs/harness.md`：验证层级、常用命令、失败排查

## 外部标准基线

- 本仓库默认遵循 OpenAI Codex 官方文章：
  - `Best practices`：<https://developers.openai.com/codex/learn/best-practices>
  - `Harness engineering: Leveraging Codex in an agent-first world`：<https://openai.com/index/harness-engineering/>
- 不依赖聊天记忆来维持这些规则；以 `docs/codex-standards.md` 的项目内落地版本为准。

## 这个仓库的真实执行入口

- API 入口：`uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- 前端入口：`streamlit run src/frontend/streamlit_app.py`
- 离线索引构建：`python scripts/build_index.py --input data/raw/`
- 从已有 JSON 补向量索引：`python scripts/load_embeddings.py --input data/processed/chunks.json`
- 图谱构建：`python scripts/build_graph.py --chunks-file data/processed/chunks.json`
- 轻量验证：`make check`

## 模块分工

- `src/data_processing/`：PDF 解析、表格抽取、OCR，负责生成 chunks
- `src/retrieval/`：BM25、Milvus 检索、RRF 融合、reranker
- `src/knowledge_graph/`：实体抽取、图谱写入、Neo4j 查询
- `src/llm/`：query 处理、prompt 组装、答案生成、Qwen 客户端
- `src/api/`：只做组装和对外接口，不承载底层检索/生成实现
- `src/frontend/`：HTTP 客户端，不直接依赖检索/图谱/LLM 模块
- `scripts/`：离线构建和数据准备

## 硬约束

- 配置以 `config/config.yaml` 为主，允许 `config/config.local.yaml` 覆盖；不要在代码里复制一套配置规则。
- `data/processed/chunks.json` 是 BM25 和离线数据链路的事实来源；修改 chunks 生成逻辑时，优先维护这个产物契约。
- `src/retrieval/`、`src/llm/`、`src/knowledge_graph/`、`src/data_processing/` 不得依赖 `src/api/` 或 `src/frontend/`。
- API 层负责 orchestration，不要把业务逻辑塞进路由函数。
- 前端层通过 HTTP 调 API，不直接 import 核心模块。
- 默认验证路径应尽量避免 GPU 和外部服务；需要重资源时，明确写出前置条件。

## 默认工作流

1. 先跑 `make check`
2. 只改一个子系统时，补跑对应测试
3. 改离线数据链路时，至少验证一次 `python scripts/build_index.py --input data/sample/ --skip-embedding`
4. 改端到端链路时，再启动 `docker compose up -d`、API、前端或评估脚本

## 交付时要报告什么

- 改了哪些行为
- 跑了哪些命令验证
- 没跑哪些重依赖验证，以及原因
- 剩余风险是否在数据、模型、外部服务还是文档
