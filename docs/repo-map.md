# Repo Map

这个文件回答三个问题：应该改哪里、哪些边界不能破、每类改动最少要验证什么。

## 目录与职责

- `src/data_processing/`
  负责 PDF 文本解析、表格抽取、电路图 OCR，产出标准化 chunks。
- `src/retrieval/`
  负责 BM25、Embedding、RRF 融合、reranker。这里处理召回与排序，不处理 HTTP。
- `src/knowledge_graph/`
  负责实体抽取、关系构建、Neo4j 读写。
- `src/llm/`
  负责意图识别、query 改写、prompt 拼接、答案生成。
- `src/api/`
  负责启动组件、组装流水线、暴露 HTTP / SSE 接口。
- `src/frontend/`
  负责 Streamlit MVP，通过 API 调用系统。
- `scripts/`
  负责离线构建和数据准备，不承载在线请求逻辑。
- `tests/`
  轻量单测、结构约束测试、RAGAS 评估脚本。

## 允许依赖方向

- `src/api/` 可以依赖 `src/llm/`、`src/retrieval/`、`src/knowledge_graph/`
- `src/frontend/` 应通过 HTTP 调用 API，不直接依赖核心模块
- `scripts/` 可以依赖核心模块和共享配置加载
- 核心模块之间允许按业务链路依赖，但不能反向依赖 `src/api/` 或 `src/frontend/`

## 当前关键产物

- `config/config.yaml`
  主配置文件
- `config/config.local.yaml`
  本地覆盖配置，和主配置递归合并
- `data/processed/chunks.json`
  BM25 和离线处理链路的基础产物
- `data/processed/chunks_with_entities.json`
  图谱抽取调试产物

## 常见改动应该落在哪

- 调整故障码 / 中文查询命中效果
  优先改 `src/retrieval/bm25_retriever.py`、`src/retrieval/hybrid_retriever.py`
- 调整上下文拼接、引用格式、答案风格
  优先改 `src/llm/answer_generator.py`
- 调整意图识别和 query 改写
  优先改 `src/llm/query_rewriter.py`
- 调整 API 协议、SSE 输出、会话行为
  优先改 `src/api/routes/`
- 调整 PDF → chunks 产物
  优先改 `src/data_processing/` 和 `scripts/build_index.py`
- 调整图谱抽取或 Neo4j 查询
  优先改 `src/knowledge_graph/` 和 `scripts/build_graph.py`

## 低风险改动模式

- 先改单一层，再补该层测试
- 把新入口写成脚本或 Make 目标，而不是只留在 README 里
- 给离线产物定义稳定文件路径，避免 agent 到处猜

## 高风险改动模式

- 在路由里直接写检索、图谱、prompt 细节
- 修改 chunks 结构但不补测试或迁移说明
- 让前端直接 import 核心模块
- 让脚本和 API 使用不同的配置合并规则
