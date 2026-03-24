# Harness

这个仓库的 harness 目标不是“什么都自动化”，而是给 agent 和维护者一条稳定、可分层、可证明的验证路径。

## 当前可用 harness

- `make check`
  编译检查 + 轻量单测 + 结构边界测试，不依赖 GPU、Milvus、Neo4j。
- `make health-ready`
  检查运行中 API 的 `/ready`，确认 BM25、Milvus、Neo4j、Qwen 和 chunks 文件是否齐备。
- `python scripts/build_index.py --input data/sample/ --skip-embedding`
  用内置示例数据生成 `data/processed/chunks.json`，并把本次运行落盘到版本化产物目录。
- `python scripts/build_index.py --input data/raw/official --skip-embedding`
  递归摄取官方资料目录中的 PDF、Tesla HTML 和预构建 chunks JSON，并在 run manifest 中记录发现统计与跳过原因。
- `python scripts/load_embeddings.py --input data/sample/sample_chunks.json`
  把现成 chunks JSON 导入 Milvus，并记录运行 manifest。
- `python scripts/build_graph.py --chunks-file data/processed/chunks.json`
  验证实体抽取和 Neo4j 写入链路，并保存版本化 `chunks_with_entities.json`。
- `python scripts/download_official_docs.py`
  下载一批官方 Tesla / BYD 文档到 `data/raw/official/`，并记录下载运行 manifest。
- `python scripts/rollback_artifact.py --pipeline build_index --previous`
  回滚 `build_index` 或 `build_graph` 的文件产物到上一个激活版本。
- `python tests/eval_ragas.py`
  对运行中的 API 做质量评估，默认使用内置评估集。

## 建议验证层级

### Level 0: 纯本地、最快反馈

先跑：

```bash
make check
```

这一步证明：

- Python 文件可编译
- 检索 / 生成相关轻量逻辑仍可用
- 模块边界没有被破坏
- `/health` 和 `/ready` 的最小契约可用
- run manifest / 版本化产物逻辑可用

### Level 1: 离线数据链路

当你改了 `src/data_processing/`、`scripts/build_index.py` 或 chunks 契约时，至少跑：

```bash
python scripts/build_index.py --input data/sample/ --skip-embedding
```

这一步证明：

- 仓库内置样例输入可以生成非空 `data/processed/chunks.json`
- 不会再出现“文档说能跑，脚本实际写出空产物”的假快乐路径

### Level 2: 检索基础设施

当你改了向量检索或 Milvus 相关逻辑时，先启动依赖：

```bash
docker compose up -d
python scripts/load_embeddings.py --input data/sample/sample_chunks.json
```

这一步证明：

- Milvus 可连接
- 向量模型可加载
- chunks JSON 可以被写入向量索引

### Level 3: 端到端

当你改了 API 编排、prompt、图谱调用、SSE 或前端时，再跑：

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
streamlit run src/frontend/streamlit_app.py
python tests/eval_ragas.py
```

这一步证明：

- 运行时组件能一起启动
- API 能返回答案和来源
- 质量指标至少能跑通一轮评估

## 常用命令入口

仓库现在提供了一个统一入口：

```bash
make help
```

推荐优先使用 Make 目标，而不是把长命令散落在聊天记录或 README 片段里。

## 运行证据与回滚

- 离线脚本现在会把每次运行写到 `data/artifacts/<pipeline>/<run_id>/`
- 每次运行至少包含：
  - `manifest.json`
  - `config.snapshot.json`
  - `run.log`
- `build_index` 和 `build_graph` 会维护 `data/artifacts/<pipeline>/current.json`
- 当前稳定路径仍保留：
  - `data/processed/chunks.json`
  - `data/processed/chunks_with_entities.json`
- 这些稳定路径不再是“唯一产物”，而是当前激活版本的同步副本
- 第一阶段回滚只覆盖文件产物，不覆盖 Milvus / Neo4j 的深度回滚

## 官方数据入口

- 官方文档清单固定在 `data/raw/official/manifest.json`
- `download_official_docs.py` 默认下载 PDF
- Tesla 官方 `service.tesla.com` 维修手册主体是 HTML 站点
- `fetch_tesla_service_manual.py` 抓下来的 HTML 页面现在可以直接被 `build_index.py` 递归发现并切成 chunks
- `build_index.py` 会把发现到的 PDF / HTML / chunks JSON 数量，以及被跳过的非 chunks JSON 示例写进 run manifest 和 `run.log`

## 失败排查

- `make check` 失败
  先看是编译错误、单测失败还是结构边界测试失败，再缩小到对应模块。
- `build_index.py` 产物为空
  检查输入目录里是否有 PDF，或是否存在 `sample_chunks.json`。
- API 启动失败
  先确认模型路径、Milvus、Neo4j 和 `config.local.yaml` 覆盖是否一致。
- 图谱查询为空
  先确认 `build_graph.py` 是否已执行，以及 Neo4j 是否有节点。
- RAGAS 失败
  先确认 API 已启动，再确认答案字段和 sources 字段格式没有破坏脚本契约。

## 仍然缺的 harness

这是下一批优先级最高的改造项：

1. 固定评测集入库
   目标：把 `tests/eval_qa_set.json` 作为仓库工件维护，避免每次临时拼样本。
2. 端到端 trace 深化
   目标：把 query、召回候选数、rerank 输入输出、图谱命中情况写成更完整的可追踪事件。
3. CI 分层
   目标：PR 默认跑 `make check`，夜间任务再跑重依赖评估。
4. 数据契约测试
   目标：显式校验 chunks 必含字段、页码类型、source/chapter 长度约束。
5. 外部服务回滚
   目标：为 Milvus / Neo4j 引入版本化导入和可执行恢复流程，而不只回滚本地文件产物。
