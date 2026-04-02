# 项目完整进度（持续交付版）

更新时间：2026-04-02

## 当前完成情况（已落地）

- 端到端链路实现完成：Query 预处理 → 混合检索 → 重排 → 图谱补充 → Qwen2‑VL 生成 → FastAPI SSE → Streamlit 前端。
- 官方数据自动化完成：Tesla/BYD 官方资料下载、Tesla HTML 手册抓取、NHTSA Manufacturer Communications 下载与 chunks 化。
- 离线自动化入口完成：`auto_pipeline.py` 支持 `preflight / hook / dry-run / with-nhtsa / with-embedding / with-graph`。
- 无 CUDA 场景验证完成：新增 `config/config.cpu.yaml`，已在 CPU 配置下跑通 Milvus 向量入库 + Neo4j 图谱重建（复用 entities）。
- 运维与可追溯完善：Run manifest、版本化产物、回滚脚本、handoff snapshot、`git-health` 状态检查入口。
- 测试覆盖持续扩展：新增 auto pipeline、Tesla fetch、NHTSA 下载、Git 状态检查等测试并纳入 `make check`。
- 运行时验收自动化完成：新增 `scripts/validate_runtime.py`，支持 `--spawn-api` 自动拉起并校验 `/health`、`/ready`、`/api/v1/chat` 契约。
- 固定检索基线完成：新增 `scripts/eval_retrieval.py`，支持 qrels、阈值门禁与版本化产物；已落地 `tests/eval_retrieval_set.json` / `eval_retrieval_qrels.json` / `eval_retrieval_thresholds.json`。
- 固定生成基线完成：新增 `scripts/eval_generation_baseline.py`，可自动拉起 API 执行固定集 RAGAS 并按阈值门禁；已落地 `tests/eval_generation_set.json` / `eval_ragas_thresholds.json`。

## 最近验证证据（2026-04-02）

- `make check` 全通过（含新增测试）。
- `auto_pipeline` 重依赖流程成功：
  - `auto_pipeline-20260402T094932Z-d316be60`
  - 阶段：`download_docs -> download_nhtsa -> fetch_manual_html -> build_chunks -> build_vector -> build_graph -> handoff`
- `load_embeddings` 成功导入 6384 条 chunks，Milvus 就绪。
- `build_graph` 成功写入 Neo4j（复用 `data/processed/chunks_with_entities.json`）。
- 运行时契约验收成功（CPU 配置，自动拉起 API）：
  - `validate_runtime-20260402T103228Z-5b783251`
  - 结果：`/health`、`/ready`、`/api/v1/chat` 全通过（chat 响应耗时约 125s）。
- 检索基线评测（qrels + 阈值）已通过：
  - `eval_retrieval-20260402T105241Z-aa5c3843`
  - Hybrid 指标：MRR `0.4426`，HitRate `0.75`，Recall@1/3/5=`0.375/0.375/0.625`（阈值通过）。
- 生成基线评测（固定集 RAGAS + 阈值）已通过：
  - `eval_generation_baseline-20260402T133942Z-6370b5f8`
  - 指标：Faithfulness `0.4462`，Answer Relevancy `0.8581`，Context Precision `1.0000`，Context Recall `0.7500`（阈值通过）。
  - 评估模型：`gpt-5.4`（`https://cmdme.cn`）。

## 尚未完成 / 风险点

- 图谱质量评估仍缺：当前能跑通构建与查询，但实体抽取准确率、关系覆盖率缺系统评估。
- API 在线压测未完成：P95 延迟与高并发稳定性尚未固化报告。
- 环境风险：当前工作机 `.git` 所在文件系统处于只读挂载（`errors=remount-ro`），影响本地 git 引用写入。
- Tesla `owners_manual` 下载仍存在 403（非阻塞，但属于数据完整性风险）。
- 评估外部依赖风险：RAGAS 在直连 OpenAI 可能受地域限制，当前已固定使用 `gpt-5.4 + https://cmdme.cn` 作为可执行默认路径。

## 下一步计划（优先级）

1. 端到端在线验证（必须）
   启动 API + 前端，补 `/ready` 运行态证据与请求级 trace 汇总。
2. 图谱质量评估（高优先）
   对实体与关系做抽样标注，补准确率/覆盖率报告。
3. 数据源扩展（高优先）
   增加更多官方维修语料入口（按品牌分批），保持来源可追溯。
4. 环境稳定性（高优先）
   修复宿主机只读挂载问题，恢复常规本地 git 提交/同步能力。

## 里程碑结论（当前阶段）

项目已从“示例链路可跑”推进到“真实官方语料 + 自动化重依赖链路可跑”，当前核心工作转入“质量基线与在线稳定性验收”阶段。
