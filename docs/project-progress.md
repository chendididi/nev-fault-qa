# 项目完整进度（中期答辩版）

更新时间：2026-03-29

## 当前完成情况

- 端到端链路已实现并可在示例数据上跑通：Query 预处理 → 混合检索 → 重排 → 图谱补充 → Qwen2‑VL 生成 → FastAPI SSE → Streamlit 前端。
- 数据工程管线已实现：PDF 分块、HTML 手册解析、故障码表格抽取、电路图 OCR（可选）、chunks 契约校验与去重、Milvus 向量索引构建。
- 知识图谱管线已实现：Qwen2‑VL 实体抽取（含缓存）、关系构建、Neo4j 写入与回滚。
- 运维与可追溯已实现：Run manifest、版本化产物、回滚脚本、handoff snapshot、结构边界测试。
- 评估脚本已实现：RAGAS 评估脚本与固定评测集、日志与结果落盘。

## 目前的验证证据

- 示例数据完整流程可执行：`build_index.py --input data/sample --skip-embedding` + `load_embeddings.py` + API + Streamlit。
- 轻量测试覆盖：检索、生成、图谱关系、数据契约、路由协议、结构边界与健康检查。
- `/health` 与 `/ready` readiness 检查已实现，覆盖 chunks、BM25、Milvus、Neo4j、Qwen 状态。

## 尚未完成 / 风险点

- 真实维修手册全量导入与质检尚未完成，当前以示例数据验证为主。
- 图谱实体抽取质量与关系覆盖率未完成系统性评估。
- 检索与生成指标（Recall@5、MRR、Faithfulness、Answer Relevancy）尚未跑基线。
- README 提到的 DashScope API 方案尚未接入代码，需要补实现或调整描述。
- 真实环境性能与稳定性（GPU 负载、Milvus/Neo4j 稳定性）待实测。

## 下一步计划（优先级排序）

1. 真实数据导入与抽样质检  
   将官方文档或自有维修手册导入 `data/raw/`，跑 `build_index.py` 与 `build_graph.py`，抽样核对 chunk、表格与 OCR 输出。
2. 质量评估与对比实验  
   跑 RAGAS 与检索指标（Recall@5/MRR），对比单路检索与混合检索、无图谱与有图谱的效果差异。
3. 端到端演示与性能测试  
   基于真实数据演示查询链路，记录 P95 延迟、模型加载与数据库稳定性。
4. 优化与补齐  
   结合评测结果优化 chunk 规则、召回与重排参数、图谱实体抽取提示词。
5. 无 GPU 方案补齐  
   需要时接入 DashScope API 并完成配置与回退逻辑。

## 里程碑结论（当前阶段）

项目主干功能已完成并具备可运行的工程化实现，当前进入真实数据验证与指标评估阶段。
