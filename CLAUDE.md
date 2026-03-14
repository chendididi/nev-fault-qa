# 新能源汽车故障诊断智能问答系统

## 项目概述
基于 Advanced RAG + 知识图谱的新能源汽车故障诊断智能问答系统。
解决维修手册3000+页难检索、新手技师经验断层问题。

## 技术栈
- LLM: Qwen2-VL-7B-Instruct（本地 RTX 4090 部署）
- 向量库: Milvus 2.4
- 知识图谱: Neo4j 5
- 检索: BM25 + BGE Embedding 双路混合 + BGE Reranker 重排
- 后端: FastAPI（SSE流式输出）
- 前端: Streamlit（MVP）/ Vue3（正式版）
- 评估: RAGAS

## 开发规范
- Python 3.10+，使用 venv
- 配置统一在 config/config.yaml，禁止硬编码路径和密钥
- 模型文件（models/）和原始PDF（data/raw/）已 gitignore，不上传
- 中文注释为主，公开接口写 docstring
- 测试用 pytest

## 基础设施启动
```bash
docker-compose up -d   # 启动 Milvus + Neo4j
```

## 数据流
```
PDF → pdf_parser → table_extractor / circuit_ocr
    → build_index（Milvus）+ build_graph（Neo4j）
    → hybrid_retriever → answer_generator → FastAPI → 前端
```

## 目录说明
- `data/raw/`：原始PDF手册（gitignore，不上传）
- `data/processed/`：解析后JSON数据
- `data/sample/`：示例数据，可提交用于测试
- `src/data_processing/`：PDF解析、表格提取、电路图OCR
- `src/knowledge_graph/`：实体抽取、关系构建、Neo4j客户端
- `src/retrieval/`：BM25、向量检索、混合检索、重排序
- `src/llm/`：Qwen客户端、Query重写、答案生成
- `src/api/`：FastAPI后端
- `src/frontend/`：Streamlit MVP前端
- `scripts/`：一次性构建脚本
- `tests/`：单元测试 + RAGAS评估

## 开发进度

> 每次完成阶段后更新此区块。

### ✅ 阶段一：项目脚手架（已完成）
所有模块骨架文件已创建，git 仓库已初始化。

### ✅ 环境部署（已完成）
- 部署机器：学校 RTX 4090，WSL2 Ubuntu，路径 `/root/ubuntuchen_file/nev-fault-qa`
- Python 环境：.venv 已配置，依赖已安装（Python 3.13，pymilvus 2.5.3）
- 模型已下载：BGE 嵌入模型、BGE 重排序模型、Qwen2-VL-7B-Instruct
- 下一步：启动 Docker（Milvus + Neo4j），导入示例数据，跑通完整链路

### ⏳ 阶段二：数据工程（待开始）
- [ ] 将官方 PDF 放入 `data/raw/`
- [ ] 验证 `pdf_parser.py` 分块效果
- [ ] 验证 `table_extractor.py` 故障码表提取
- [ ] 运行 `python scripts/build_index.py --input data/raw/`
- [ ] 验收：Milvus 可按语义检索到相关片段

### ⏳ 阶段三：知识图谱（待开始）
- [ ] 运行 `python scripts/build_graph.py`
- [ ] 验收：输入 P0300，返回相关零部件+子系统+维修链路

### ⏳ 阶段四：混合检索（待开始）
- [ ] 验证 Recall@5 和 MRR 指标
- [ ] 与单路检索对比提升效果

### ⏳ 阶段五：系统集成（待开始）
- [ ] 端到端：输入故障码/现象 → 获得答案 + 来源页码
- [ ] Streamlit MVP 可用

### ⏳ 阶段六：评估优化（待开始）
- [ ] 运行 `python tests/eval_ragas.py`
- [ ] 目标：Faithfulness > 0.80，Answer Relevancy > 0.80
