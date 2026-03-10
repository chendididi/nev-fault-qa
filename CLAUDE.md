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
