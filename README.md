# 新能源汽车故障诊断智能问答系统
# NEV Fault Diagnosis Intelligent Q&A System

> "数字老师傅" — 将故障排查时间从小时级缩短至分钟级
> "Digital Master Technician" — Reducing fault diagnosis from hours to minutes

## 项目背景 / Background

新能源汽车维修行业面临两大痛点：
1. **手册检索难**：比亚迪/特斯拉等品牌维修手册超过3000页，技师难以快速定位故障信息
2. **经验断层**：新手技师缺乏资深师傅的诊断经验，易走弯路

本系统通过 **Advanced RAG + 知识图谱** 融合方案，实现：
- 故障码精确匹配（BM25）
- 故障现象模糊语义检索（BGE Embedding）
- 故障因果链推理（Neo4j 知识图谱）
- 多模态理解（电路图、接线表）

---

The NEV repair industry faces two major pain points:
1. **Manual retrieval difficulty**: BYD/Tesla repair manuals exceed 3000 pages
2. **Experience gap**: Junior technicians lack the diagnostic intuition of senior masters

This system combines **Advanced RAG + Knowledge Graph** to enable:
- Exact fault code matching (BM25)
- Fuzzy symptom semantic search (BGE Embedding)
- Fault causation chain reasoning (Neo4j Knowledge Graph)
- Multimodal understanding (circuit diagrams, wiring tables)

## 系统架构 / Architecture

```
用户输入 (故障码 / 故障现象)
        │
        ▼
  Query重写 + 意图识别
        │
   ┌────┴────┐
   │         │
BM25精确   BGE语义
  检索       检索
   │         │
   └────┬────┘
        │  RRF融合
        ▼
   BGE Reranker重排
   (Top20 → Top5)
        │
   ┌────┴────┐
   │         │
RAG上下文  知识图谱
  片段       查询
   └────┬────┘
        ▼
  Qwen2-VL-7B生成
  (含引用溯源)
        │
        ▼
  FastAPI SSE流式输出
        │
        ▼
  Streamlit / Vue3前端
```

## 快速开始 / Quick Start

```bash
# 1. 克隆项目
git clone <repo-url>
cd nev-fault-qa

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境
cp config/config.yaml config/config.local.yaml
# 编辑 config.local.yaml，填写模型路径等

# 5. 启动基础设施
docker-compose up -d

# 6. 处理示例数据（可选，用示例数据验证流程）
python scripts/build_index.py --input data/sample/
python scripts/build_graph.py --input data/sample/

# 7. 启动 MVP 前端
streamlit run src/frontend/streamlit_app.py
```

## 技术栈 / Tech Stack

| 组件 | 技术 | 说明 |
|------|------|------|
| 多模态LLM | Qwen2-VL-7B-Instruct | 本地RTX 4090部署，16GB显存 |
| 向量数据库 | Milvus 2.4 | 语义检索 |
| 知识图谱 | Neo4j 5 | 故障因果推理 |
| 精确检索 | BM25 (rank_bm25) | 故障码匹配 |
| 语义嵌入 | BAAI/bge-base-zh-v1.5 | 中文语义编码 |
| 重排序 | BAAI/bge-reranker-v2-m3 | Top20→Top5精排 |
| PDF解析 | PyMuPDF + pdfplumber | 文本+表格提取 |
| 后端 | FastAPI | SSE流式输出 |
| 前端(MVP) | Streamlit | 快速验证 |
| 前端(正式) | Vue3 + TypeScript | 生产级UI |
| 评估 | RAGAS | Faithfulness + Answer Relevance |

## 评估指标 / Evaluation Metrics

- **检索层**：Recall@5、MRR（vs 单路检索基线）
- **生成层**：Faithfulness、Answer Relevance、Context Precision（RAGAS）

## 项目结构 / Project Structure

```
nev-fault-qa/
├── config/config.yaml          # 统一配置
├── data/
│   ├── processed/              # 解析后JSON
│   └── sample/                 # 示例数据（已提交）
├── src/
│   ├── data_processing/        # PDF解析管线
│   ├── knowledge_graph/        # 知识图谱构建
│   ├── retrieval/              # 混合检索
│   ├── llm/                    # LLM推理
│   ├── api/                    # FastAPI后端
│   └── frontend/               # 前端
├── scripts/                    # 索引/图谱构建脚本
└── tests/                      # 测试 + RAGAS评估
```

## 开源协议 / License

MIT License
