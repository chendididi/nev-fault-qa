# 系统架构文档

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户层                              │
│  Streamlit MVP  /  Vue3 正式版  /  REST API 直接调用      │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    FastAPI 后端                           │
│  /api/v1/chat/stream (SSE)  /api/v1/diagnosis/*          │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    核心问答链路                            │
│                                                          │
│  用户输入                                                 │
│     │                                                    │
│     ▼                                                    │
│  QueryProcessor                                          │
│  ├── detect_intent()  →  FAULT_CODE / SYMPTOM / ...      │
│  └── rewrite_query()  →  规范化查询                       │
│     │                                                    │
│     ▼                                                    │
│  HybridRetriever                                         │
│  ├── BM25Retriever.search()    → 故障码精确匹配            │
│  ├── EmbeddingRetriever.search()→ BGE语义向量检索          │
│  └── reciprocal_rank_fusion()  → RRF融合Top-20           │
│     │                                                    │
│     ▼                                                    │
│  BGEReranker.rerank()          → Top-20 → Top-5          │
│     │                                                    │
│     ├── Neo4jClient.query_fault_code()  (知识图谱补充)    │
│     │                                                    │
│     ▼                                                    │
│  AnswerGenerator.stream_generate()                       │
│  └── Qwen2-VL-7B-Instruct → 流式输出 + 引用溯源           │
└─────────────────────────────────────────────────────────┘
```

## 数据处理管线

```
data/raw/*.pdf
    │
    ├── pdf_parser.py        → 文本分块（512字符，64重叠）
    │   └── chunks: [{chunk_id, text, source, page, chapter}]
    │
    ├── table_extractor.py   → 故障码定义表
    │   └── [{fault_code, description, cause, source, page}]
    │
    └── circuit_ocr.py       → Qwen2-VL 电路图 OCR
        └── [{chunk_id, text, source, page, chapter="circuit_diagram"}]

    所有 chunks 合并后：
    ├── build_index.py → data/processed/chunks.json
    │   ├── Milvus: EmbeddingRetriever.insert()
    │   └── BM25: BM25Retriever.build() (运行时加载)
    │
    └── build_graph.py
        ├── entity_extractor.py → Qwen2-VL 抽取实体
        └── relation_builder.py → Neo4j 写入节点和关系
```

## 知识图谱模式

```
节点类型：
  (FaultCode {code: "P0300"})
  (Component {name: "火花塞"})
  (Symptom {description: "发动机抖动"})
  (Subsystem {name: "发动机系统"})
  (Tool {name: "诊断仪"})

关系类型：
  (FaultCode)-[:INDICATES {weight}]->(Component)
  (FaultCode)-[:HAS_SYMPTOM]->(Symptom)
  (Component)-[:BELONGS_TO]->(Subsystem)
  (FaultCode)-[:CAUSED_BY]->(Component)
  (Symptom)-[:REQUIRES_TOOL]->(Tool)
```

## 检索策略对比

| 查询类型 | 示例 | 主要检索器 |
|---------|------|-----------|
| 精确故障码 | "P0300" | BM25 为主（精确匹配） |
| 故障现象描述 | "发动机抖动" | BGE Embedding 为主（语义） |
| 混合查询 | "P0300 会导致什么现象" | RRF 融合 |

## 性能指标目标

| 指标 | 目标值 | 说明 |
|------|-------|------|
| Recall@5 | > 0.85 | 相关文档命中率 |
| MRR | > 0.75 | 平均倒数排名 |
| Faithfulness | > 0.80 | 答案与检索内容的一致性 |
| Answer Relevancy | > 0.80 | 答案对问题的相关性 |
| 端到端延迟 | < 30s | P95 响应时间（RTX 4090） |
