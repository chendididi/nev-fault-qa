# 新能源汽车故障诊断智能问答系统
# NEV Fault Diagnosis Intelligent Q&A System

> "数字老师傅" — 将故障排查时间从小时级缩短至分钟级

基于 **Advanced RAG + 知识图谱** 的新能源汽车故障诊断问答系统，解决维修手册 3000+ 页难检索、新手技师经验断层问题。

---

## 目录

1. [系统架构](#系统架构)
2. [环境要求](#环境要求)
3. [快速开始（示例数据）](#快速开始示例数据)
4. [完整部署（真实手册）](#完整部署真实手册)
5. [启动服务](#启动服务)
6. [API 接口文档](#api-接口文档)
7. [配置说明](#配置说明)
8. [开发进度](#开发进度)

---

## 系统架构

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

---

## 环境要求

### 硬件
| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| 内存 | 32GB | 64GB |
| 磁盘 | 50GB（模型 + 数据） | 100GB |

> 无 GPU 时可跳过 Qwen 本地部署，改用阿里云 DashScope API（见[配置说明](#配置说明)）。

### 软件
- Python 3.10+
- Docker + Docker Compose（用于 Milvus 和 Neo4j）
- CUDA 11.8+（本地 GPU 部署时需要）

---

## 快速开始（示例数据）

> 用内置的 5 条示例数据验证完整流程，**无需真实手册**。

### 第一步：克隆项目并安装依赖

```bash
git clone <repo-url>
cd ~/ubuntuchen_file/nev-fault-qa

python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 第二步：启动基础服务（Milvus + Neo4j）

```bash
docker compose up -d
```

等待约 30 秒，验证服务正常：

```bash
# Milvus 健康检查
curl http://localhost:9091/healthz

# Neo4j 浏览器（用户名: neo4j，密码: nev_password_change_me）
# 浏览器访问 http://localhost:7474
```

### 第三步：准备示例数据

将示例数据复制到 `data/processed/`（供 BM25 索引使用）：

```bash
cp data/sample/sample_chunks.json data/processed/chunks.json
```

将示例数据导入 Milvus（向量索引）：

```bash
python - <<'EOF'
import json, sys
sys.path.insert(0, ".")
from src.retrieval.embedding_retriever import EmbeddingRetriever

chunks = json.loads(open("data/sample/sample_chunks.json", encoding="utf-8").read())
retriever = EmbeddingRetriever(device="cuda:0")  # 无 GPU 改为 device="cpu"
retriever.insert(chunks)
print(f"已导入 {len(chunks)} 条示例数据到 Milvus")
EOF
```

### 第四步：下载模型

所有模型均可通过代码自动下载，无需手动从网页下载。

> **国内网络提示**：HuggingFace 在中国大陆访问不稳定，建议先设置镜像：
> ```bash
> # Linux/Mac（临时生效）
> export HF_ENDPOINT=https://hf-mirror.com
>
> # Windows PowerShell（临时生效）
> $env:HF_ENDPOINT="https://hf-mirror.com"
> ```
> 或在虚拟环境激活后将上面的 export 写入 `.venv/bin/activate` 末尾，每次自动生效。

**BGE 嵌入模型（约 400MB）+ 重排序模型（约 1.1GB）**

这两个模型在第一次运行代码时会**自动下载**，无需额外操作。也可提前手动触发：

```bash
python scripts/download_models.py
```

**Qwen2-VL-7B（约 16GB，需要 GPU）**

通过 ModelScope 下载（国内速度快，无需梯子）：

```bash
python scripts/download_models.py --qwen
```

下载完成后，确认 `config/config.yaml` 中路径正确：

```yaml
models:
  qwen_vl:
    model_path: "/root/ubuntuchen_file/models/Qwen2-VL-7B-Instruct"
    device: "cuda:0"
```

> 无 GPU 或暂不下载 Qwen，可跳过此步，使用阿里云 API（见下方方案 B）。

**方案 B：阿里云 DashScope API（无需本地 GPU，适合开发阶段）**

1. 在阿里云百炼平台注册并申请 API Key（有免费额度）
2. 设置环境变量：

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxx"
```

3. 修改 `config/config.yaml`：

```yaml
models:
  qwen_vl:
    backend: "dashscope"
    model_name: "qwen-vl-plus"
```

---

## 完整部署（真实手册）

### 第一步：放入 PDF 手册

```bash
# 将维修手册 PDF 放入 data/raw/ 目录
cp /path/to/your/manual.pdf data/raw/
```

### 第二步：解析 PDF 并构建向量索引

```bash
# 解析 PDF + 存入 Milvus（约需 10-30 分钟，取决于文件大小）
python scripts/build_index.py --input data/raw/

# 如需跳过 Milvus，仅生成 chunks.json（可只用 BM25 检索）
python scripts/build_index.py --input data/raw/ --skip-embedding
```

解析结果保存在 `data/processed/chunks.json`。

### 第三步：构建知识图谱

```bash
# 从 chunks.json 提取实体关系并写入 Neo4j（约需 20-60 分钟）
python scripts/build_graph.py --input data/processed/chunks.json
```

完成后可在 Neo4j 浏览器（`http://localhost:7474`）执行以下 Cypher 验证：

```cypher
MATCH (fc:FaultCode) RETURN count(fc) AS 故障码数量
```

---

## 启动服务

### 启动后端 API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

启动成功日志：

```
=== NEV Fault QA System 启动中 ===
BM25 索引加载完成，5 个 chunks
=== 所有组件初始化完成，服务就绪 ===
INFO:     Uvicorn running on http://0.0.0.0:8000
```

健康检查：

```bash
curl http://localhost:8000/health
# 返回：{"status":"ok","version":"0.1.0"}
```

### 启动前端（Streamlit MVP）

```bash
streamlit run src/frontend/streamlit_app.py
```

浏览器访问 `http://localhost:8501`。

**界面功能**：
- 左侧快捷按钮：点击常见故障码（P0300、U0100 等）直接查询
- 主区域：流式对话，答案逐字显示
- 展开"查看引用来源"：显示答案来自手册哪一页

---

## API 接口文档

交互式文档：`http://localhost:8000/docs`

### 流式对话

```
POST /api/v1/chat/stream
Content-Type: application/json

{
  "message": "P0300 是什么故障？如何排查？",
  "session_id": "optional-uuid"
}
```

响应为 SSE 流（`text/event-stream`）：

```
data: {"token": "P0300"}
data: {"token": " 表示随机"}
data: {"token": "/多缸失火"}
...
data: {"done": true, "sources": [{"ref_id": 1, "source": "manual.pdf", "page": 42, ...}]}
```

### 非流式对话

```
POST /api/v1/chat
Content-Type: application/json

{
  "message": "发动机抖动怎么排查？"
}
```

响应：

```json
{
  "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "answer": "根据维修手册，发动机抖动的排查步骤如下...",
  "sources": [
    {"ref_id": 1, "source": "manual.pdf", "page": 15, "chapter": "第一章", "snippet": "..."}
  ]
}
```

### 按故障码诊断

```
POST /api/v1/diagnosis/fault-code
Content-Type: application/json

{"code": "P0300"}
```

响应：

```json
{
  "fault_code": "P0300",
  "components": ["火花塞", "点火线圈"],
  "subsystems": ["点火系统"],
  "symptoms": ["发动机抖动", "怠速不稳"],
  "rag_answer": "P0300 表示随机/多缸失火...",
  "sources": [...]
}
```

### 按症状诊断

```
POST /api/v1/diagnosis/symptom
Content-Type: application/json

{"symptom": "发动机抖动，怠速不稳"}
```

响应：

```json
{
  "possible_fault_codes": ["P0300", "P0301"],
  "rag_answer": "根据您描述的症状...",
  "sources": [...]
}
```

### 获取对话历史

```
GET /api/v1/chat/history/{session_id}
```

---

## 配置说明

主配置文件：`config/config.yaml`

本地覆盖（不提交 Git）：复制为 `config/config.local.yaml` 后修改。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `models.qwen_vl.model_path` | `/root/ubuntuchen_file/models/Qwen2-VL-7B-Instruct` | Qwen 模型本地路径 |
| `models.qwen_vl.device` | `cuda:0` | 推理设备，CPU 填 `cpu` |
| `models.embedding.model_name` | `BAAI/bge-base-zh-v1.5` | 嵌入模型（HF Hub 或本地路径）|
| `models.reranker.model_name` | `BAAI/bge-reranker-v2-m3` | 重排序模型 |
| `milvus.host` | `localhost` | Milvus 地址 |
| `milvus.port` | `19530` | Milvus 端口 |
| `neo4j.uri` | `bolt://localhost:7687` | Neo4j 连接地址 |
| `neo4j.password` | `nev_password_change_me` | Neo4j 密码（与 docker-compose 一致）|
| `retrieval.bm25_top_k` | `20` | BM25 每路候选数 |
| `retrieval.embedding_top_k` | `20` | 语义检索每路候选数 |
| `retrieval.rerank_output_k` | `5` | 最终送入 LLM 的片段数 |
| `data_processing.chunk_size` | `512` | PDF 分块大小（字符数）|
| `data_processing.chunk_overlap` | `64` | 相邻块重叠字符数 |

### 常见配置场景

**无 GPU 运行（仅 CPU）**：

```yaml
models:
  qwen_vl:
    device: "cpu"
  embedding:
    device: "cpu"
  reranker:
    device: "cpu"
```

> CPU 模式下推理较慢，建议仅用于功能验证。

**调整检索精度**：

```yaml
retrieval:
  rerank_output_k: 3    # 减少上下文片段数，加快生成速度
  bm25_top_k: 10        # 减少候选数，加快检索速度
```

---

## 运行测试

```bash
# 单元测试
pytest tests/test_retrieval.py -v
pytest tests/test_generation.py -v

# 所有测试
pytest tests/ -v

# RAGAS 质量评估（需要完整系统运行）
python tests/eval_ragas.py --qa-file tests/eval_qa_set.json
```

评估目标指标：

| 指标 | 目标 |
|------|------|
| Faithfulness（忠实度）| > 0.80 |
| Answer Relevancy（答案相关性）| > 0.75 |
| Context Precision（上下文精度）| > 0.70 |
| Context Recall（上下文召回）| > 0.70 |

---

## 项目结构

```
nev-fault-qa/
├── config/
│   └── config.yaml              # 统一配置（本地覆盖用 config.local.yaml）
├── data/
│   ├── raw/                     # 原始 PDF 手册（gitignore，不上传）
│   ├── processed/               # 解析后 JSON（build_index.py 生成）
│   └── sample/                  # 示例数据，5 条 chunk（已提交）
├── src/
│   ├── data_processing/
│   │   ├── pdf_parser.py        # PDF 文本分块
│   │   ├── table_extractor.py   # 故障码表格提取
│   │   └── circuit_ocr.py       # 电路图 OCR（Qwen2-VL 多模态）
│   ├── retrieval/
│   │   ├── bm25_retriever.py    # BM25 精确检索
│   │   ├── embedding_retriever.py # BGE 语义检索 + Milvus
│   │   ├── hybrid_retriever.py  # RRF 融合
│   │   └── reranker.py          # BGE Cross-Encoder 重排序
│   ├── knowledge_graph/
│   │   ├── entity_extractor.py  # 实体提取（故障码/零件/症状）
│   │   ├── neo4j_client.py      # Neo4j 查询客户端
│   │   └── relation_builder.py  # 图谱构建（写入 Neo4j）
│   ├── llm/
│   │   ├── qwen_client.py       # Qwen2-VL 推理封装
│   │   ├── query_rewriter.py    # 意图识别 + Query 改写
│   │   └── answer_generator.py  # RAG 答案生成
│   ├── api/
│   │   ├── main.py              # FastAPI 入口 + 组件初始化
│   │   └── routes/
│   │       ├── chat.py          # 对话接口（SSE 流式）
│   │       └── diagnosis.py     # 故障诊断接口
│   └── frontend/
│       └── streamlit_app.py     # Streamlit MVP 前端
├── scripts/
│   ├── build_index.py           # 构建 Milvus 向量索引
│   └── build_graph.py           # 构建 Neo4j 知识图谱
├── tests/
│   ├── test_retrieval.py        # 检索模块单元测试
│   ├── test_generation.py       # 生成模块单元测试
│   └── eval_ragas.py            # RAGAS 质量评估
├── docker-compose.yml           # Milvus + Neo4j 一键启动
├── requirements.txt             # Python 依赖
└── CLAUDE.md                    # 开发进度记录
```

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 多模态 LLM | Qwen2-VL-7B-Instruct | 本地 RTX 4090，16GB 显存 |
| 向量数据库 | Milvus 2.4 | 语义检索 |
| 知识图谱 | Neo4j 5 | 故障因果推理 |
| 精确检索 | BM25 (rank_bm25) | 故障码关键词匹配 |
| 语义嵌入 | BAAI/bge-base-zh-v1.5 | 中文语义编码，768 维 |
| 重排序 | BAAI/bge-reranker-v2-m3 | Top20 → Top5 精排 |
| PDF 解析 | PyMuPDF + pdfplumber | 文本 + 表格提取 |
| 后端 | FastAPI + uvicorn | SSE 流式输出 |
| 前端（MVP） | Streamlit | 快速验证 |
| 前端（正式） | Vue3 + TypeScript | 生产级 UI（待开发）|
| 评估 | RAGAS | Faithfulness + Answer Relevance |

---

## 开发进度

### ✅ 阶段一：项目脚手架（已完成）
所有模块代码已实现，git 仓库已初始化。

### ⏳ 阶段二：数据工程（待开始）
- [ ] 将官方 PDF 放入 `data/raw/`
- [ ] 验证 `pdf_parser.py` 分块效果
- [ ] 验证 `table_extractor.py` 故障码表提取
- [ ] 运行 `python scripts/build_index.py --input data/raw/`
- [ ] 验收：Milvus 可按语义检索到相关片段

### ⏳ 阶段三：知识图谱（待开始）
- [ ] 运行 `python scripts/build_graph.py`
- [ ] 验收：输入 P0300，返回相关零部件 + 子系统 + 维修链路

### ⏳ 阶段四：混合检索（待开始）
- [ ] 验证 Recall@5 和 MRR 指标
- [ ] 与单路检索对比提升效果

### ⏳ 阶段五：系统集成（待开始）
- [ ] 端到端：输入故障码/现象 → 获得答案 + 来源页码
- [ ] Streamlit MVP 可用

### ⏳ 阶段六：评估优化（待开始）
- [ ] 运行 `python tests/eval_ragas.py`
- [ ] 目标：Faithfulness > 0.80，Answer Relevancy > 0.80

---

## 开源协议

MIT License
