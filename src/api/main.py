"""
FastAPI 应用入口

启动命令：
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes.chat import router as chat_router
from src.api.routes.diagnosis import router as diagnosis_router


def _load_config() -> dict:
    """加载 config/config.yaml。"""
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化模型和数据库连接。"""
    logger.info("=== NEV Fault QA System 启动中 ===")
    cfg = _load_config()
    app.state.config = cfg

    # 延迟导入（避免启动时加载未安装的依赖）
    from src.llm.qwen_client import QwenClient
    from src.llm.query_rewriter import QueryProcessor
    from src.llm.answer_generator import AnswerGenerator
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.embedding_retriever import EmbeddingRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import BGEReranker
    from src.knowledge_graph.neo4j_client import Neo4jClient

    qwen_cfg = cfg["models"]["qwen_vl"]
    emb_cfg = cfg["models"]["embedding"]
    reranker_cfg = cfg["models"]["reranker"]
    milvus_cfg = cfg["milvus"]
    neo4j_cfg = cfg["neo4j"]
    retrieval_cfg = cfg["retrieval"]

    # 初始化组件
    app.state.qwen = QwenClient(
        model_path=qwen_cfg["model_path"],
        device=qwen_cfg["device"],
        max_new_tokens=qwen_cfg["max_new_tokens"],
        temperature=qwen_cfg["temperature"],
        do_sample=qwen_cfg["do_sample"],
    )
    app.state.query_processor = QueryProcessor(app.state.qwen)
    app.state.answer_generator = AnswerGenerator(app.state.qwen)

    app.state.embedding_retriever = EmbeddingRetriever(
        model_name=emb_cfg["model_name"],
        milvus_host=milvus_cfg["host"],
        milvus_port=milvus_cfg["port"],
        collection_name=milvus_cfg["collection_name"],
        dim=milvus_cfg["dim"],
        device=emb_cfg["device"],
    )

    app.state.bm25_retriever = BM25Retriever()
    # BM25 索引从持久化 chunks 加载（由 build_index.py 生成）
    import json
    from pathlib import Path
    chunks_file = Path("data/processed/chunks.json")
    if chunks_file.exists():
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        app.state.bm25_retriever.build(chunks)
        logger.info(f"BM25 索引加载完成，{len(chunks)} 个 chunks")
    else:
        logger.warning("data/processed/chunks.json 不存在，BM25 检索不可用")

    app.state.hybrid_retriever = HybridRetriever(
        bm25=app.state.bm25_retriever,
        embedding=app.state.embedding_retriever,
        rrf_k=retrieval_cfg["rrf_k"],
        bm25_top_k=retrieval_cfg["bm25_top_k"],
        embedding_top_k=retrieval_cfg["embedding_top_k"],
    )

    app.state.reranker = BGEReranker(
        model_name=reranker_cfg["model_name"],
        device=reranker_cfg["device"],
        batch_size=reranker_cfg["batch_size"],
    )

    app.state.neo4j = Neo4jClient(
        uri=neo4j_cfg["uri"],
        username=neo4j_cfg["username"],
        password=neo4j_cfg["password"],
        database=neo4j_cfg["database"],
    )

    logger.info("=== 所有组件初始化完成，服务就绪 ===")
    yield

    # 关闭时清理
    app.state.neo4j.close()
    logger.info("=== 服务已停止 ===")


app = FastAPI(
    title="新能源汽车故障诊断智能问答系统",
    description="Advanced RAG + 知识图谱融合的 NEV 故障诊断 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS（允许 Streamlit / Vue3 跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1", tags=["对话"])
app.include_router(diagnosis_router, prefix="/api/v1", tags=["故障诊断"])


@app.get("/health")
async def health_check():
    """健康检查接口。"""
    return {"status": "ok", "version": "0.1.0"}
