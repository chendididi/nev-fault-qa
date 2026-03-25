"""
FastAPI 应用入口

启动命令：
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.health import router as health_router
from src.api.routes.chat import router as chat_router
from src.api.routes.diagnosis import router as diagnosis_router
from src.config_loader import load_config
from src.ops.logging_setup import log_context, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化模型和数据库连接。"""
    cfg = load_config()
    setup_logging(cfg, component="api")
    app.state.config = cfg
    app.state.chunks_file_path = "data/processed/chunks.json"
    logger.info("=== NEV Fault QA System 启动中 ===")

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
    logging_cfg = cfg.get("logging", {})
    trace_enabled = bool(logging_cfg.get("trace_enabled", False))

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
        trace_enabled=trace_enabled,
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
        trace_enabled=trace_enabled,
    )

    logger.info("=== 所有组件初始化完成，服务就绪 ===")
    yield

    # 关闭时清理
    app.state.neo4j.close()
    logger.info("=== 服务已停止 ===")


def create_app(lifespan_handler=lifespan) -> FastAPI:
    app = FastAPI(
        title="新能源汽车故障诊断智能问答系统",
        description="Advanced RAG + 知识图谱融合的 NEV 故障诊断 API",
        version="0.1.0",
        lifespan=lifespan_handler,
    )

    # CORS（允许 Streamlit / Vue3 跨域）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        trace_id = request.headers.get("X-Trace-ID") or request_id
        request.state.request_id = request_id
        request.state.trace_id = trace_id

        with log_context(
            component="api",
            request_id=request_id,
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        ):
            started_at = perf_counter()
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("request_failed")
                raise

            latency_ms = int((perf_counter() - started_at) * 1000)
            logger.bind(
                session_id=getattr(request.state, "session_id", None),
                status_code=response.status_code,
                latency_ms=latency_ms,
            ).info("request_completed")
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id
            return response

    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api/v1", tags=["对话"])
    app.include_router(diagnosis_router, prefix="/api/v1", tags=["故障诊断"])
    return app


app = create_app()
