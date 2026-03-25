"""
多轮对话接口（SSE 流式输出）

POST /api/v1/chat
  - 接收用户消息和对话历史
  - 返回 text/event-stream 流式响应

GET /api/v1/chat/history/{session_id}
  - 获取对话历史（内存存储，重启后清空）
"""

import hashlib
import json
import re
import uuid
from collections import defaultdict
from time import perf_counter

from fastapi import APIRouter, Request
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.ops.logging_setup import log_context

router = APIRouter()

# 简单内存存储对话历史（生产环境应替换为 Redis）
_sessions: dict[str, list[dict]] = defaultdict(list)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None    # None 则自动生成


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict]
    trace_id: str | None = None


def _hash_query(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()[:12]


def _graph_stats(graph_data: dict | None) -> dict:
    if not graph_data:
        return {
            "graph_hit": False,
            "graph_components": 0,
            "graph_subsystems": 0,
            "graph_symptoms": 0,
            "graph_possible_fault_codes": 0,
        }
    return {
        "graph_hit": True,
        "graph_components": len(graph_data.get("components", [])),
        "graph_subsystems": len(graph_data.get("subsystems", [])),
        "graph_symptoms": len(graph_data.get("symptoms", [])),
        "graph_possible_fault_codes": len(graph_data.get("possible_fault_codes", [])),
    }


def _query_graph_data(state, query: str, rewritten_query: str, intent_value: str) -> dict | None:
    fault_codes = re.findall(r"[PBCU]\d{4}", f"{query} {rewritten_query}", re.IGNORECASE)
    if fault_codes:
        return state.neo4j.query_fault_code(fault_codes[0].upper())

    if intent_value == "symptom":
        keywords = [token for token in re.split(r"[\s,，。；;、]+", rewritten_query) if token]
        if not keywords:
            return None
        graph_results = state.neo4j.query_symptom_fault_codes(keywords[:6])
        possible_codes = list(dict.fromkeys(
            result["fault_code"] for result in graph_results if result.get("fault_code")
        ))[:5]
        symptoms = list(dict.fromkeys(
            result["symptom"] for result in graph_results if result.get("symptom")
        ))[:5]
        if possible_codes or symptoms:
            return {
                "fault_code": possible_codes[0] if possible_codes else "",
                "possible_fault_codes": possible_codes,
                "symptoms": symptoms,
                "components": [],
                "subsystems": [],
            }

    return None


async def _rag_pipeline(request: Request, query: str) -> tuple[list[dict], dict | None]:
    """
    执行完整 RAG 流水线，返回 (answer, sources)。

    步骤：Query重写 → 混合检索 → 重排 → 知识图谱查询 → 生成答案
    """
    state = request.app.state

    trace_payload = {
        "query_hash": _hash_query(query),
        "query_length": len(query),
    }

    total_start = perf_counter()
    # 1. Query 预处理
    rewrite_start = perf_counter()
    rewritten_query, intent = state.query_processor.process(query)
    intent_value = getattr(intent, "value", str(intent))
    rewrite_ms = int((perf_counter() - rewrite_start) * 1000)
    logger.bind(intent=intent_value, rewritten_query=rewritten_query).info("query_processed")

    # 2. 混合检索
    retrieval_start = perf_counter()
    retrieval_cfg = state.config["retrieval"]
    candidates, retrieval_stats = state.hybrid_retriever.retrieve_with_stats(
        rewritten_query,
        top_k=retrieval_cfg["rerank_input_k"],
    )
    retrieve_ms = int((perf_counter() - retrieval_start) * 1000)

    # 3. 重排序
    rerank_start = perf_counter()
    top_chunks = state.reranker.rerank(
        rewritten_query,
        candidates,
        top_k=retrieval_cfg["rerank_output_k"],
    )
    rerank_ms = int((perf_counter() - rerank_start) * 1000)

    # 4. 知识图谱查询
    graph_start = perf_counter()
    graph_data = _query_graph_data(state, query, rewritten_query, intent_value)
    graph_ms = int((perf_counter() - graph_start) * 1000)
    logger.bind(
        intent=intent_value,
        candidate_count=len(candidates),
        rerank_count=len(top_chunks),
        graph_hit=bool(graph_data),
    ).info("rag_pipeline_completed")

    total_ms = int((perf_counter() - total_start) * 1000)
    graph_stats = _graph_stats(graph_data)
    logger.bind(
        intent=intent_value,
        rewritten_length=len(rewritten_query),
        rerank_input_count=len(candidates),
        rerank_output_count=len(top_chunks),
        rewrite_ms=rewrite_ms,
        retrieve_ms=retrieve_ms,
        rerank_ms=rerank_ms,
        graph_ms=graph_ms,
        total_ms=total_ms,
        **trace_payload,
        **retrieval_stats,
        **graph_stats,
    ).info("rag_trace")

    return top_chunks, graph_data


@router.post("/chat/stream")
async def chat_stream(chat_req: ChatRequest, request: Request):
    """
    SSE 流式对话接口。
    """
    session_id = chat_req.session_id or str(uuid.uuid4())
    request.state.session_id = session_id
    query = chat_req.message
    request_logger = logger.bind(session_id=session_id, route="/api/v1/chat/stream")
    trace_id = getattr(request.state, "trace_id", None)

    async def event_generator():
        with log_context(session_id=session_id, route="/api/v1/chat/stream"):
            try:
                top_chunks, graph_data = await _rag_pipeline(request, query)
            except Exception as e:
                logger.exception("rag_pipeline_failed")
                yield {"data": json.dumps({"error": str(e), "trace_id": trace_id}, ensure_ascii=False)}
                return

        # 流式生成答案
        full_answer = ""
        gen = request.app.state.answer_generator.stream_generate(
            query=query,
            chunks=top_chunks,
            graph_data=graph_data,
        )

        for token in gen:
            full_answer += token
            yield {"data": json.dumps({"token": token}, ensure_ascii=False)}

        # 发送最终 sources
        sources = request.app.state.answer_generator.format_sources(top_chunks)
        request_logger.bind(source_count=len(sources)).info("chat_stream_completed")
        yield {
            "data": json.dumps(
                {"done": True, "sources": sources, "session_id": session_id, "trace_id": trace_id},
                ensure_ascii=False,
            )
        }

        # 存储对话历史
        _sessions[session_id].append({"role": "user", "content": query})
        _sessions[session_id].append({"role": "assistant", "content": full_answer})

    return EventSourceResponse(event_generator())


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_req: ChatRequest, request: Request):
    """
    非流式对话接口（兼容性备用）。
    """
    session_id = chat_req.session_id or str(uuid.uuid4())
    request.state.session_id = session_id
    query = chat_req.message
    request_logger = logger.bind(session_id=session_id, route="/api/v1/chat")
    trace_id = getattr(request.state, "trace_id", None)

    with log_context(session_id=session_id, route="/api/v1/chat"):
        top_chunks, graph_data = await _rag_pipeline(request, query)

    answer = request.app.state.answer_generator.generate(
        query=query,
        chunks=top_chunks,
        graph_data=graph_data,
    )
    sources = request.app.state.answer_generator.format_sources(top_chunks)

    _sessions[session_id].append({"role": "user", "content": query})
    _sessions[session_id].append({"role": "assistant", "content": answer})
    request_logger.bind(source_count=len(sources)).info("chat_completed")

    return ChatResponse(session_id=session_id, answer=answer, sources=sources, trace_id=trace_id)


@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    """获取指定会话的对话历史。"""
    return {"session_id": session_id, "history": _sessions.get(session_id, [])}
