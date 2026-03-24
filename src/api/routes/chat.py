"""
多轮对话接口（SSE 流式输出）

POST /api/v1/chat
  - 接收用户消息和对话历史
  - 返回 text/event-stream 流式响应

GET /api/v1/chat/history/{session_id}
  - 获取对话历史（内存存储，重启后清空）
"""

import json
import uuid
from collections import defaultdict

from fastapi import APIRouter, Request
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

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


async def _rag_pipeline(request: Request, query: str) -> tuple[str, list[dict]]:
    """
    执行完整 RAG 流水线，返回 (answer, sources)。

    步骤：Query重写 → 混合检索 → 重排 → 知识图谱查询 → 生成答案
    """
    state = request.app.state

    # 1. Query 预处理
    rewritten_query, intent = state.query_processor.process(query)
    logger.info(f"意图: {intent}, 改写后: {rewritten_query!r}")

    # 2. 混合检索
    retrieval_cfg = state.config["retrieval"]
    candidates = state.hybrid_retriever.retrieve(
        rewritten_query,
        top_k=retrieval_cfg["rerank_input_k"],
    )

    # 3. 重排序
    top_chunks = state.reranker.rerank(
        rewritten_query,
        candidates,
        top_k=retrieval_cfg["rerank_output_k"],
    )

    # 4. 知识图谱查询（仅故障码意图）
    import re
    graph_data = None
    fault_codes = re.findall(r"[PBCU]\d{4}", query, re.IGNORECASE)
    if fault_codes:
        graph_data = state.neo4j.query_fault_code(fault_codes[0].upper())

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

    async def event_generator():
        try:
            top_chunks, graph_data = await _rag_pipeline(request, query)
        except Exception as e:
            logger.error(f"RAG 流水线错误: {e}")
            yield {"data": json.dumps({"error": str(e)}, ensure_ascii=False)}
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
                {"done": True, "sources": sources, "session_id": session_id},
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

    return ChatResponse(session_id=session_id, answer=answer, sources=sources)


@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    """获取指定会话的对话历史。"""
    return {"session_id": session_id, "history": _sessions.get(session_id, [])}
