"""
聊天路由轻量测试。

避免加载真实模型，通过 stub app.state 验证 API 契约。
"""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.chat import router


class StubQueryProcessor:
    def process(self, query: str):
        return query, "general"


class StubHybridRetriever:
    def retrieve(self, query: str, top_k: int = 20):
        return [
            {
                "chunk_id": "c1",
                "text": "P0300 表示随机/多缸失火。",
                "source": "sample.pdf",
                "page": 1,
                "chapter": "示例",
            }
        ]

    def retrieve_with_stats(self, query: str, top_k: int = 20):
        return self.retrieve(query, top_k), {
            "bm25_hits": 1,
            "embedding_hits": 0,
            "rrf_candidates": 1,
            "bm25_top_k": 20,
            "embedding_top_k": 20,
            "rrf_k": 60,
        }


class StubReranker:
    def rerank(self, query: str, chunks: list[dict], top_k: int = 5):
        return chunks[:top_k]


class StubNeo4j:
    def query_fault_code(self, fault_code: str):
        return {
            "fault_code": fault_code,
            "components": ["火花塞"],
            "subsystems": ["发动机系统"],
            "symptoms": ["抖动"],
        }

    def query_symptom_fault_codes(self, keywords: list[str]):
        return [{"fault_code": "P0300", "symptom": "抖动"}]


class StubAnswerGenerator:
    def generate(self, query: str, chunks: list[dict], graph_data: dict | None = None):
        return "诊断结果"

    def stream_generate(self, query: str, chunks: list[dict], graph_data: dict | None = None):
        yield "诊断"
        yield "结果"

    def format_sources(self, chunks: list[dict]):
        return [
            {
                "ref_id": 1,
                "source": chunks[0]["source"],
                "page": chunks[0]["page"],
                "chapter": chunks[0]["chapter"],
                "snippet": chunks[0]["text"],
            }
        ]


def build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.config = {"retrieval": {"rerank_input_k": 20, "rerank_output_k": 5}}
    app.state.query_processor = StubQueryProcessor()
    app.state.hybrid_retriever = StubHybridRetriever()
    app.state.reranker = StubReranker()
    app.state.neo4j = StubNeo4j()
    app.state.answer_generator = StubAnswerGenerator()
    return TestClient(app)


def test_chat_returns_session_id_and_sources():
    client = build_test_client()

    response = client.post("/api/v1/chat", json={"message": "P0300 是什么故障"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "诊断结果"
    assert data["session_id"]
    assert data["sources"][0]["source"] == "sample.pdf"


def test_chat_stream_done_event_includes_session_id():
    client = build_test_client()

    with client.stream("POST", "/api/v1/chat/stream", json={"message": "P0300 是什么故障"}) as response:
        assert response.status_code == 200
        events = [line for line in response.iter_lines() if line]

    payloads = [
        json.loads(line.removeprefix("data:").strip())
        for line in events
        if line.startswith("data:")
    ]

    assert payloads[0]["token"] == "诊断"
    assert payloads[1]["token"] == "结果"
    assert payloads[-1]["done"] is True
    assert payloads[-1]["session_id"]
    assert payloads[-1]["sources"][0]["source"] == "sample.pdf"
