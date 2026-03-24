"""健康检查与 readiness 测试。"""

from fastapi.testclient import TestClient

from src.api.main import create_app


class StubBM25Retriever:
    is_built = True


class StubEmbeddingRetriever:
    def ping(self) -> bool:
        return True


class StubNeo4jClient:
    def ping(self) -> bool:
        return True


def build_test_client(chunks_file_path) -> TestClient:
    app = create_app(lifespan_handler=None)
    app.state.bm25_retriever = StubBM25Retriever()
    app.state.embedding_retriever = StubEmbeddingRetriever()
    app.state.neo4j = StubNeo4jClient()
    app.state.qwen = object()
    app.state.chunks_file_path = str(chunks_file_path)
    return TestClient(app)


def test_health_returns_liveness_payload(tmp_path):
    client = build_test_client(tmp_path / "chunks.json")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == "0.1.0"
    assert response.headers["X-Request-ID"]


def test_ready_reports_ok_when_all_checks_pass(tmp_path):
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text("[]", encoding="utf-8")
    client = build_test_client(chunks_file)

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["chunks_file_present"] is True
    assert payload["checks"]["bm25_loaded"] is True
    assert payload["checks"]["milvus_connected"] is True
    assert payload["checks"]["neo4j_connected"] is True
    assert payload["checks"]["qwen_loaded"] is True


def test_ready_reports_degraded_when_dependency_missing(tmp_path):
    client = build_test_client(tmp_path / "missing.json")
    client.app.state.qwen = None

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["checks"]["chunks_file_present"] is False
    assert payload["checks"]["qwen_loaded"] is False

