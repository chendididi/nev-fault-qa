"""知识图谱关系构建测试。"""

from src.knowledge_graph.relation_builder import build_graph_from_chunks


class FakeNeo4j:
    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []

    def run(self, query: str, parameters: dict | None = None):
        self.calls.append((query, parameters))
        return []


def test_build_graph_from_chunks_creates_provenance_relations():
    chunk = {
        "chunk_id": "chunk-1",
        "text": "P0300 多缸失火，检查火花塞，使用诊断仪。",
        "source": "manual.pdf",
        "page": 12,
        "chapter": "发动机故障",
        "entities": {
            "fault_codes": ["P0300"],
            "components": ["火花塞"],
            "symptoms": ["发动机抖动"],
            "subsystems": ["发动机系统"],
            "tools": ["诊断仪"],
        },
    }
    neo4j = FakeNeo4j()

    build_graph_from_chunks([chunk], neo4j, run_id="run-123")

    call_text = "\n".join(query for query, _ in neo4j.calls)
    assert "CAUSED_BY" in call_text
    assert "REQUIRES_TOOL" in call_text
    assert any(params.get("chunk_id") == "chunk-1" for _, params in neo4j.calls if params)
    assert any(params.get("run_id") == "run-123" for _, params in neo4j.calls if params)
