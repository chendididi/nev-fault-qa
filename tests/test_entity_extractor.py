"""知识图谱实体抽取测试。"""

from unittest.mock import MagicMock

from src.knowledge_graph.entity_extractor import batch_extract_entities, normalize_entities


def test_normalize_entities_uppercases_fault_codes_and_dedupes():
    normalized = normalize_entities(
        {
            "fault_codes": ["p0300", "P0300", "无效值"],
            "components": ["火花塞", " 火花塞 "],
            "symptoms": "发动机抖动",
            "subsystems": None,
            "tools": ["诊断仪"],
        }
    )

    assert normalized["fault_codes"] == ["P0300"]
    assert normalized["components"] == ["火花塞"]
    assert normalized["symptoms"] == ["发动机抖动"]
    assert normalized["subsystems"] == []
    assert normalized["tools"] == ["诊断仪"]


def test_batch_extract_entities_uses_cache_when_present(tmp_path):
    client = MagicMock()
    client.generate.return_value = '{"fault_codes": ["P0300"], "components": [], "symptoms": [], "subsystems": [], "tools": []}'
    chunks = [{"chunk_id": "c1", "text": "P0300 故障", "source": "manual.pdf", "page": 1, "chapter": ""}]

    first = batch_extract_entities(chunks, client, cache_dir=tmp_path)
    assert first[0]["entities"]["fault_codes"] == ["P0300"]
    assert client.generate.call_count == 1

    second_client = MagicMock()
    second = batch_extract_entities(chunks, second_client, cache_dir=tmp_path)
    assert second[0]["entities"]["fault_codes"] == ["P0300"]
    assert second_client.generate.call_count == 0

