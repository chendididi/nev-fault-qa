"""chunks 产物契约测试。"""

import pytest

from src.data_processing.chunk_contract import validate_chunks
from scripts.build_index import _table_records_to_chunks


def test_validate_chunks_accepts_valid_payload():
    validate_chunks(
        [
            {
                "chunk_id": "c1",
                "text": "P0300 多缸失火",
                "source": "sample.pdf",
                "page": 1,
                "chapter": "示例",
            }
        ]
    )


def test_validate_chunks_rejects_missing_fields():
    with pytest.raises(ValueError, match="缺少字段"):
        validate_chunks([{"chunk_id": "c1", "text": "x"}])


def test_table_records_are_normalized_to_chunks():
    chunks = _table_records_to_chunks(
        "sample.pdf",
        [
            {
                "fault_code": "P0300",
                "description": "多缸失火",
                "cause": "点火系统异常",
                "page": 10,
            }
        ],
    )

    assert len(chunks) == 1
    assert chunks[0]["source"] == "sample.pdf"
    assert chunks[0]["page"] == 10
    assert chunks[0]["chapter"] == "fault_code_table"
    assert "P0300" in chunks[0]["text"]
