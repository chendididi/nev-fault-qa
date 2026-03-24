"""
build_index 脚本的轻量测试。
"""

import json
from unittest.mock import patch

import pytest

from scripts.build_index import load_input_chunks, table_records_to_chunks


def test_load_input_chunks_from_sample_json(tmp_path):
    chunks = [
        {
            "chunk_id": "chunk_001",
            "text": "P0300 故障码表示多缸失火。",
            "source": "sample.pdf",
            "page": 1,
            "chapter": "示例章节",
        }
    ]
    sample_file = tmp_path / "sample_chunks.json"
    sample_file.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    loaded = load_input_chunks(tmp_path, chunk_size=512, chunk_overlap=64)

    assert loaded == chunks


def test_load_input_chunks_from_direct_json_file(tmp_path):
    chunk_file = tmp_path / "chunks.json"
    chunk_file.write_text(
        json.dumps(
            [{"chunk_id": "c1", "text": "test", "source": "x.pdf", "page": 1, "chapter": ""}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = load_input_chunks(chunk_file, chunk_size=512, chunk_overlap=64)

    assert loaded[0]["chunk_id"] == "c1"


def test_load_input_chunks_for_single_pdf_invokes_pdf_pipeline(tmp_path):
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    with patch("scripts.build_index.parse_pdf", return_value=[{"chunk_id": "pdf1"}]) as mocked_parse:
        loaded = load_input_chunks(
            pdf_file,
            chunk_size=512,
            chunk_overlap=64,
            table_extraction=False,
            with_circuit_ocr=False,
        )

    mocked_parse.assert_called_once()
    assert loaded == [{"chunk_id": "pdf1"}]


def test_table_records_to_chunks_generates_searchable_text():
    chunks = table_records_to_chunks(
        "manual.pdf",
        [
            {
                "fault_code": "P0300",
                "description": "多缸失火",
                "cause": "点火系统异常",
                "page": 12,
            }
        ],
    )

    assert chunks[0]["chapter"] == "fault_code_table"
    assert "P0300" in chunks[0]["text"]
    assert "点火系统异常" in chunks[0]["text"]


def test_load_input_chunks_raises_when_dir_has_no_supported_input(tmp_path):
    with pytest.raises(ValueError, match="未发现可导入的 PDF 或 JSON"):
        load_input_chunks(tmp_path, chunk_size=512, chunk_overlap=64)
