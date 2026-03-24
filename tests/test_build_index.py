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


def test_load_input_chunks_recursively_combines_supported_inputs(tmp_path):
    pdf_file = tmp_path / "official" / "byd" / "manual.pdf"
    pdf_file.parent.mkdir(parents=True)
    pdf_file.write_bytes(b"%PDF-1.4")

    html_file = tmp_path / "official" / "tesla" / "manual" / "GUID-1.html"
    html_file.parent.mkdir(parents=True)
    html_file.write_text("<html></html>", encoding="utf-8")

    chunk_file = tmp_path / "official" / "byd" / "prebuilt_chunks.json"
    chunk_file.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "json1",
                    "text": "预构建 chunks",
                    "source": "byd.json",
                    "page": 1,
                    "chapter": "预构建",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (tmp_path / "official" / "manifest.json").write_text(
        json.dumps({"entries": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "official" / "byd" / "manual.pdf.meta.json").write_text(
        json.dumps({"url": "https://example.com/manual.pdf"}, ensure_ascii=False),
        encoding="utf-8",
    )

    with (
        patch(
            "scripts.build_index.parse_pdf",
            return_value=[
                {
                    "chunk_id": "pdf1",
                    "text": "PDF chunk",
                    "source": "manual.pdf",
                    "page": 1,
                    "chapter": "PDF",
                }
            ],
        ) as mocked_parse_pdf,
        patch("scripts.build_index.extract_tables", return_value=[]),
        patch(
            "scripts.build_index.parse_html_file",
            return_value=[
                {
                    "chunk_id": "html1",
                    "text": "HTML chunk",
                    "source": "GUID-1.html",
                    "page": 1,
                    "chapter": "HTML",
                }
            ],
        ) as mocked_parse_html,
    ):
        loaded, metadata = load_input_chunks(
            tmp_path / "official",
            chunk_size=512,
            chunk_overlap=64,
            return_metadata=True,
        )

    assert {chunk["chunk_id"] for chunk in loaded} == {"json1", "pdf1", "html1"}
    assert metadata["pdf_count"] == 1
    assert metadata["html_count"] == 1
    assert metadata["chunk_json_count"] == 1
    assert metadata["skipped_json_count"] == 2
    mocked_parse_pdf.assert_called_once_with(pdf_file, chunk_size=512, chunk_overlap=64)
    mocked_parse_html.assert_called_once_with(html_file, chunk_size=512, chunk_overlap=64)


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


def test_load_input_chunks_multiple_inputs_reconciles_duplicate_chunk_ids(tmp_path):
    first_chunks = tmp_path / "first_chunks.json"
    second_chunks = tmp_path / "second_chunks.json"
    third_chunks = tmp_path / "third_chunks.json"

    payload_a = [
        {
            "chunk_id": "dup1",
            "text": "相同的 chunk",
            "source": "a.pdf",
            "page": 1,
            "chapter": "A",
        }
    ]
    payload_b = [
        {
            "chunk_id": "dup1",
            "text": "相同的 chunk",
            "source": "a.pdf",
            "page": 1,
            "chapter": "A",
        }
    ]
    payload_c = [
        {
            "chunk_id": "dup1",
            "text": "发生冲突的 chunk",
            "source": "b.pdf",
            "page": 2,
            "chapter": "B",
        }
    ]

    first_chunks.write_text(json.dumps(payload_a, ensure_ascii=False), encoding="utf-8")
    second_chunks.write_text(json.dumps(payload_b, ensure_ascii=False), encoding="utf-8")
    third_chunks.write_text(json.dumps(payload_c, ensure_ascii=False), encoding="utf-8")

    loaded, metadata = load_input_chunks(
        [first_chunks, second_chunks, third_chunks],
        chunk_size=512,
        chunk_overlap=64,
        return_metadata=True,
    )

    assert len(loaded) == 2
    assert metadata["chunk_json_count"] == 3
    assert metadata["skipped_duplicate_chunk_count"] == 1
    assert metadata["remapped_chunk_id_count"] == 1
    assert len({chunk["chunk_id"] for chunk in loaded}) == 2
    assert any(chunk["chunk_id"] == "dup1" for chunk in loaded)
    assert any(chunk["text"] == "发生冲突的 chunk" and chunk["chunk_id"] != "dup1" for chunk in loaded)


def test_load_input_chunks_raises_when_dir_has_no_supported_input(tmp_path):
    with pytest.raises(ValueError, match="未发现可导入的 PDF、HTML 或 chunks JSON"):
        load_input_chunks(tmp_path, chunk_size=512, chunk_overlap=64)
