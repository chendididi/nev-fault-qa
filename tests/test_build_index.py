"""
build_index 脚本的轻量测试。
"""

import json

import pytest

from scripts.build_index import load_input_chunks


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


def test_load_input_chunks_raises_when_dir_has_no_supported_input(tmp_path):
    with pytest.raises(ValueError, match="未发现可导入的 PDF 或 sample_chunks.json"):
        load_input_chunks(tmp_path, chunk_size=512, chunk_overlap=64)
