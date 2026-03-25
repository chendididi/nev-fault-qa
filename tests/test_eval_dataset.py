"""
固定评测集结构测试。
"""

import json
from pathlib import Path

import pytest


def test_eval_dataset_exists_and_valid():
    qa_path = Path("tests/eval_qa_set.json")
    assert qa_path.exists(), "tests/eval_qa_set.json 缺失"

    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload, "评测集必须是非空列表"

    required = {"question", "ground_truth", "ground_truth_context"}
    for idx, item in enumerate(payload):
        assert required.issubset(item), f"第 {idx} 项缺少字段"
        for key in required:
            assert isinstance(item[key], str) and item[key].strip(), f"第 {idx} 项 {key} 为空"

    # 防止评测集意外缩到 1 条
    assert len(payload) >= 5, "评测集条目过少"
