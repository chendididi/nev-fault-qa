"""
答案生成模块单元测试

测试 Prompt 构建、来源格式化等不依赖 GPU 的逻辑。
LLM 推理本身通过 mock 跳过。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.answer_generator import (
    AnswerGenerator,
    _build_context,
    _build_graph_section,
)
from src.llm.query_rewriter import QueryIntent, detect_intent


# ─── 测试数据 ──────────────────────────────────────────────────────────
SAMPLE_CHUNKS = [
    {
        "chunk_id": "c1",
        "text": "P0300 故障码表示多缸失火，需检查点火系统。",
        "source": "byd_manual.pdf",
        "page": 42,
        "chapter": "发动机系统",
    },
    {
        "chunk_id": "c2",
        "text": "失火故障可能由火花塞老化、燃油喷射器堵塞引起。",
        "source": "byd_manual.pdf",
        "page": 43,
        "chapter": "发动机系统",
    },
]

SAMPLE_GRAPH_DATA = {
    "fault_code": "P0300",
    "components": ["火花塞", "燃油喷射器"],
    "subsystems": ["发动机系统"],
    "symptoms": ["发动机抖动", "怠速不稳"],
}


# ─── 上下文构建测试 ────────────────────────────────────────────────────
class TestBuildContext:
    def test_basic_context(self):
        ctx = _build_context(SAMPLE_CHUNKS)
        assert "P0300" in ctx
        assert "byd_manual.pdf" in ctx
        assert "42" in ctx

    def test_max_chars_limit(self):
        """max_chars 参数限制总长度。"""
        ctx = _build_context(SAMPLE_CHUNKS, max_chars=50)
        assert len(ctx) <= 200  # 包含来源标签，可能略超

    def test_source_reference_format(self):
        ctx = _build_context(SAMPLE_CHUNKS)
        assert "[1]" in ctx
        assert "[2]" in ctx


# ─── 图谱信息构建测试 ─────────────────────────────────────────────────
class TestBuildGraphSection:
    def test_with_data(self):
        section = _build_graph_section(SAMPLE_GRAPH_DATA)
        assert "P0300" in section
        assert "火花塞" in section
        assert "发动机系统" in section

    def test_with_none(self):
        section = _build_graph_section(None)
        assert section == ""

    def test_with_empty(self):
        section = _build_graph_section({"fault_code": "P0300", "components": [], "subsystems": [], "symptoms": []})
        assert section == ""


# ─── AnswerGenerator 测试（mock LLM）─────────────────────────────────
class TestAnswerGenerator:
    @pytest.fixture
    def generator(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = "P0300 表示多缸失火，建议检查火花塞。[1] byd_manual.pdf 第42页"
        return AnswerGenerator(mock_client)

    def test_generate_calls_llm(self, generator):
        answer = generator.generate("P0300是什么", SAMPLE_CHUNKS, SAMPLE_GRAPH_DATA)
        assert len(answer) > 0
        generator._client.generate.assert_called_once()

    def test_format_sources(self, generator):
        sources = generator.format_sources(SAMPLE_CHUNKS)
        assert len(sources) == 2
        assert sources[0]["ref_id"] == 1
        assert sources[0]["source"] == "byd_manual.pdf"
        assert sources[0]["page"] == 42
        assert "snippet" in sources[0]


# ─── 意图识别测试 ─────────────────────────────────────────────────────
class TestDetectIntent:
    def test_fault_code_intent(self):
        assert detect_intent("P0300 是什么故障") == QueryIntent.FAULT_CODE

    def test_fault_code_u_series(self):
        assert detect_intent("U0100 怎么处理") == QueryIntent.FAULT_CODE

    def test_procedure_intent(self):
        assert detect_intent("怎么更换火花塞") == QueryIntent.PROCEDURE

    def test_symptom_intent(self):
        assert detect_intent("车子发动机抖动") == QueryIntent.SYMPTOM

    def test_general_without_llm(self):
        """无法快速判断时，不传 client 返回 GENERAL。"""
        intent = detect_intent("新能源汽车充电多少钱")
        assert intent == QueryIntent.GENERAL
