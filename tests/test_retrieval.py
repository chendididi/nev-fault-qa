"""
检索模块单元测试

测试 BM25、向量检索、混合检索的基本功能。
使用示例数据（data/sample/）避免依赖完整 PDF。
"""

import pytest

from src.retrieval.bm25_retriever import BM25Retriever, _tokenize
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion


# ─── 测试数据 ──────────────────────────────────────────────────────────
SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk_001",
        "text": "P0300 随机/多缸失火检测。该故障码表示发动机出现随机或多缸失火，可能导致排放超标。",
        "source": "sample.pdf",
        "page": 42,
        "chapter": "第三章 发动机故障码",
    },
    {
        "chunk_id": "chunk_002",
        "text": "电池管理系统（BMS）负责监控电池组的电压、温度和电流。当检测到异常时会触发相关故障码。",
        "source": "sample.pdf",
        "page": 100,
        "chapter": "第五章 电驱动系统",
    },
    {
        "chunk_id": "chunk_003",
        "text": "车辆启动困难可能由多种原因引起：电池电量不足、启动电机故障、点火系统异常等。",
        "source": "sample.pdf",
        "page": 15,
        "chapter": "第一章 常见故障现象",
    },
    {
        "chunk_id": "chunk_004",
        "text": "U0100 与发动机控制模块通信丢失。检查 CAN 总线连接和 ECU 供电电压。",
        "source": "sample.pdf",
        "page": 200,
        "chapter": "第七章 网络通信故障码",
    },
]


# ─── Tokenizer 测试 ────────────────────────────────────────────────────
class TestTokenizer:
    def test_tokenize_fault_code_preserved(self):
        """故障码不应被分词切割。"""
        tokens = _tokenize("P0300 故障")
        assert "P0300" in tokens

    def test_tokenize_chinese(self):
        """中文应被正确分词。"""
        tokens = _tokenize("电池管理系统")
        assert len(tokens) > 0

    def test_tokenize_mixed(self):
        """中英混合文本。"""
        tokens = _tokenize("U0100 CAN总线通信故障")
        assert "U0100" in tokens
        assert any("CAN" in t or "总线" in t for t in tokens)


# ─── BM25 检索测试 ─────────────────────────────────────────────────────
class TestBM25Retriever:
    @pytest.fixture
    def retriever(self):
        r = BM25Retriever()
        r.build(SAMPLE_CHUNKS)
        return r

    def test_build(self, retriever):
        assert retriever.is_built

    def test_fault_code_search(self, retriever):
        """故障码精确搜索应命中相关文档。"""
        results = retriever.search("P0300", top_k=3)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "chunk_001"

    def test_semantic_search(self, retriever):
        """语义关键词搜索。"""
        results = retriever.search("启动困难", top_k=3)
        assert len(results) > 0

    def test_top_k_limit(self, retriever):
        results = retriever.search("故障", top_k=2)
        assert len(results) <= 2

    def test_not_built_raises(self):
        r = BM25Retriever()
        with pytest.raises(RuntimeError, match="未构建"):
            r.search("test")


# ─── RRF 融合测试 ─────────────────────────────────────────────────────
class TestRRF:
    def test_rrf_merges_results(self):
        bm25_results = [
            {"chunk_id": "a", "text": "text a", "bm25_score": 10.0},
            {"chunk_id": "b", "text": "text b", "bm25_score": 5.0},
        ]
        emb_results = [
            {"chunk_id": "b", "text": "text b", "embedding_score": 0.9},
            {"chunk_id": "c", "text": "text c", "embedding_score": 0.7},
        ]
        merged = reciprocal_rank_fusion(bm25_results, emb_results, k=60)

        ids = [m["chunk_id"] for m in merged]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids

    def test_rrf_both_ranked_higher(self):
        """同时出现在两路检索的文档 RRF 分数更高。"""
        bm25_results = [
            {"chunk_id": "shared", "text": "", "bm25_score": 5.0},
            {"chunk_id": "only_bm25", "text": "", "bm25_score": 10.0},
        ]
        emb_results = [
            {"chunk_id": "shared", "text": "", "embedding_score": 0.9},
            {"chunk_id": "only_emb", "text": "", "embedding_score": 0.8},
        ]
        merged = reciprocal_rank_fusion(bm25_results, emb_results)

        # "shared" 出现在两路，应该排名更高
        merged_ids = [m["chunk_id"] for m in merged]
        assert merged_ids[0] == "shared"

    def test_rrf_score_field(self):
        """验证 rrf_score 字段存在。"""
        results = reciprocal_rank_fusion(
            [{"chunk_id": "x", "text": ""}],
            [{"chunk_id": "x", "text": ""}],
        )
        assert "rrf_score" in results[0]
