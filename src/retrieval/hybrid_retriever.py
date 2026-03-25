"""
混合检索器（BM25 + Embedding RRF 融合）

使用 Reciprocal Rank Fusion (RRF) 算法将 BM25 精确检索和向量语义检索的结果合并。
RRF 公式：score(d) = Σ 1/(k + rank_i(d))，k 通常取 60
"""

from loguru import logger

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever


def _rrf_score(rank: int, k: int = 60) -> float:
    """RRF 单路得分。"""
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    embedding_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    对两路检索结果执行 RRF 融合。

    Args:
        bm25_results: BM25 检索结果列表（已按分数排序）
        embedding_results: 向量检索结果列表（已按分数排序）
        k: RRF 常数

    Returns:
        按 RRF 分数排序的融合结果列表，每个 dict 含 rrf_score 字段
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    # BM25 路
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank, k)
        chunk_map[cid] = chunk

    # Embedding 路
    for rank, chunk in enumerate(embedding_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank, k)
        chunk_map.setdefault(cid, chunk)

    # 合并并排序
    merged = []
    for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"] = score
        merged.append(chunk)

    return merged


class HybridRetriever:
    """BM25 + 向量检索混合检索器。"""

    def __init__(
        self,
        bm25: BM25Retriever,
        embedding: EmbeddingRetriever,
        rrf_k: int = 60,
        bm25_top_k: int = 20,
        embedding_top_k: int = 20,
        trace_enabled: bool = False,
    ):
        """
        Args:
            bm25: 已构建的 BM25Retriever
            embedding: 已初始化的 EmbeddingRetriever
            rrf_k: RRF 常数
            bm25_top_k: BM25 每路候选数
            embedding_top_k: 向量检索每路候选数
        """
        self._bm25 = bm25
        self._embedding = embedding
        self._rrf_k = rrf_k
        self._bm25_top_k = bm25_top_k
        self._embedding_top_k = embedding_top_k
        self._trace_enabled = trace_enabled

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """
        执行混合检索并返回 RRF 融合结果。

        Args:
            query: 用户查询（支持故障码或自然语言故障现象）
            top_k: 最终返回候选数（送入 reranker 前）

        Returns:
            按 rrf_score 排序的 chunk 列表
        """
        merged, _ = self.retrieve_with_stats(query, top_k=top_k)
        return merged

    def retrieve_with_stats(self, query: str, top_k: int = 20) -> tuple[list[dict], dict]:
        """
        执行混合检索并返回 (结果, 统计信息)。

        Returns:
            (chunk 列表, stats dict)
        """
        logger.debug(f"混合检索: {query!r}")

        bm25_results = self._bm25.search(query, top_k=self._bm25_top_k)
        embedding_results = self._embedding.search(query, top_k=self._embedding_top_k)

        logger.debug(f"BM25 命中: {len(bm25_results)}, Embedding 命中: {len(embedding_results)}")

        merged = reciprocal_rank_fusion(bm25_results, embedding_results, k=self._rrf_k)
        merged = merged[:top_k]
        stats = {
            "bm25_hits": len(bm25_results),
            "embedding_hits": len(embedding_results),
            "rrf_candidates": len(merged),
            "bm25_top_k": self._bm25_top_k,
            "embedding_top_k": self._embedding_top_k,
            "rrf_k": self._rrf_k,
        }

        if self._trace_enabled:
            logger.bind(
                bm25_hits=stats["bm25_hits"],
                embedding_hits=stats["embedding_hits"],
                rrf_candidates=stats["rrf_candidates"],
                bm25_top_k=stats["bm25_top_k"],
                embedding_top_k=stats["embedding_top_k"],
                rrf_k=stats["rrf_k"],
            ).info("retrieval_trace")
        return merged, stats
