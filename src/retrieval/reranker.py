"""
BGE Reranker 重排序

使用 BAAI/bge-reranker-v2-m3 对混合检索的 Top-K 候选进行精排，
将 Top-20 压缩为 Top-5，提高最终送入 LLM 的上下文质量。
"""

from loguru import logger
from sentence_transformers import CrossEncoder


class BGEReranker:
    """基于 Cross-Encoder 的重排序器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda:0",
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: 重排序模型名称（HuggingFace Hub 或本地路径）
            device: 推理设备
            batch_size: 批次大小
        """
        logger.info(f"加载重排序模型: {model_name}")
        self._model = CrossEncoder(model_name, device=device, max_length=512)
        self._batch_size = batch_size

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        对候选 chunk 列表进行重排序。

        Args:
            query: 用户查询
            chunks: 混合检索返回的候选列表
            top_k: 重排后保留数量

        Returns:
            按 rerank_score 降序排列的 top_k 个 chunk
        """
        if not chunks:
            return []

        # Cross-Encoder 输入：(query, passage) 对
        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)

        # 附加分数并排序
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(f"重排序完成，Top-{top_k} 最高分: {reranked[0]['rerank_score']:.4f}")

        return reranked[:top_k]
