"""
BM25 精确检索器

适合故障码精确匹配和关键词检索。
使用 rank_bm25 库，索引在内存中构建（数据量有限，无需持久化）。
若需要持久化，序列化 chunks 列表后 pickle 保存即可。
"""

import re
from pathlib import Path

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """
    中文分词 + 保留故障码（P/B/C/U + 4位数字）不被分割。

    故障码先用占位符保护，分词后还原。
    """
    # 保护故障码
    fault_code_pattern = re.compile(r"[PBCU]\d{4}", re.IGNORECASE)
    codes = fault_code_pattern.findall(text)
    protected = text
    for i, code in enumerate(codes):
        protected = protected.replace(code, f"__FC{i}__")

    # 结巴分词
    tokens = list(jieba.cut(protected))

    # 还原故障码
    result = []
    for token in tokens:
        for i, code in enumerate(codes):
            token = token.replace(f"__FC{i}__", code.upper())
        result.append(token)

    return [t for t in result if t.strip()]


class BM25Retriever:
    """BM25 精确检索器，支持故障码和自然语言查询。"""

    def __init__(self):
        self._corpus: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[dict]) -> None:
        """
        从文本块列表构建 BM25 索引。

        Args:
            chunks: pdf_parser 输出的 chunk 列表，每个 chunk 含 text 字段
        """
        logger.info(f"构建 BM25 索引，共 {len(chunks)} 个文本块")
        self._corpus = chunks
        self._tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 索引构建完成")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25 检索。

        Args:
            query: 查询字符串（故障码或自然语言）
            top_k: 返回候选数量

        Returns:
            List of dicts，每个 dict 为原始 chunk 加上 bm25_score 字段
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 build()")

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # 取 top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            chunk = dict(self._corpus[idx])
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        return results

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None
