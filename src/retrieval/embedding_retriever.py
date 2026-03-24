"""
向量语义检索器

使用 BAAI/bge-base-zh-v1.5 对文本编码，存储到 Milvus 2.4。
适合故障现象的模糊语义匹配（如"启动困难"→相关故障片段）。
"""

from typing import Any

import numpy as np
from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    """基于 Milvus + BGE 的语义检索器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        collection_name: str = "nev_chunks",
        dim: int = 768,
        device: str = "cuda:0",
    ):
        """
        Args:
            model_name: 嵌入模型名称（HuggingFace Hub 或本地路径）
            milvus_host: Milvus 地址
            milvus_port: Milvus 端口
            collection_name: Milvus Collection 名称
            dim: 向量维度
            device: 推理设备
        """
        self._model_name = model_name
        self._collection_name = collection_name
        self._dim = dim
        self._model: SentenceTransformer | None = None
        self._collection: Collection | None = None

        # 连接 Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        logger.info(f"Milvus 连接成功: {milvus_host}:{milvus_port}")

        # 加载嵌入模型
        logger.info(f"加载嵌入模型: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)

        # 初始化 Collection
        self._init_collection()

    def _init_collection(self) -> None:
        """创建或加载 Milvus Collection。"""
        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            self._collection.load()
            logger.info(f"加载已有 Collection: {self._collection_name}")
            return

        # 定义 Schema
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
        ]
        schema = CollectionSchema(fields, description="NEV fault manual chunks")
        self._collection = Collection(self._collection_name, schema)

        # 创建 IVF_FLAT 索引
        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        self._collection.create_index("embedding", index_params)
        self._collection.load()
        logger.info(f"创建新 Collection: {self._collection_name}")

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        对文本列表编码为向量。BGE 模型需要添加查询前缀。

        Args:
            texts: 文本列表
            normalize: 是否归一化（与 IP 内积等价于余弦相似度）

        Returns:
            shape (N, dim) 的 numpy array
        """
        # BGE 查询前缀（官方建议）
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=64,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def insert(self, chunks: list[dict], batch_size: int = 256) -> None:
        """
        将文本块批量插入 Milvus。

        Args:
            chunks: pdf_parser 输出的 chunk 列表
            batch_size: 每批次插入数量
        """
        logger.info(f"向 Milvus 插入 {len(chunks)} 个文本块")
        texts = [c["text"] for c in chunks]
        embeddings = self.encode(texts)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_emb = embeddings[i:i + batch_size]
            data = [
                [c["chunk_id"] for c in batch],
                [c["text"][:4000] for c in batch],          # VARCHAR 截断保护
                [c.get("source", "") for c in batch],
                [c.get("page", 0) for c in batch],
                [c.get("chapter", "")[:500] for c in batch],
                batch_emb.tolist(),
            ]
            self._collection.insert(data)

        self._collection.flush()
        logger.info("Milvus 插入完成")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        语义检索。

        Args:
            query: 查询字符串
            top_k: 返回候选数量

        Returns:
            List of dicts，含 embedding_score 字段
        """
        query_vec = self.encode([query])[0].tolist()
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}

        results = self._collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id", "text", "source", "page", "chapter"],
        )

        hits = []
        for hit in results[0]:
            chunk = {
                "chunk_id": hit.entity.get("chunk_id"),
                "text": hit.entity.get("text"),
                "source": hit.entity.get("source"),
                "page": hit.entity.get("page"),
                "chapter": hit.entity.get("chapter"),
                "embedding_score": float(hit.score),
            }
            hits.append(chunk)

        return hits

    def ping(self) -> bool:
        """轻量连接检查，用于 readiness。"""
        try:
            return self._collection is not None and utility.has_collection(self._collection_name)
        except Exception:
            return False
