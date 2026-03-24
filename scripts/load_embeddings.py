"""
从现有 chunks JSON 加载向量到 Milvus。

适用于：
- 示例数据快速启动
- 已有 chunks.json，后补向量索引

使用方法：
    python scripts/load_embeddings.py --input data/sample/sample_chunks.json
    python scripts/load_embeddings.py --input data/processed/chunks.json
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.retrieval.embedding_retriever import EmbeddingRetriever


def main():
    parser = argparse.ArgumentParser(description="从 chunks JSON 加载向量索引")
    parser.add_argument("--input", required=True, help="chunks JSON 文件路径")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--batch-size", type=int, default=256, help="Milvus 批量插入大小")
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        sys.exit(1)

    cfg = load_config(args.config)
    emb_cfg = cfg["models"]["embedding"]
    milvus_cfg = cfg["milvus"]

    chunks = json.loads(input_file.read_text(encoding="utf-8"))
    if not chunks:
        logger.error(f"输入文件为空: {input_file}")
        sys.exit(1)

    retriever = EmbeddingRetriever(
        model_name=emb_cfg["model_name"],
        milvus_host=milvus_cfg["host"],
        milvus_port=milvus_cfg["port"],
        collection_name=milvus_cfg["collection_name"],
        dim=milvus_cfg["dim"],
        device=emb_cfg["device"],
    )
    retriever.insert(chunks, batch_size=args.batch_size)
    logger.info(f"向量索引导入完成，共 {len(chunks)} 条: {input_file}")


if __name__ == "__main__":
    main()
