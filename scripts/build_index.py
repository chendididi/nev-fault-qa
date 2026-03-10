"""
构建 Milvus 向量索引 + BM25 持久化

使用方法：
    python scripts/build_index.py --input data/raw/
    python scripts/build_index.py --input data/sample/  # 用示例数据测试
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.pdf_parser import parse_pdf_dir
from src.data_processing.table_extractor import extract_tables
from src.retrieval.embedding_retriever import EmbeddingRetriever


def main():
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--input", required=True, help="输入 PDF 目录路径")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--skip-embedding", action="store_true", help="跳过 Milvus 索引（仅保存 chunks JSON）")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    proc_cfg = cfg["data_processing"]
    emb_cfg = cfg["models"]["embedding"]
    milvus_cfg = cfg["milvus"]

    # ─── Step 1: PDF 解析 ────────────────────────────────────────────
    logger.info(f"Step 1/3: 解析 PDF 目录 {input_dir}")
    chunks = list(parse_pdf_dir(
        input_dir,
        chunk_size=proc_cfg["chunk_size"],
        chunk_overlap=proc_cfg["chunk_overlap"],
    ))
    logger.info(f"共解析 {len(chunks)} 个文本块")

    # ─── Step 2: 保存 chunks JSON（BM25 索引源）────────────────────
    logger.info("Step 2/3: 保存 chunks.json（用于 BM25 索引）")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_file = output_dir / "chunks.json"
    chunks_file.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"chunks.json 已保存: {chunks_file}")

    if args.skip_embedding:
        logger.info("已跳过 Milvus 索引构建")
        return

    # ─── Step 3: 构建 Milvus 向量索引 ────────────────────────────────
    logger.info("Step 3/3: 构建 Milvus 向量索引")
    retriever = EmbeddingRetriever(
        model_name=emb_cfg["model_name"],
        milvus_host=milvus_cfg["host"],
        milvus_port=milvus_cfg["port"],
        collection_name=milvus_cfg["collection_name"],
        dim=milvus_cfg["dim"],
        device=emb_cfg["device"],
    )
    retriever.insert(chunks, batch_size=256)
    logger.info("向量索引构建完成！")


if __name__ == "__main__":
    main()
