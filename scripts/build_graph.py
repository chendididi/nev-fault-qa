"""
构建 Neo4j 知识图谱

从 PDF 中抽取实体和关系，写入 Neo4j。
依赖 data/processed/chunks.json 已由 build_index.py 生成。

使用方法：
    python scripts/build_graph.py --input data/raw/
    python scripts/build_graph.py --input data/sample/
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.entity_extractor import batch_extract_entities
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.relation_builder import build_graph_from_chunks, create_indexes
from src.llm.qwen_client import QwenClient


def main():
    parser = argparse.ArgumentParser(description="构建 Neo4j 知识图谱")
    parser.add_argument("--input", default=None, help="输入 PDF 目录（可选，使用已有 chunks.json 则不需要）")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--chunks-file", default="data/processed/chunks.json", help="已有 chunks JSON 路径")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qwen_cfg = cfg["models"]["qwen_vl"]
    neo4j_cfg = cfg["neo4j"]

    # ─── Step 1: 加载 chunks ─────────────────────────────────────────
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        logger.error(f"chunks.json 不存在，请先运行 build_index.py: {chunks_file}")
        sys.exit(1)

    logger.info(f"Step 1/3: 加载 chunks from {chunks_file}")
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    logger.info(f"加载 {len(chunks)} 个文本块")

    # ─── Step 2: 实体抽取 ────────────────────────────────────────────
    logger.info("Step 2/3: 实体抽取（使用 Qwen2-VL）")
    qwen = QwenClient(
        model_path=qwen_cfg["model_path"],
        device=qwen_cfg["device"],
    )

    def progress(current, total):
        if current % 50 == 0:
            logger.info(f"实体抽取进度: {current}/{total} ({100*current//total}%)")

    chunks_with_entities = batch_extract_entities(chunks, qwen, progress_callback=progress)

    # 保存抽取结果（用于调试）
    enriched_file = Path("data/processed/chunks_with_entities.json")
    enriched_file.write_text(
        json.dumps(chunks_with_entities, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"实体抽取结果已保存: {enriched_file}")

    # ─── Step 3: 构建知识图谱 ─────────────────────────────────────────
    logger.info("Step 3/3: 写入 Neo4j 知识图谱")
    with Neo4jClient(
        uri=neo4j_cfg["uri"],
        username=neo4j_cfg["username"],
        password=neo4j_cfg["password"],
        database=neo4j_cfg["database"],
    ) as neo4j:
        # 创建索引和约束
        create_indexes(neo4j)
        # 批量写入
        build_graph_from_chunks(chunks_with_entities, neo4j)
        # 打印统计
        stats = neo4j.get_stats()
        logger.info(f"图谱统计: {stats}")

    logger.info("知识图谱构建完成！")


if __name__ == "__main__":
    main()
