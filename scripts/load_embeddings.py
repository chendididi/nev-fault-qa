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
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder
from src.retrieval.embedding_retriever import EmbeddingRetriever


def main():
    parser = argparse.ArgumentParser(description="从 chunks JSON 加载向量索引")
    parser.add_argument("--input", required=True, help="chunks JSON 文件路径")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--batch-size", type=int, default=256, help="Milvus 批量插入大小")
    args = parser.parse_args()

    input_file = Path(args.input)
    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    run = RunRecorder.start(
        pipeline="load_embeddings",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "input_file": str(input_file),
            "batch_size": args.batch_size,
        },
    )
    setup_logging(cfg, component="load_embeddings", run_log_path=run.run_dir / "run.log")
    emb_cfg = cfg["models"]["embedding"]
    milvus_cfg = cfg["milvus"]

    if not input_file.exists():
        message = f"输入文件不存在: {input_file}"
        logger.error(message)
        run.mark_failure(message)
        sys.exit(1)

    try:
        with log_context(run_id=run.run_id, pipeline="load_embeddings"):
            chunks = json.loads(input_file.read_text(encoding="utf-8"))
            if not chunks:
                raise ValueError(f"输入文件为空: {input_file}")

            retriever = EmbeddingRetriever(
                model_name=emb_cfg["model_name"],
                milvus_host=milvus_cfg["host"],
                milvus_port=milvus_cfg["port"],
                collection_name=milvus_cfg["collection_name"],
                dim=milvus_cfg["dim"],
                device=emb_cfg["device"],
            )
            retriever.insert(chunks, batch_size=args.batch_size)
            run.update_stats(chunk_count=len(chunks), milvus_ready=retriever.ping())
            run.mark_success()
            logger.info(f"向量索引导入完成，共 {len(chunks)} 条: {input_file}")
    except Exception as exc:
        logger.exception("load_embeddings_failed")
        run.mark_failure(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
