"""
回滚 Milvus 向量索引（重建当前 collection）。

使用方法：
    python scripts/rollback_milvus.py --previous
    python scripts/rollback_milvus.py --run-id build_index-...
    python scripts/rollback_milvus.py --chunks-file /path/to/chunks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from pymilvus import connections, utility

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.ops.artifact_store import ArtifactStore
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder
from src.retrieval.embedding_retriever import EmbeddingRetriever


def _resolve_target_run(store: ArtifactStore, pipeline: str, requested_run_id: str | None, use_previous: bool) -> str:
    if requested_run_id:
        return requested_run_id

    current = store.get_current(pipeline)
    if not current:
        raise ValueError(f"pipeline {pipeline} 当前没有激活版本")

    if not use_previous:
        raise ValueError("请提供 --run-id、--previous 或 --chunks-file")

    runs = store.list_runs(pipeline)
    previous_runs = [run_id for run_id in runs if run_id != current["run_id"]]
    if not previous_runs:
        raise ValueError(f"pipeline {pipeline} 没有可回滚的上一个版本")
    return previous_runs[-1]


def _resolve_chunks_file(
    store: ArtifactStore,
    pipeline: str,
    run_id: str,
) -> Path:
    manifest_path = store.run_dir(pipeline, run_id) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到 run manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for output in manifest.get("outputs", []):
        if output.get("name") == "chunks.json":
            return Path(output["artifact_path"])
    raise ValueError(f"run {run_id} 未找到 chunks.json 输出")


def main() -> None:
    parser = argparse.ArgumentParser(description="回滚 Milvus 向量索引")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--pipeline", default="build_index", help="来源 pipeline（默认 build_index）")
    parser.add_argument("--run-id", default=None, help="目标 run_id")
    parser.add_argument("--previous", action="store_true", help="回滚到前一个版本")
    parser.add_argument("--chunks-file", default=None, help="直接指定 chunks.json 路径")
    parser.add_argument("--batch-size", type=int, default=256, help="Milvus 批量插入大小")
    parser.add_argument("--skip-drop", action="store_true", help="不删除现有 collection（可能产生重复）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, component="rollback_milvus")
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    store = ArtifactStore(artifacts_root)

    emb_cfg = cfg["models"]["embedding"]
    milvus_cfg = cfg["milvus"]
    collection_name = milvus_cfg["collection_name"]

    run = RunRecorder.start(
        pipeline="rollback_milvus",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "pipeline": args.pipeline,
            "run_id": args.run_id,
            "previous": args.previous,
            "chunks_file": args.chunks_file,
            "batch_size": args.batch_size,
            "skip_drop": args.skip_drop,
        },
    )

    try:
        if args.chunks_file:
            chunks_file = Path(args.chunks_file)
            source_run_id = None
        else:
            target_run_id = _resolve_target_run(store, args.pipeline, args.run_id, args.previous)
            chunks_file = _resolve_chunks_file(store, args.pipeline, target_run_id)
            source_run_id = target_run_id

        if not chunks_file.exists():
            raise FileNotFoundError(f"chunks 文件不存在: {chunks_file}")

        with log_context(run_id=run.run_id, pipeline="rollback_milvus"):
            logger.info(f"准备回滚 Milvus collection: {collection_name}")
            if source_run_id:
                logger.info(f"使用来源 run: {source_run_id}")
            logger.info(f"使用 chunks 文件: {chunks_file}")

            chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
            if not chunks:
                raise ValueError(f"chunks 文件为空: {chunks_file}")

            connections.connect(host=milvus_cfg["host"], port=milvus_cfg["port"])
            dropped = False
            if not args.skip_drop and utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                dropped = True
                logger.info(f"已删除旧 collection: {collection_name}")

            retriever = EmbeddingRetriever(
                model_name=emb_cfg["model_name"],
                milvus_host=milvus_cfg["host"],
                milvus_port=milvus_cfg["port"],
                collection_name=collection_name,
                dim=milvus_cfg["dim"],
                device=emb_cfg["device"],
            )
            retriever.insert(chunks, batch_size=args.batch_size)

            run.update_stats(
                chunk_count=len(chunks),
                collection_name=collection_name,
                dropped_collection=dropped,
                source_run_id=source_run_id,
                milvus_ready=retriever.ping(),
            )
            run.mark_success()
            logger.info(f"Milvus 回滚完成，共导入 {len(chunks)} 条")
    except Exception as exc:
        logger.exception("rollback_milvus_failed")
        run.mark_failure(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
