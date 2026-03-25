"""
回滚 Neo4j 知识图谱（从 chunks_with_entities.json 重建）。

使用方法：
    python scripts/rollback_graph.py --previous
    python scripts/rollback_graph.py --run-id build_graph-...
    python scripts/rollback_graph.py --entities-file /path/to/chunks_with_entities.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.relation_builder import build_graph_from_chunks, create_indexes
from src.ops.artifact_store import ArtifactStore
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


def _resolve_target_run(store: ArtifactStore, pipeline: str, requested_run_id: str | None, use_previous: bool) -> str:
    if requested_run_id:
        return requested_run_id

    current = store.get_current(pipeline)
    if not current:
        raise ValueError(f"pipeline {pipeline} 当前没有激活版本")

    if not use_previous:
        raise ValueError("请提供 --run-id、--previous 或 --entities-file")

    runs = store.list_runs(pipeline)
    previous_runs = [run_id for run_id in runs if run_id != current["run_id"]]
    if not previous_runs:
        raise ValueError(f"pipeline {pipeline} 没有可回滚的上一个版本")
    return previous_runs[-1]


def _resolve_entities_file(store: ArtifactStore, pipeline: str, run_id: str) -> Path:
    manifest_path = store.run_dir(pipeline, run_id) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到 run manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for output in manifest.get("outputs", []):
        if output.get("name") == "chunks_with_entities.json":
            return Path(output["artifact_path"])
    raise ValueError(f"run {run_id} 未找到 chunks_with_entities.json 输出")


def main() -> None:
    parser = argparse.ArgumentParser(description="回滚 Neo4j 知识图谱")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--pipeline", default="build_graph", help="来源 pipeline（默认 build_graph）")
    parser.add_argument("--run-id", default=None, help="目标 run_id")
    parser.add_argument("--previous", action="store_true", help="回滚到前一个版本")
    parser.add_argument("--entities-file", default=None, help="直接指定 chunks_with_entities.json")
    parser.add_argument("--skip-clear", action="store_true", help="不清空 Neo4j（可能产生重复）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, component="rollback_graph")
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    store = ArtifactStore(artifacts_root)
    neo4j_cfg = cfg["neo4j"]

    run = RunRecorder.start(
        pipeline="rollback_graph",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "pipeline": args.pipeline,
            "run_id": args.run_id,
            "previous": args.previous,
            "entities_file": args.entities_file,
            "skip_clear": args.skip_clear,
        },
    )

    try:
        if args.entities_file:
            entities_file = Path(args.entities_file)
            source_run_id = None
        else:
            target_run_id = _resolve_target_run(store, args.pipeline, args.run_id, args.previous)
            entities_file = _resolve_entities_file(store, args.pipeline, target_run_id)
            source_run_id = target_run_id

        if not entities_file.exists():
            raise FileNotFoundError(f"entities 文件不存在: {entities_file}")

        with log_context(run_id=run.run_id, pipeline="rollback_graph"):
            logger.info("准备回滚 Neo4j 图谱")
            if source_run_id:
                logger.info(f"使用来源 run: {source_run_id}")
            logger.info(f"使用 entities 文件: {entities_file}")

            chunks_with_entities = json.loads(entities_file.read_text(encoding="utf-8"))
            if not chunks_with_entities:
                raise ValueError(f"entities 文件为空: {entities_file}")

            with Neo4jClient(
                uri=neo4j_cfg["uri"],
                username=neo4j_cfg["username"],
                password=neo4j_cfg["password"],
                database=neo4j_cfg["database"],
            ) as neo4j:
                if not args.skip_clear:
                    neo4j.run("MATCH (n) DETACH DELETE n")
                    logger.info("已清空 Neo4j")

                create_indexes(neo4j)
                build_graph_from_chunks(chunks_with_entities, neo4j, run_id=run.run_id)
                stats = neo4j.get_stats()
                run.update_stats(**stats)
                logger.info(f"图谱统计: {stats}")

            run.update_stats(
                chunk_count=len(chunks_with_entities),
                source_run_id=source_run_id,
                skipped_clear=args.skip_clear,
            )
            run.mark_success()
            logger.info("Neo4j 图谱回滚完成")
    except Exception as exc:
        logger.exception("rollback_graph_failed")
        run.mark_failure(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
