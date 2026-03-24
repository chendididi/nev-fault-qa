"""
回滚版本化产物。

使用方法：
    python scripts/rollback_artifact.py --pipeline build_index --previous
    python scripts/rollback_artifact.py --pipeline build_graph --run-id build_graph-...
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.ops.artifact_store import ArtifactStore
from src.ops.logging_setup import log_context, setup_logging


def _resolve_target_run(store: ArtifactStore, pipeline: str, requested_run_id: str | None, use_previous: bool) -> str:
    if requested_run_id:
        return requested_run_id

    current = store.get_current(pipeline)
    if not current:
        raise ValueError(f"pipeline {pipeline} 当前没有激活版本")

    if not use_previous:
        raise ValueError("请提供 --run-id 或 --previous")

    runs = store.list_runs(pipeline)
    previous_runs = [run_id for run_id in runs if run_id != current["run_id"]]
    if not previous_runs:
        raise ValueError(f"pipeline {pipeline} 没有可回滚的上一个版本")
    return previous_runs[-1]


def main():
    parser = argparse.ArgumentParser(description="回滚 build_index/build_graph 的文件产物")
    parser.add_argument("--pipeline", required=True, choices=["build_index", "build_graph"])
    parser.add_argument("--run-id", default=None, help="目标 run_id")
    parser.add_argument("--previous", action="store_true", help="回滚到前一个版本")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, component="rollback_artifact")
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    store = ArtifactStore(artifacts_root)

    try:
        target_run_id = _resolve_target_run(store, args.pipeline, args.run_id, args.previous)
        with log_context(pipeline=args.pipeline, run_id=target_run_id):
            result = store.activate_run(args.pipeline, target_run_id)
            logger.info(f"已回滚到 {args.pipeline}:{target_run_id}")
            for output in result["outputs"]:
                logger.info(
                    f"激活产物 {output['name']}: {output['artifact_path']} -> {output['active_path']}"
                )
    except Exception as exc:
        logger.exception("rollback_artifact_failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
