"""
构建 Neo4j 知识图谱

从 PDF 中抽取实体和关系，写入 Neo4j。
依赖 data/processed/chunks.json 已由 build_index.py 生成。

使用方法：
    python scripts/build_graph.py --chunks-file data/processed/chunks.json
    python scripts/build_graph.py --input data/processed/chunks.json
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.knowledge_graph.entity_extractor import batch_extract_entities
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.relation_builder import build_graph_from_chunks, create_indexes
from src.llm.qwen_client import QwenClient
from src.ops.artifact_store import ArtifactStore
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


def resolve_chunks_file(input_arg: str | None, chunks_file_arg: str | None) -> Path:
    if chunks_file_arg:
        return Path(chunks_file_arg)
    if input_arg:
        input_path = Path(input_arg)
        if input_path.is_dir():
            return input_path / "chunks.json"
        return input_path
    return Path("data/processed/chunks.json")


def main():
    parser = argparse.ArgumentParser(description="构建 Neo4j 知识图谱")
    parser.add_argument("--input", default=None, help="chunks JSON 文件或包含 chunks.json 的目录")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--chunks-file", default=None, help="已有 chunks JSON 路径；优先级高于 --input")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    run = RunRecorder.start(
        pipeline="build_graph",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "chunks_file": args.chunks_file,
            "input_dir": args.input,
        },
    )
    setup_logging(cfg, component="build_graph", run_log_path=run.run_dir / "run.log")
    store = ArtifactStore(artifacts_root)

    qwen_cfg = cfg["models"]["qwen_vl"]
    neo4j_cfg = cfg["neo4j"]

    # ─── Step 1: 加载 chunks ─────────────────────────────────────────
    chunks_file = resolve_chunks_file(args.input, args.chunks_file)
    if not chunks_file.exists():
        message = f"chunks.json 不存在，请先运行 build_index.py: {chunks_file}"
        logger.error(message)
        run.mark_failure(message)
        sys.exit(1)

    try:
        with log_context(run_id=run.run_id, pipeline="build_graph"):
            logger.info(f"Step 1/3: 加载 chunks from {chunks_file}")
            chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
            logger.info(f"加载 {len(chunks)} 个文本块")
            run.update_stats(chunk_count=len(chunks))

            # ─── Step 2: 实体抽取 ────────────────────────────────────
            logger.info("Step 2/3: 实体抽取（使用 Qwen2-VL）")
            qwen = QwenClient(
                model_path=qwen_cfg["model_path"],
                device=qwen_cfg["device"],
            )
            entity_cache_dir = Path(artifacts_root) / "build_graph" / "entity_cache"
            run.note(f"entity_cache_dir={entity_cache_dir}")

            def progress(current, total):
                if current % 50 == 0:
                    logger.info(f"实体抽取进度: {current}/{total} ({100*current//total}%)")

            chunks_with_entities = batch_extract_entities(
                chunks,
                qwen,
                progress_callback=progress,
                cache_dir=entity_cache_dir,
            )

            output = store.stage_json(
                pipeline="build_graph",
                run_id=run.run_id,
                artifact_name="chunks_with_entities.json",
                payload=chunks_with_entities,
                active_path="data/processed/chunks_with_entities.json",
            )
            run.add_output(output)
            logger.info(f"实体抽取结果已保存: {output['artifact_path']}")

            # ─── Step 3: 构建知识图谱 ────────────────────────────────
            logger.info("Step 3/3: 写入 Neo4j 知识图谱")
            with Neo4jClient(
                uri=neo4j_cfg["uri"],
                username=neo4j_cfg["username"],
                password=neo4j_cfg["password"],
                database=neo4j_cfg["database"],
            ) as neo4j:
                create_indexes(neo4j)
                build_graph_from_chunks(chunks_with_entities, neo4j, run_id=run.run_id)
                stats = neo4j.get_stats()
                run.update_stats(**stats)
                logger.info(f"图谱统计: {stats}")

            store.activate(
                pipeline="build_graph",
                run_id=run.run_id,
                outputs=run.data["outputs"],
            )
            logger.info("知识图谱构建完成！")
            run.mark_success()
    except Exception as exc:
        logger.exception("build_graph_failed")
        run.mark_failure(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
