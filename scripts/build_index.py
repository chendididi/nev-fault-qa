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

from loguru import logger

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data_processing.pdf_parser import parse_pdf_dir
from src.ops.artifact_store import ArtifactStore
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


def load_input_chunks(
    input_dir: str | Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    从输入目录加载 chunks。

    支持两类输入：
    - 真实 PDF 目录
    - 内置/预构建的 sample_chunks.json
    """
    input_dir = Path(input_dir)

    pdfs = list(input_dir.glob("*.pdf"))
    if pdfs:
        logger.info(f"检测到 {len(pdfs)} 个 PDF 文件，走 PDF 解析链路")
        return list(parse_pdf_dir(input_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    sample_file = input_dir / "sample_chunks.json"
    if sample_file.exists():
        logger.info(f"检测到预构建样例数据: {sample_file}")
        chunks = json.loads(sample_file.read_text(encoding="utf-8"))
        if not isinstance(chunks, list):
            raise ValueError(f"样例数据格式错误，预期为 JSON 数组: {sample_file}")
        return chunks

    raise ValueError(f"输入目录 {input_dir} 中未发现可导入的 PDF 或 sample_chunks.json")


def main():
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--input", required=True, help="输入目录路径（支持 PDF 目录或 sample_chunks.json 所在目录）")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--skip-embedding", action="store_true", help="跳过 Milvus 索引（仅保存 chunks JSON）")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    run = RunRecorder.start(
        pipeline="build_index",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "input_dir": str(args.input),
            "skip_embedding": args.skip_embedding,
        },
    )
    setup_logging(cfg, component="build_index", run_log_path=run.run_dir / "run.log")
    store = ArtifactStore(artifacts_root)

    input_dir = Path(args.input)
    if not input_dir.exists():
        message = f"输入目录不存在: {input_dir}"
        logger.error(message)
        run.mark_failure(message)
        sys.exit(1)

    proc_cfg = cfg["data_processing"]
    emb_cfg = cfg["models"]["embedding"]
    milvus_cfg = cfg["milvus"]

    # ─── Step 1: PDF 解析 ────────────────────────────────────────────
    try:
        with log_context(run_id=run.run_id, pipeline="build_index"):
            logger.info(f"Step 1/3: 从 {input_dir} 加载 chunks")
            chunks = load_input_chunks(
                input_dir,
                chunk_size=proc_cfg["chunk_size"],
                chunk_overlap=proc_cfg["chunk_overlap"],
            )
            if not chunks:
                raise ValueError(f"未加载到任何文本块，请检查输入目录: {input_dir}")

            logger.info(f"共加载 {len(chunks)} 个文本块")
            run.update_stats(chunk_count=len(chunks))

            # ─── Step 2: 保存 chunks JSON（BM25 索引源）────────────────
            logger.info("Step 2/3: 保存 chunks.json（用于 BM25 索引）")
            output = store.stage_json(
                pipeline="build_index",
                run_id=run.run_id,
                artifact_name="chunks.json",
                payload=chunks,
                active_path="data/processed/chunks.json",
            )
            run.add_output(output)
            logger.info(f"chunks.json 已保存: {output['artifact_path']}")

            if args.skip_embedding:
                run.note("跳过了 Milvus 索引构建")
                run.update_stats(embedding_built=False)
                store.activate(
                    pipeline="build_index",
                    run_id=run.run_id,
                    outputs=run.data["outputs"],
                )
                logger.info("已跳过 Milvus 索引构建")
                run.mark_success()
                return

            # ─── Step 3: 构建 Milvus 向量索引 ────────────────────────
            logger.info("Step 3/3: 构建 Milvus 向量索引")
            from src.retrieval.embedding_retriever import EmbeddingRetriever

            retriever = EmbeddingRetriever(
                model_name=emb_cfg["model_name"],
                milvus_host=milvus_cfg["host"],
                milvus_port=milvus_cfg["port"],
                collection_name=milvus_cfg["collection_name"],
                dim=milvus_cfg["dim"],
                device=emb_cfg["device"],
            )
            retriever.insert(chunks, batch_size=256)
            run.update_stats(embedding_built=True)
            store.activate(
                pipeline="build_index",
                run_id=run.run_id,
                outputs=run.data["outputs"],
            )
            logger.info("向量索引构建完成！")
            run.mark_success()
    except Exception as exc:
        logger.exception("build_index_failed")
        run.mark_failure(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
