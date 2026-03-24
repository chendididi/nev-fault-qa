"""
构建 Milvus 向量索引 + BM25 持久化

使用方法：
    python scripts/build_index.py --input data/raw/
    python scripts/build_index.py --input data/sample/  # 用示例数据测试
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from loguru import logger

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data_processing.chunk_contract import validate_chunks
from src.data_processing.circuit_ocr import extract_circuit_descriptions
from src.data_processing.pdf_parser import parse_pdf, parse_pdf_dir
from src.data_processing.table_extractor import extract_tables
from src.ops.artifact_store import ArtifactStore
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


def load_input_chunks(
    input_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    *,
    table_extraction: bool = True,
    with_circuit_ocr: bool = False,
    circuit_image_ratio: float = 0.3,
    qwen_config: dict | None = None,
) -> list[dict]:
    """
    从输入路径加载 chunks。

    支持以下输入：
    - 单个 PDF
    - 包含多个 PDF 的目录
    - 单个 chunks JSON
    - 目录中的 `sample_chunks.json` 或 `chunks.json`
    """
    input_path = Path(input_path)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        logger.info(f"检测到单个 PDF 文件: {input_path}")
        chunks = parse_pdf(input_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if table_extraction:
            chunks.extend(_table_records_to_chunks(input_path.name, extract_tables(input_path)))

        if with_circuit_ocr:
            from src.llm.qwen_client import QwenClient

            qwen_client = QwenClient(
                model_path=(qwen_config or {}).get("model_path", "Qwen/Qwen2-VL-7B-Instruct"),
                device=(qwen_config or {}).get("device", "cuda:0"),
                max_new_tokens=(qwen_config or {}).get("max_new_tokens", 2048),
                temperature=(qwen_config or {}).get("temperature", 0.1),
                do_sample=(qwen_config or {}).get("do_sample", False),
            )
            chunks.extend(
                extract_circuit_descriptions(
                    input_path,
                    image_ratio_threshold=circuit_image_ratio,
                    qwen_client=qwen_client,
                )
            )

        return chunks

    if input_path.is_file() and input_path.suffix.lower() == ".json":
        logger.info(f"检测到预构建 chunks JSON: {input_path}")
        chunks = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(chunks, list):
            raise ValueError(f"样例数据格式错误，预期为 JSON 数组: {input_path}")
        return chunks

    pdfs = list(input_path.glob("*.pdf")) if input_path.is_dir() else []
    if pdfs:
        logger.info(f"检测到 {len(pdfs)} 个 PDF 文件，走 PDF 解析链路")
        chunks = list(parse_pdf_dir(input_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

        if table_extraction:
            for pdf in pdfs:
                table_records = extract_tables(pdf)
                chunks.extend(_table_records_to_chunks(pdf.name, table_records))

        if with_circuit_ocr:
            from src.llm.qwen_client import QwenClient

            qwen_client = QwenClient(
                model_path=(qwen_config or {}).get("model_path", "Qwen/Qwen2-VL-7B-Instruct"),
                device=(qwen_config or {}).get("device", "cuda:0"),
                max_new_tokens=(qwen_config or {}).get("max_new_tokens", 2048),
                temperature=(qwen_config or {}).get("temperature", 0.1),
                do_sample=(qwen_config or {}).get("do_sample", False),
            )
            for pdf in pdfs:
                chunks.extend(
                    extract_circuit_descriptions(
                        pdf,
                        image_ratio_threshold=circuit_image_ratio,
                        qwen_client=qwen_client,
                    )
                )

        return chunks

    if input_path.is_dir():
        for candidate_name in ("sample_chunks.json", "chunks.json"):
            sample_file = input_path / candidate_name
            if sample_file.exists():
                logger.info(f"检测到预构建样例数据: {sample_file}")
                chunks = json.loads(sample_file.read_text(encoding="utf-8"))
                if not isinstance(chunks, list):
                    raise ValueError(f"样例数据格式错误，预期为 JSON 数组: {sample_file}")
                return chunks

    raise ValueError(f"输入路径 {input_path} 中未发现可导入的 PDF 或 JSON")


def _table_records_to_chunks(source_name: str, records: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for index, record in enumerate(records):
        text = "\n".join(
            [
                f"故障码: {record['fault_code']}",
                f"描述: {record.get('description', '').strip()}",
                f"可能原因: {record.get('cause', '').strip()}",
            ]
        ).strip()
        raw = f"{source_name}_{record['page']}_{record['fault_code']}_{index}_table"
        chunk_id = hashlib.md5(raw.encode()).hexdigest()[:12]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": source_name,
                "page": int(record["page"]),
                "chapter": "fault_code_table",
            }
        )
    return chunks


table_records_to_chunks = _table_records_to_chunks


def main():
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--input", required=True, help="输入路径（支持 PDF 文件、PDF 目录或 chunks JSON）")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--skip-embedding", action="store_true", help="跳过 Milvus 索引（仅保存 chunks JSON）")
    parser.add_argument("--with-circuit-ocr", action="store_true", help="启用电路图 OCR（需要本地 Qwen2-VL）")
    parser.add_argument("--disable-table-extraction", action="store_true", help="禁用 PDF 表格提取")
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
            "with_circuit_ocr": args.with_circuit_ocr,
            "table_extraction": not args.disable_table_extraction,
        },
    )
    setup_logging(cfg, component="build_index", run_log_path=run.run_dir / "run.log")
    store = ArtifactStore(artifacts_root)

    input_dir = Path(args.input)
    if not input_dir.exists():
        message = f"输入路径不存在: {input_dir}"
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
                table_extraction=proc_cfg.get("table_extraction", True) and not args.disable_table_extraction,
                with_circuit_ocr=bool(args.with_circuit_ocr or proc_cfg.get("circuit_ocr", False)),
                circuit_image_ratio=proc_cfg.get("circuit_image_ratio", 0.3),
                qwen_config=cfg["models"]["qwen_vl"],
            )
            if not chunks:
                raise ValueError(f"未加载到任何文本块，请检查输入目录: {input_dir}")

            validate_chunks(chunks)
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
