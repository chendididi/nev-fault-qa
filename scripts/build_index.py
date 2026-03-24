"""
构建 Milvus 向量索引 + BM25 持久化

使用方法：
    python scripts/build_index.py --input data/raw/
    python scripts/build_index.py --input data/sample/  # 用示例数据测试
    python scripts/build_index.py --input data/raw/official data/processed/byd_chunks.json --skip-embedding
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data_processing.chunk_contract import validate_chunks
from src.data_processing.circuit_ocr import extract_circuit_descriptions
from src.data_processing.html_manual_parser import parse_html_file
from src.data_processing.pdf_parser import parse_pdf
from src.data_processing.table_extractor import extract_tables
from src.ops.artifact_store import ArtifactStore
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


def load_input_chunks(
    input_path: str | Path | list[str] | list[Path],
    chunk_size: int,
    chunk_overlap: int,
    *,
    table_extraction: bool = True,
    with_circuit_ocr: bool = False,
    circuit_image_ratio: float = 0.3,
    qwen_config: dict | None = None,
    return_metadata: bool = False,
) -> list[dict]:
    """
    从输入路径加载 chunks。

    支持以下输入：
    - 单个 PDF
    - 单个 HTML
    - 单个 chunks JSON
    - 包含 PDF / HTML / chunks JSON 的目录（递归发现）
    - 多个输入路径的组合
    """
    discovery = _discover_inputs(input_path)
    if not discovery.has_supported_inputs():
        roots = ", ".join(discovery.input_roots)
        raise ValueError(f"输入路径 {roots} 中未发现可导入的 PDF、HTML 或 chunks JSON")

    logger.info(
        "输入发现完成：{} 个 PDF，{} 个 HTML，{} 个 chunks JSON，跳过 {} 个非 chunks JSON".format(
            len(discovery.pdf_files),
            len(discovery.html_files),
            len(discovery.chunk_json_files),
            len(discovery.skipped_json_files),
        )
    )
    for skipped in discovery.skipped_json_files[:5]:
        logger.info(f"跳过非 chunks JSON: {skipped['path']} ({skipped['reason']})")

    chunks: list[dict] = []
    metadata = discovery.to_metadata()

    for json_file, json_chunks in discovery.chunk_json_files:
        logger.info(f"加载预构建 chunks JSON: {json_file} ({len(json_chunks)} 个文本块)")
        chunks.extend(json_chunks)

    qwen_client = None
    if with_circuit_ocr and discovery.pdf_files:
        from src.llm.qwen_client import QwenClient

        qwen_client = QwenClient(
            model_path=(qwen_config or {}).get("model_path", "Qwen/Qwen2-VL-7B-Instruct"),
            device=(qwen_config or {}).get("device", "cuda:0"),
            max_new_tokens=(qwen_config or {}).get("max_new_tokens", 2048),
            temperature=(qwen_config or {}).get("temperature", 0.1),
            do_sample=(qwen_config or {}).get("do_sample", False),
        )

    for pdf_file in discovery.pdf_files:
        logger.info(f"解析 PDF 文件: {pdf_file}")
        chunks.extend(parse_pdf(pdf_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

        if table_extraction:
            chunks.extend(_table_records_to_chunks(pdf_file.name, extract_tables(pdf_file)))

        if qwen_client is not None:
            chunks.extend(
                extract_circuit_descriptions(
                    pdf_file,
                    image_ratio_threshold=circuit_image_ratio,
                    qwen_client=qwen_client,
                )
            )

    for html_file in discovery.html_files:
        logger.info(f"解析 HTML 文件: {html_file}")
        chunks.extend(
            parse_html_file(
                html_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    chunks, duplicate_stats = _reconcile_duplicate_chunk_ids(chunks)
    metadata.update(duplicate_stats)

    if return_metadata:
        return chunks, metadata
    return chunks


@dataclass
class InputDiscovery:
    input_roots: list[str] = field(default_factory=list)
    pdf_files: list[Path] = field(default_factory=list)
    html_files: list[Path] = field(default_factory=list)
    chunk_json_files: list[tuple[Path, list[dict]]] = field(default_factory=list)
    skipped_json_files: list[dict[str, str]] = field(default_factory=list)

    def extend(self, other: "InputDiscovery") -> None:
        self.input_roots.extend(other.input_roots)
        self.pdf_files.extend(other.pdf_files)
        self.html_files.extend(other.html_files)
        self.chunk_json_files.extend(other.chunk_json_files)
        self.skipped_json_files.extend(other.skipped_json_files)

    def has_supported_inputs(self) -> bool:
        return bool(self.pdf_files or self.html_files or self.chunk_json_files)

    def to_metadata(self) -> dict:
        metadata: dict[str, object] = {
            "input_roots": self.input_roots,
            "pdf_count": len(self.pdf_files),
            "html_count": len(self.html_files),
            "chunk_json_count": len(self.chunk_json_files),
            "skipped_json_count": len(self.skipped_json_files),
        }
        if self.skipped_json_files:
            metadata["skipped_json_samples"] = self.skipped_json_files[:5]
        return metadata


def _normalize_input_paths(input_path: str | Path | list[str] | list[Path]) -> list[Path]:
    if isinstance(input_path, (str, Path)):
        raw_paths = [input_path]
    else:
        raw_paths = list(input_path)
    normalized = [Path(path) for path in raw_paths]
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in normalized:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _try_load_chunk_json(json_path: Path) -> tuple[list[dict] | None, str | None]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"JSON 解析失败: {exc}"

    if not isinstance(payload, list):
        return None, "顶层不是 JSON 数组"
    if not payload:
        return None, "chunks 列表为空"

    try:
        validate_chunks(payload)
    except ValueError as exc:
        return None, str(exc)
    return payload, None


def _discover_directory_inputs(input_dir: Path) -> InputDiscovery:
    discovery = InputDiscovery(input_roots=[str(input_dir)])
    discovery.pdf_files = sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())
    discovery.html_files = sorted(path for path in input_dir.rglob("*.html") if path.is_file())

    for json_file in sorted(path for path in input_dir.rglob("*.json") if path.is_file()):
        if json_file.name.endswith(".meta.json"):
            discovery.skipped_json_files.append(
                {"path": str(json_file), "reason": "元数据文件，不是 chunks JSON"}
            )
            continue

        chunks, reason = _try_load_chunk_json(json_file)
        if chunks is None:
            discovery.skipped_json_files.append({"path": str(json_file), "reason": reason or "未知原因"})
            continue
        discovery.chunk_json_files.append((json_file, chunks))

    return discovery


def _discover_inputs(input_path: str | Path | list[str] | list[Path]) -> InputDiscovery:
    discovery = InputDiscovery()
    for path in _normalize_input_paths(input_path):
        if not path.exists():
            raise FileNotFoundError(f"输入路径不存在: {path}")

        if path.is_file() and path.suffix.lower() == ".pdf":
            logger.info(f"检测到单个 PDF 文件: {path}")
            discovery.extend(InputDiscovery(input_roots=[str(path)], pdf_files=[path]))
            continue

        if path.is_file() and path.suffix.lower() == ".html":
            logger.info(f"检测到单个 HTML 手册页面: {path}")
            discovery.extend(InputDiscovery(input_roots=[str(path)], html_files=[path]))
            continue

        if path.is_file() and path.suffix.lower() == ".json":
            logger.info(f"检测到预构建 chunks JSON: {path}")
            chunks, reason = _try_load_chunk_json(path)
            if chunks is None:
                raise ValueError(f"预构建 chunks JSON 无效: {path} ({reason})")
            discovery.extend(InputDiscovery(input_roots=[str(path)], chunk_json_files=[(path, chunks)]))
            continue

        if path.is_dir():
            discovery.extend(_discover_directory_inputs(path))
            continue

        raise ValueError(f"暂不支持的输入路径: {path}")

    return discovery


def _chunk_signature(chunk: dict) -> tuple:
    return (
        chunk.get("text"),
        chunk.get("source"),
        chunk.get("page"),
        chunk.get("chapter"),
    )


def _make_conflict_chunk_id(chunk: dict, salt: int) -> str:
    raw = json.dumps(
        {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "source": chunk.get("source"),
            "page": chunk.get("page"),
            "chapter": chunk.get("chapter"),
            "salt": salt,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _reconcile_duplicate_chunk_ids(chunks: list[dict]) -> tuple[list[dict], dict[str, object]]:
    seen: dict[str, dict] = {}
    resolved: list[dict] = []
    skipped_duplicate_count = 0
    remapped_chunk_id_count = 0
    remapped_samples: list[dict[str, str]] = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        existing = seen.get(chunk_id)
        if existing is None:
            seen[chunk_id] = chunk
            resolved.append(chunk)
            continue

        if _chunk_signature(existing) == _chunk_signature(chunk):
            skipped_duplicate_count += 1
            continue

        replacement = dict(chunk)
        salt = 1
        new_chunk_id = _make_conflict_chunk_id(replacement, salt)
        while new_chunk_id in seen:
            salt += 1
            new_chunk_id = _make_conflict_chunk_id(replacement, salt)

        replacement["chunk_id"] = new_chunk_id
        seen[new_chunk_id] = replacement
        resolved.append(replacement)
        remapped_chunk_id_count += 1
        if len(remapped_samples) < 5:
            remapped_samples.append(
                {
                    "original_chunk_id": chunk_id,
                    "new_chunk_id": new_chunk_id,
                    "source": str(chunk.get("source", "")),
                }
            )

    if skipped_duplicate_count:
        logger.warning(f"发现 {skipped_duplicate_count} 个完全重复的 chunk，已跳过重复项")
    if remapped_chunk_id_count:
        logger.warning(
            f"发现 {remapped_chunk_id_count} 个 chunk_id 冲突，已重写 chunk_id，示例: {remapped_samples}"
        )

    metadata: dict[str, object] = {
        "skipped_duplicate_chunk_count": skipped_duplicate_count,
        "remapped_chunk_id_count": remapped_chunk_id_count,
    }
    if remapped_samples:
        metadata["remapped_chunk_id_samples"] = remapped_samples
    return resolved, metadata


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
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="输入路径（支持 PDF/HTML/chunks JSON 文件，或包含这些文件的目录；可一次传多个）",
    )
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
            "input_paths": args.input,
            "skip_embedding": args.skip_embedding,
            "with_circuit_ocr": args.with_circuit_ocr,
            "table_extraction": not args.disable_table_extraction,
        },
    )
    setup_logging(cfg, component="build_index", run_log_path=run.run_dir / "run.log")
    store = ArtifactStore(artifacts_root)

    proc_cfg = cfg["data_processing"]
    emb_cfg = cfg["models"]["embedding"]
    milvus_cfg = cfg["milvus"]

    # ─── Step 1: PDF 解析 ────────────────────────────────────────────
    try:
        with log_context(run_id=run.run_id, pipeline="build_index"):
            logger.info(f"Step 1/3: 从 {args.input} 加载 chunks")
            chunks, input_metadata = load_input_chunks(
                args.input,
                chunk_size=proc_cfg["chunk_size"],
                chunk_overlap=proc_cfg["chunk_overlap"],
                table_extraction=proc_cfg.get("table_extraction", True) and not args.disable_table_extraction,
                with_circuit_ocr=bool(args.with_circuit_ocr or proc_cfg.get("circuit_ocr", False)),
                circuit_image_ratio=proc_cfg.get("circuit_image_ratio", 0.3),
                qwen_config=cfg["models"]["qwen_vl"],
                return_metadata=True,
            )
            if not chunks:
                raise ValueError(f"未加载到任何文本块，请检查输入路径: {args.input}")

            validate_chunks(chunks)
            run.update_stats(**input_metadata)
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
