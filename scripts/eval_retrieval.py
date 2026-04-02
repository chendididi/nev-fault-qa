"""
固定检索基线评估（Recall@K / MRR）。

默认使用：
- 评测集: tests/eval_qa_set.json
- chunks: data/processed/chunks.json
- 检索器: BM25

增强能力：
- `--qrels-file`：使用人工标注相关集合（推荐）
- `--thresholds-file`：阈值门禁（用于回归）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data_processing.chunk_contract import validate_chunks
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder
from src.retrieval.bm25_retriever import BM25Retriever


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _question_key(question: str) -> str:
    return " ".join(question.split())


def load_eval_dataset(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("评测集必须是非空数组")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"评测集第 {idx} 项不是对象")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"评测集第 {idx} 项 question 为空")
        normalized.append(item)

    return normalized, {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_size": len(normalized),
    }


def load_chunks(path: str | Path) -> tuple[list[dict], dict[str, Any]]:
    chunks_path = Path(path)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    validate_chunks(chunks)
    return chunks, {
        "chunks_path": str(chunks_path),
        "chunks_sha256": _sha256(chunks_path),
        "chunk_count": len(chunks),
    }


def load_qrels(
    path: str | Path,
    *,
    questions: list[str],
) -> tuple[list[set[str]], dict[str, Any]]:
    qrels_path = Path(path)
    payload = json.loads(qrels_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("qrels 文件必须是非空数组")

    mapping: dict[str, set[str]] = {}
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"qrels 第 {idx} 项不是对象")
        question = item.get("question")
        rel_ids = item.get("relevant_chunk_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"qrels 第 {idx} 项 question 为空")
        if not isinstance(rel_ids, list) or not rel_ids:
            raise ValueError(f"qrels 第 {idx} 项 relevant_chunk_ids 必须是非空数组")
        clean_ids = {cid.strip() for cid in rel_ids if isinstance(cid, str) and cid.strip()}
        if not clean_ids:
            raise ValueError(f"qrels 第 {idx} 项 relevant_chunk_ids 无有效 chunk_id")
        mapping[_question_key(question)] = clean_ids

    relevance_sets: list[set[str]] = []
    missing_questions: list[str] = []
    for question in questions:
        key = _question_key(question)
        if key not in mapping:
            missing_questions.append(question)
            relevance_sets.append(set())
        else:
            relevance_sets.append(set(mapping[key]))

    if missing_questions:
        raise ValueError(
            f"qrels 缺少 {len(missing_questions)} 个问题映射，示例: {missing_questions[:3]}"
        )

    extra_qrels = sorted(set(mapping.keys()) - {_question_key(q) for q in questions})
    return relevance_sets, {
        "label_source": "qrels",
        "qrels_path": str(qrels_path),
        "qrels_sha256": _sha256(qrels_path),
        "qrels_count": len(mapping),
        "extra_qrels_count": len(extra_qrels),
        "extra_qrels_samples": extra_qrels[:5],
    }


def build_silver_relevance(
    qa_items: list[dict[str, Any]],
    *,
    bm25: BM25Retriever,
    silver_top_k: int,
    fallback_chunk_id: str,
) -> tuple[list[set[str]], dict[str, Any]]:
    relevance_sets: list[set[str]] = []
    fallback_count = 0
    for item in qa_items:
        label_query = (
            item.get("ground_truth_context")
            or item.get("ground_truth")
            or item.get("question")
            or ""
        )
        hits = bm25.search(str(label_query), top_k=silver_top_k)
        chunk_ids = {hit["chunk_id"] for hit in hits if hit.get("chunk_id")}
        if not chunk_ids:
            chunk_ids = {fallback_chunk_id}
            fallback_count += 1
        relevance_sets.append(chunk_ids)

    return relevance_sets, {
        "label_source": "silver_bm25",
        "silver_label_strategy": "bm25(label_query)->top_k chunk_ids",
        "silver_top_k": silver_top_k,
        "silver_fallback_count": fallback_count,
    }


def _rank_of_first_hit(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str]) -> int | None:
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return rank
    return None


def _compute_metric_summary(per_question: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    query_count = len(per_question)
    recall_at: dict[str, float] = {}
    for k in k_values:
        hits = sum(item["recall_at"][str(k)] for item in per_question)
        recall_at[str(k)] = hits / query_count if query_count else 0.0

    mrr = (sum(item["mrr"] for item in per_question) / query_count) if query_count else 0.0
    hit_rate = (
        sum(1 for item in per_question if item["rank"] is not None) / query_count
        if query_count
        else 0.0
    )
    return {
        "query_count": query_count,
        "mrr": mrr,
        "hit_rate": hit_rate,
        "recall_at": recall_at,
    }


def evaluate_retriever(
    *,
    qa_items: list[dict[str, Any]],
    relevance_sets: list[set[str]],
    retriever_name: str,
    retrieve_fn: Callable[[str, int], list[dict]],
    retrieve_top_k: int,
    k_values: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_question: list[dict[str, Any]] = []

    for idx, item in enumerate(qa_items):
        question = item["question"]
        relevant_ids = relevance_sets[idx]
        retrieved = retrieve_fn(question, retrieve_top_k)
        retrieved_ids = [chunk.get("chunk_id") for chunk in retrieved if chunk.get("chunk_id")]
        rank = _rank_of_first_hit(retrieved_ids, relevant_ids)
        recall_values = {
            str(k): int(rank is not None and rank <= k)
            for k in k_values
        }
        mrr = (1.0 / rank) if rank else 0.0
        per_question.append(
            {
                "index": idx,
                "question": question,
                "rank": rank,
                "mrr": mrr,
                "recall_at": recall_values,
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved_chunk_ids_top10": retrieved_ids[:10],
            }
        )

    summary = _compute_metric_summary(per_question, k_values)
    summary["retriever"] = retriever_name
    summary["retrieve_top_k"] = retrieve_top_k
    return summary, per_question


def _build_hybrid_retriever(cfg: dict, bm25: BM25Retriever):
    from src.retrieval.embedding_retriever import EmbeddingRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever

    emb_cfg = cfg["models"]["embedding"]
    milvus_cfg = cfg["milvus"]
    retrieval_cfg = cfg["retrieval"]
    embedding = EmbeddingRetriever(
        model_name=emb_cfg["model_name"],
        milvus_host=milvus_cfg["host"],
        milvus_port=milvus_cfg["port"],
        collection_name=milvus_cfg["collection_name"],
        dim=milvus_cfg["dim"],
        device=emb_cfg["device"],
    )
    return HybridRetriever(
        bm25=bm25,
        embedding=embedding,
        rrf_k=retrieval_cfg["rrf_k"],
        bm25_top_k=retrieval_cfg["bm25_top_k"],
        embedding_top_k=retrieval_cfg["embedding_top_k"],
        trace_enabled=bool(cfg.get("logging", {}).get("trace_enabled", False)),
    )


def _parse_k_values(raw_values: list[int]) -> list[int]:
    unique = sorted({value for value in raw_values if value > 0})
    if not unique:
        raise ValueError("--k 至少包含一个正整数")
    return unique


def load_thresholds(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds_path = Path(path)
    payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("阈值文件必须是非空对象")
    return payload, {
        "thresholds_path": str(thresholds_path),
        "thresholds_sha256": _sha256(thresholds_path),
    }


def evaluate_thresholds(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    overall_pass = True

    for retriever_name, rules in thresholds.items():
        metric = metrics.get(retriever_name)
        retriever_checks: list[dict[str, Any]] = []
        retriever_pass = True

        if metric is None:
            retriever_pass = False
            overall_pass = False
            checks[retriever_name] = {
                "passed": False,
                "checks": [],
                "reason": "该检索器未产出指标",
            }
            continue

        if not isinstance(rules, dict):
            raise ValueError(f"阈值配置 {retriever_name} 必须是对象")

        if "mrr_min" in rules:
            actual = float(metric.get("mrr", 0.0))
            minimum = float(rules["mrr_min"])
            passed = actual >= minimum
            retriever_checks.append(
                {"metric": "mrr", "actual": actual, "minimum": minimum, "passed": passed}
            )
            retriever_pass = retriever_pass and passed

        if "hit_rate_min" in rules:
            actual = float(metric.get("hit_rate", 0.0))
            minimum = float(rules["hit_rate_min"])
            passed = actual >= minimum
            retriever_checks.append(
                {"metric": "hit_rate", "actual": actual, "minimum": minimum, "passed": passed}
            )
            retriever_pass = retriever_pass and passed

        recall_rules = rules.get("recall_at_min", {})
        if recall_rules is not None:
            if not isinstance(recall_rules, dict):
                raise ValueError(f"阈值配置 {retriever_name}.recall_at_min 必须是对象")
            recalls = metric.get("recall_at", {})
            for key, minimum_value in recall_rules.items():
                k = str(key)
                actual = float(recalls.get(k, 0.0))
                minimum = float(minimum_value)
                passed = actual >= minimum
                retriever_checks.append(
                    {
                        "metric": f"recall_at_{k}",
                        "actual": actual,
                        "minimum": minimum,
                        "passed": passed,
                    }
                )
                retriever_pass = retriever_pass and passed

        checks[retriever_name] = {
            "passed": retriever_pass,
            "checks": retriever_checks,
        }
        overall_pass = overall_pass and retriever_pass

    return {
        "overall_pass": overall_pass,
        "retrievers": checks,
    }


def run_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")

    k_values = _parse_k_values(args.k)
    run = RunRecorder.start(
        pipeline="eval_retrieval",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "qa_file": args.qa_file,
            "chunks_file": args.chunks_file,
            "qrels_file": args.qrels_file,
            "thresholds_file": args.thresholds_file,
            "fail_on_threshold": args.fail_on_threshold,
            "mode": args.mode,
            "k_values": k_values,
            "retrieve_top_k": args.retrieve_top_k,
            "silver_top_k": args.silver_top_k,
            "output": args.output,
        },
    )
    setup_logging(cfg, component="eval_retrieval", run_log_path=run.run_dir / "run.log")

    try:
        with log_context(run_id=run.run_id, pipeline="eval_retrieval"):
            qa_items, dataset_meta = load_eval_dataset(args.qa_file)
            questions = [item["question"] for item in qa_items]
            chunks, chunk_meta = load_chunks(args.chunks_file)
            if not chunks:
                raise ValueError("chunks 文件为空，无法评估检索")

            bm25 = BM25Retriever()
            bm25.build(chunks)

            if args.qrels_file:
                relevance_sets, label_meta = load_qrels(args.qrels_file, questions=questions)
            else:
                relevance_sets, label_meta = build_silver_relevance(
                    qa_items,
                    bm25=bm25,
                    silver_top_k=args.silver_top_k,
                    fallback_chunk_id=chunks[0]["chunk_id"],
                )

            metrics: dict[str, Any] = {}
            details: dict[str, Any] = {}
            notes: list[str] = []

            bm25_summary, bm25_details = evaluate_retriever(
                qa_items=qa_items,
                relevance_sets=relevance_sets,
                retriever_name="bm25",
                retrieve_fn=bm25.search,
                retrieve_top_k=args.retrieve_top_k,
                k_values=k_values,
            )
            metrics["bm25"] = bm25_summary
            details["bm25"] = bm25_details

            if args.mode in {"auto", "hybrid"}:
                try:
                    hybrid = _build_hybrid_retriever(cfg, bm25)
                    hybrid_summary, hybrid_details = evaluate_retriever(
                        qa_items=qa_items,
                        relevance_sets=relevance_sets,
                        retriever_name="hybrid",
                        retrieve_fn=hybrid.retrieve,
                        retrieve_top_k=args.retrieve_top_k,
                        k_values=k_values,
                    )
                    metrics["hybrid"] = hybrid_summary
                    details["hybrid"] = hybrid_details
                except Exception as exc:
                    if args.mode == "hybrid":
                        raise
                    message = f"hybrid 评估不可用，已自动降级为 bm25: {exc}"
                    notes.append(message)
                    logger.warning(message)

            threshold_meta: dict[str, Any] = {}
            threshold_result: dict[str, Any] | None = None
            if args.thresholds_file:
                thresholds, threshold_meta = load_thresholds(args.thresholds_file)
                threshold_result = evaluate_thresholds(metrics, thresholds)
                if not threshold_result["overall_pass"]:
                    notes.append("阈值校验未通过")

            result_payload = {
                "run_id": run.run_id,
                "dataset": dataset_meta,
                "chunks": chunk_meta,
                "labels": label_meta,
                "mode": args.mode,
                "k_values": k_values,
                "retrieve_top_k": args.retrieve_top_k,
                "metrics": metrics,
                "thresholds": threshold_result,
                "notes": notes,
                "details": details,
            }
            if threshold_meta:
                result_payload["threshold_meta"] = threshold_meta

            result_path = run.run_dir / "retrieval_metrics.json"
            result_path.write_text(
                json.dumps(result_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            run.add_output(
                {
                    "name": "retrieval_metrics.json",
                    "artifact_path": str(result_path),
                    "active_path": None,
                    "activate_on_publish": False,
                }
            )

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                run.add_output(
                    {
                        "name": output_path.name,
                        "artifact_path": str(output_path),
                        "active_path": None,
                        "activate_on_publish": False,
                    }
                )

            run.update_stats(
                dataset_size=dataset_meta["dataset_size"],
                chunk_count=chunk_meta["chunk_count"],
                label_source=label_meta.get("label_source"),
                metrics={
                    name: {
                        "mrr": summary["mrr"],
                        "hit_rate": summary["hit_rate"],
                        "recall_at": summary["recall_at"],
                    }
                    for name, summary in metrics.items()
                },
                threshold_pass=(
                    None if threshold_result is None else bool(threshold_result["overall_pass"])
                ),
            )

            if threshold_result and args.fail_on_threshold and not threshold_result["overall_pass"]:
                run.mark_failure("检索基线阈值校验未通过")
                return 1

            run.mark_success()
            return 0
    except Exception as exc:
        logger.exception("eval_retrieval_failed")
        run.mark_failure(str(exc))
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="计算固定检索基线（Recall@K / MRR）")
    parser.add_argument("--qa-file", default="tests/eval_qa_set.json", help="评测集 JSON 文件（至少含 question）")
    parser.add_argument("--chunks-file", default="data/processed/chunks.json", help="检索语料 chunks JSON")
    parser.add_argument("--qrels-file", default=None, help="人工标注 qrels JSON 文件")
    parser.add_argument("--thresholds-file", default=None, help="阈值配置 JSON 文件")
    parser.add_argument("--fail-on-threshold", action="store_true", help="阈值不达标时返回非 0")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument(
        "--mode",
        default="bm25",
        choices=["bm25", "hybrid", "auto"],
        help="评估模式：bm25=仅 BM25，hybrid=强制混合检索，auto=尝试 hybrid 失败后回退",
    )
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5], help="计算 Recall@K 的 K 列表")
    parser.add_argument("--retrieve-top-k", type=int, default=20, help="每次检索取前 N 个候选")
    parser.add_argument("--silver-top-k", type=int, default=3, help="无 qrels 时 silver label 的 top-k")
    parser.add_argument("--output", default=None, help="可选输出文件路径")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_eval(args))


if __name__ == "__main__":
    main()
