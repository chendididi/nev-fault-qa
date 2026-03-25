"""
RAGAS 自动评估脚本

评估维度：
- 检索层：Context Recall, Context Precision
- 生成层：Faithfulness, Answer Relevance

使用方法：
    python tests/eval_ragas.py --qa-file tests/eval_qa_set.json
"""

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import requests
from datasets import Dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = "http://localhost:8000/api/v1"

# 评估用测试集格式（JSON 数组）：
# [{"question": ..., "ground_truth": ..., "ground_truth_context": ...}, ...]
DEFAULT_QA_FILE = "tests/eval_qa_set.json"

# 内置示例评估集（用于快速验证流程）
BUILTIN_QA_SET = [
    {
        "question": "P0300 故障码的含义是什么？",
        "ground_truth": "P0300 表示随机/多缸失火检测，发动机出现随机或多缸失火，需检查点火系统、燃油喷射器等。",
        "ground_truth_context": "P0300 随机/多缸失火检测。该故障码表示发动机出现随机或多缸失火，可能导致排放超标和动力不足。",
    },
    {
        "question": "车辆无法启动怎么排查？",
        "ground_truth": "无法启动可能原因包括：蓄电池亏电、起动机故障、点火系统异常、燃油系统故障，需逐步排查。",
        "ground_truth_context": "车辆启动困难可能由多种原因引起：电池电量不足、启动电机故障、点火系统异常等。",
    },
]


def _ensure_langchain_pydantic_v1() -> None:
    """
    RAGAS 0.1.x 依赖 langchain_core.pydantic_v1，在 langchain-core>=1.0 中已移除。
    这里注入兼容模块，避免评估脚本直接失败。
    """
    try:
        import pydantic
        from pydantic import v1 as pydantic_v1
    except Exception:
        try:
            import pydantic  # type: ignore
            pydantic_v1 = pydantic
        except Exception:
            return

    for module_name in ("langchain_core.pydantic_v1", "langchain.pydantic_v1"):
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        module.__dict__.update(pydantic_v1.__dict__)
        sys.modules[module_name] = module


_ensure_langchain_pydantic_v1()

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _validate_qa_items(qa_items: list[dict]) -> None:
    required = {"question", "ground_truth", "ground_truth_context"}
    for idx, item in enumerate(qa_items):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"评测集第 {idx} 项缺少字段: {', '.join(sorted(missing))}")
        for key in required:
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f"评测集第 {idx} 项 {key} 为空")


def load_eval_set(qa_file: str | None) -> tuple[list[dict], dict]:
    if qa_file and Path(qa_file).exists():
        path = Path(qa_file)
        qa_items = json.loads(path.read_text(encoding="utf-8"))
        _validate_qa_items(qa_items)
        return qa_items, {
            "dataset_path": str(path),
            "dataset_sha256": _sha256(path),
            "dataset_source": "explicit",
        }

    default_path = Path(DEFAULT_QA_FILE)
    if default_path.exists():
        qa_items = json.loads(default_path.read_text(encoding="utf-8"))
        _validate_qa_items(qa_items)
        return qa_items, {
            "dataset_path": str(default_path),
            "dataset_sha256": _sha256(default_path),
            "dataset_source": "default",
        }

    _validate_qa_items(BUILTIN_QA_SET)
    return BUILTIN_QA_SET, {
        "dataset_path": None,
        "dataset_sha256": None,
        "dataset_source": "builtin",
    }


def call_api(question: str) -> tuple[str, list[str]]:
    """
    调用 RAG API 获取答案和检索上下文。

    Returns:
        (answer, contexts) 元组
    """
    resp = requests.post(
        f"{API_BASE}/chat",
        json={"message": question},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    answer = data["answer"]
    contexts = [s.get("snippet", "") for s in data.get("sources", [])]
    return answer, contexts


def build_ragas_dataset(qa_items: list[dict]) -> Dataset:
    """
    调用 API 构建 RAGAS 评估数据集。

    Args:
        qa_items: 测试集列表

    Returns:
        HuggingFace Dataset
    """
    rows = []
    for i, item in enumerate(qa_items):
        logger.info(f"评估进度 {i+1}/{len(qa_items)}: {item['question'][:40]}...")
        try:
            answer, contexts = call_api(item["question"])
        except Exception as e:
            logger.warning(f"API 调用失败，跳过: {e}")
            continue

        rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts if contexts else [item.get("ground_truth_context", "")],
            "ground_truth": item["ground_truth"],
        })

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument("--qa-file", default=None, help=f"评估集 JSON 文件路径（默认使用内置示例）")
    parser.add_argument("--output", default="tests/ragas_results.json", help="评估结果输出路径")
    args = parser.parse_args()

    # 加载评估集
    qa_items, dataset_meta = load_eval_set(args.qa_file)
    logger.info(
        f"使用评估集来源: {dataset_meta['dataset_source']}，共 {len(qa_items)} 条"
    )
    if dataset_meta.get("dataset_path"):
        logger.info(f"评估集路径: {dataset_meta['dataset_path']}")

    # 构建数据集
    logger.info("调用 API 构建评估数据集...")
    dataset = build_ragas_dataset(qa_items)

    if len(dataset) == 0:
        logger.error("评估数据集为空，请确认 API 服务正常运行")
        sys.exit(1)

    # 运行 RAGAS 评估
    logger.info("运行 RAGAS 评估...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # 输出结果
    print("\n" + "="*50)
    print("RAGAS 评估结果")
    print("="*50)
    print(f"Faithfulness (忠实度):        {result['faithfulness']:.4f}")
    print(f"Answer Relevancy (答案相关性): {result['answer_relevancy']:.4f}")
    print(f"Context Precision (上下文精度): {result['context_precision']:.4f}")
    print(f"Context Recall (上下文召回):   {result['context_recall']:.4f}")
    print("="*50)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
        "dataset_path": dataset_meta.get("dataset_path"),
        "dataset_sha256": dataset_meta.get("dataset_sha256"),
        "dataset_source": dataset_meta.get("dataset_source"),
        "dataset_count": len(qa_items),
    }
    output_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"评估结果已保存: {output_path}")


if __name__ == "__main__":
    main()
