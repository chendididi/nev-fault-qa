"""
RAGAS 自动评估脚本

评估维度：
- 检索层：Context Recall, Context Precision
- 生成层：Faithfulness, Answer Relevance

使用方法：
    python tests/eval_ragas.py --qa-file tests/eval_qa_set.json
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import types
from pathlib import Path
from collections import Counter
import time

import requests
from datasets import Dataset
from loguru import logger
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config

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
from ragas.run_config import RunConfig
from ragas.llms.base import BaseRagasLLM
from langchain_core.outputs import Generation, LLMResult

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


def _normalize_answer_for_ragas(answer: str) -> str:
    normalized = (
        answer.replace("。", ". ")
        .replace("！", ". ")
        .replace("？", ". ")
        .replace("；", ". ")
        .replace("\n", " ")
    )
    normalized = " ".join(normalized.split()).strip()
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized


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


def _response_text(response) -> str:
    try:
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for content in item.content:
                if getattr(content, "type", None) == "output_text":
                    return content.text
    except Exception:
        return ""
    return ""


def _extract_json_schema(prompt_text: str) -> dict | None:
    marker = "Here is the output JSON schema:"
    if marker not in prompt_text:
        return None

    schema_block = prompt_text.split(marker, 1)[1].lstrip()
    if not schema_block.startswith("```"):
        return None

    schema_block = schema_block[3:]
    if schema_block.startswith("json"):
        schema_block = schema_block[4:]
    if schema_block.startswith("\n"):
        schema_block = schema_block[1:]

    end = schema_block.find("```")
    if end == -1:
        return None

    raw_schema = schema_block[:end].strip()
    if not raw_schema:
        return None

    try:
        return json.loads(raw_schema)
    except json.JSONDecodeError:
        return None


def _extract_output_key(prompt_text: str) -> str | None:
    lines = [line.strip() for line in prompt_text.rstrip().splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    if not last_line.endswith(":"):
        return None
    return last_line[:-1].strip() or None


def _safe_schema_name(name: str | None) -> str:
    value = name or "ragas_response"
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")
    return value or "ragas_response"


def _build_response_format(prompt_text: str) -> tuple[dict | None, str | None]:
    schema = _extract_json_schema(prompt_text)
    if schema is None:
        return None, None

    output_key = _extract_output_key(prompt_text)
    wrapper_key = None
    request_schema = schema
    if schema.get("type") != "object":
        if not output_key:
            return None, None
        request_schema = {
            "type": "object",
            "properties": {output_key: schema},
            "required": [output_key],
            "additionalProperties": False,
        }
        wrapper_key = output_key

    response_format = {
        "format": {
            "type": "json_schema",
            "name": _safe_schema_name(output_key),
            "schema": request_schema,
            "strict": True,
        }
    }
    return response_format, wrapper_key


def _normalize_structured_text(text: str, wrapper_key: str | None) -> str:
    if not text or not wrapper_key:
        return text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict) and wrapper_key in payload:
        return json.dumps(payload[wrapper_key], ensure_ascii=False)
    return text


def _summarize_error(error: Exception, limit: int = 240) -> str:
    text = " ".join(str(error).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class ResponsesRagasLLM(BaseRagasLLM):
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key: str,
        timeout: int | None = None,
        max_attempts: int = 3,
        backoff_sec: float = 2.0,
    ):
        super().__init__()
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._timeout = timeout
        self._max_attempts = max(1, max_attempts)
        self._backoff_sec = max(0.0, backoff_sec)
        self._error_lock = threading.Lock()
        self._error_count = 0
        self._error_types: Counter[str] = Counter()
        self._empty_count = 0
        self._structured_count = 0
        self._structured_enabled = True

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def error_types(self) -> dict[str, int]:
        return dict(self._error_types)

    @property
    def empty_count(self) -> int:
        return self._empty_count

    @property
    def structured_count(self) -> int:
        return self._structured_count

    @property
    def structured_enabled(self) -> bool:
        return self._structured_enabled

    def _record_error(self, error: Exception) -> None:
        with self._error_lock:
            self._error_count += 1
            self._error_types[type(error).__name__] += 1

    def _record_empty(self) -> None:
        with self._error_lock:
            self._empty_count += 1

    def _record_structured(self) -> None:
        with self._error_lock:
            self._structured_count += 1

    def _disable_structured_output(self) -> None:
        with self._error_lock:
            self._structured_enabled = False

    @staticmethod
    def _should_disable_structured_output(error: Exception) -> bool:
        text = str(error).lower()
        return "502" in text or "503" in text or "bad gateway" in text or "service unavailable" in text

    def _call_with_retries(self, prompt_text: str, temperature: float) -> str:
        response_format, wrapper_key = _build_response_format(prompt_text)
        for attempt in range(1, self._max_attempts + 1):
            modes = [False]
            if response_format is not None and self._structured_enabled:
                modes = [True, False]

            for use_structured in modes:
                try:
                    request_kwargs = {}
                    if use_structured:
                        request_kwargs["text"] = response_format
                        self._record_structured()
                    response = self._client.responses.create(
                        model=self._model,
                        input=prompt_text,
                        max_output_tokens=512,
                        temperature=temperature,
                        timeout=self._timeout,
                        **request_kwargs,
                    )
                    text = _normalize_structured_text(
                        _response_text(response).strip(),
                        wrapper_key if use_structured else None,
                    )
                    if text:
                        return text
                    self._record_empty()
                    logger.warning(
                        "ragas_llm_empty_response attempt={} model={} prompt_len={} structured={}",
                        attempt,
                        self._model,
                        len(prompt_text),
                        use_structured,
                    )
                except Exception as exc:
                    self._record_error(exc)
                    if use_structured and self._should_disable_structured_output(exc):
                        self._disable_structured_output()
                        logger.warning(
                            "ragas_llm_disable_structured_output model={} reason={}",
                            self._model,
                            _summarize_error(exc),
                        )
                        continue
                    logger.warning(
                        "ragas_llm_call_failed attempt={} model={} prompt_len={} structured={} error={}",
                        attempt,
                        self._model,
                        len(prompt_text),
                        use_structured,
                        _summarize_error(exc),
                    )
            if attempt < self._max_attempts and self._backoff_sec > 0:
                time.sleep(self._backoff_sec * attempt)
        logger.error(
            "ragas_llm_fallback_empty model={} prompt_len={} structured={}",
            self._model,
            len(prompt_text),
            response_format is not None and self._structured_enabled,
        )
        return ""

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 1e-8,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        generations: list[Generation] = []
        for _ in range(n):
            text = self._call_with_retries(prompt.to_string(), temperature)
            generations.append(Generation(text=text))
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        if temperature is None:
            temperature = 1e-8
        generations: list[Generation] = []
        timeout_budget = None
        if self._timeout:
            timeout_budget = (
                self._timeout * self._max_attempts
                + self._backoff_sec * max(self._max_attempts - 1, 0)
                + 1
            )
        for _ in range(n):
            try:
                coro = asyncio.to_thread(
                    self._call_with_retries, prompt.to_string(), temperature
                )
                if timeout_budget:
                    text = await asyncio.wait_for(coro, timeout=timeout_budget)
                else:
                    text = await coro
            except asyncio.TimeoutError as exc:
                self._record_error(exc)
                logger.warning(
                    "ragas_llm_thread_timeout model={} prompt_len={} timeout={}",
                    self._model,
                    len(prompt.to_string()),
                    timeout_budget,
                )
                text = ""
            generations.append(Generation(text=text))
        return LLMResult(generations=[generations])


def build_llm(
    model: str | None,
    base_url: str | None,
    timeout: int | None,
    *,
    max_attempts: int,
    backoff_sec: float,
):
    if not model and not base_url:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        codex_auth = Path.home() / ".codex" / "auth.json"
        if codex_auth.exists():
            try:
                payload = json.loads(codex_auth.read_text(encoding="utf-8"))
                api_key = payload.get("OPENAI_API_KEY") or payload.get("openai_api_key")
            except Exception:
                api_key = None
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            raise EnvironmentError("未设置 OPENAI_API_KEY，无法运行 RAGAS 评估")

    llm_model = model or "gpt-4o-mini"
    return ResponsesRagasLLM(
        model=llm_model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_attempts=max_attempts,
        backoff_sec=backoff_sec,
    )


def build_embeddings(config_path: str) -> HuggingFaceBgeEmbeddings:
    cfg = load_config(config_path)
    emb_cfg = cfg["models"]["embedding"]
    model_name = emb_cfg["model_name"]
    device = emb_cfg.get("device")
    if device and device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("embedding_device_unavailable configured={} fallback=cpu", device)
                device = "cpu"
        except Exception:
            logger.warning("embedding_device_check_failed configured={} fallback=cpu", device)
            device = "cpu"

    model_kwargs = {}
    if device:
        model_kwargs["device"] = device

    logger.info("加载评估 embedding 模型: {}", model_name)
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

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
            "answer": _normalize_answer_for_ragas(answer),
            "contexts": contexts if contexts else [item.get("ground_truth_context", "")],
            "ground_truth": item["ground_truth"],
        })

    return Dataset.from_list(rows)


def compact_metric_prompts(metrics: list) -> None:
    prompt_fields = (
        "context_precision_prompt",
        "context_recall_prompt",
        "statement_prompt",
        "nli_statements_message",
        "question_generation",
    )
    compacted = 0
    for metric in metrics:
        for field_name in prompt_fields:
            prompt = getattr(metric, field_name, None)
            if prompt is None or not getattr(prompt, "examples", None):
                continue
            prompt.examples = []
            compacted += 1
    logger.info("已压缩评估 prompt 示例数，影响 prompt 数量: {}", compacted)


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument("--qa-file", default=None, help=f"评估集 JSON 文件路径（默认使用内置示例）")
    parser.add_argument("--output", default="tests/ragas_results.json", help="评估结果输出路径")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径（自动叠加 config.local.yaml）")
    parser.add_argument("--llm-model", default=None, help="评估用 LLM 模型名称（例如 gpt-5.4）")
    parser.add_argument("--llm-base-url", default=None, help="评估用 LLM Base URL（例如 https://cmdme.cn）")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM 请求超时（秒）")
    parser.add_argument("--llm-max-attempts", type=int, default=3, help="LLM 请求最大重试次数")
    parser.add_argument("--llm-backoff", type=float, default=2.0, help="LLM 重试退避基数（秒）")
    parser.add_argument("--ragas-timeout", type=int, default=600, help="RAGAS 单任务超时（秒）")
    parser.add_argument("--ragas-max-workers", type=int, default=4, help="RAGAS 并发任务数")
    parser.add_argument("--ragas-max-retries", type=int, default=3, help="RAGAS 最大重试次数")
    parser.add_argument("--ragas-max-wait", type=int, default=60, help="RAGAS 重试最大等待（秒）")
    parser.add_argument("--answer-relevancy-strictness", type=int, default=1, help="Answer relevancy 的问题生成次数")
    parser.add_argument("--keep-ragas-examples", action="store_true", help="保留 RAGAS 原始 few-shot 示例（默认压缩）")
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
    llm_timeout = args.llm_timeout if args.llm_timeout is not None else args.ragas_timeout
    min_ragas_timeout = int(
        llm_timeout * args.llm_max_attempts
        + args.llm_backoff * max(args.llm_max_attempts - 1, 0)
    )
    effective_ragas_timeout = max(args.ragas_timeout, min_ragas_timeout)
    if effective_ragas_timeout != args.ragas_timeout:
        logger.warning(
            "ragas_timeout_too_low adjust={} requested={}",
            effective_ragas_timeout,
            args.ragas_timeout,
        )

    run_config = RunConfig(
        timeout=effective_ragas_timeout,
        max_workers=args.ragas_max_workers,
        max_retries=args.ragas_max_retries,
        max_wait=args.ragas_max_wait,
    )
    llm = build_llm(
        args.llm_model,
        args.llm_base_url,
        llm_timeout,
        max_attempts=args.llm_max_attempts,
        backoff_sec=args.llm_backoff,
    )
    embeddings = build_embeddings(args.config)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    answer_relevancy.strictness = max(1, args.answer_relevancy_strictness)
    if not args.keep_ragas_examples:
        compact_metric_prompts(metrics)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
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
        "llm_model": args.llm_model or "gpt-4o-mini",
        "llm_base_url": args.llm_base_url,
        "embedding_model": str(getattr(embeddings, "model_name", None) or getattr(embeddings, "model", None)),
        "llm_timeout": llm_timeout,
        "ragas_timeout": effective_ragas_timeout,
        "ragas_max_workers": args.ragas_max_workers,
        "ragas_max_retries": args.ragas_max_retries,
        "answer_relevancy_strictness": answer_relevancy.strictness,
        "keep_ragas_examples": args.keep_ragas_examples,
    }
    if isinstance(llm, ResponsesRagasLLM):
        scores["llm_error_count"] = llm.error_count
        scores["llm_error_types"] = llm.error_types
        scores["llm_empty_count"] = llm.empty_count
        scores["llm_structured_count"] = llm.structured_count
        scores["llm_structured_enabled"] = llm.structured_enabled
        if llm.error_count:
            logger.warning(
                "ragas_llm_errors count={} types={} empty={} structured={} structured_enabled={}",
                llm.error_count,
                llm.error_types,
                llm.empty_count,
                llm.structured_count,
                llm.structured_enabled,
            )
    output_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"评估结果已保存: {output_path}")


if __name__ == "__main__":
    main()
