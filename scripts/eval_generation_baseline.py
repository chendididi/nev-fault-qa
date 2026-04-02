"""
固定生成基线评估（RAGAS + 阈值门禁）。

流程：
1) 可选自动拉起 API
2) 调用 tests/eval_ragas.py 生成固定集评估结果
3) 按阈值文件判定是否通过
4) 写入版本化产物与运行 manifest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _probe_api_ready(base_url: str, timeout_sec: int, request_timeout_sec: int) -> tuple[bool, str | None]:
    endpoint = f"{_normalize_base_url(base_url)}/health"
    deadline = time.monotonic() + timeout_sec
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(endpoint, timeout=request_timeout_sec)
            if resp.status_code == 200:
                return True, None
            last_error = f"/health status={resp.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(1.0)
    return False, last_error


def _build_api_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        args.uvicorn_app,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def _build_ragas_command(
    args: argparse.Namespace,
    *,
    ragas_output: Path,
    api_base: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "tests/eval_ragas.py",
        "--qa-file",
        args.qa_file,
        "--output",
        str(ragas_output),
        "--config",
        args.config,
        "--api-base",
        api_base,
        "--api-timeout",
        str(args.api_timeout),
        "--ragas-timeout",
        str(args.ragas_timeout),
        "--ragas-max-workers",
        str(args.ragas_max_workers),
        "--ragas-max-retries",
        str(args.ragas_max_retries),
        "--llm-max-attempts",
        str(args.llm_max_attempts),
        "--llm-backoff",
        str(args.llm_backoff),
        "--answer-relevancy-strictness",
        str(args.answer_relevancy_strictness),
    ]
    if args.keep_ragas_examples:
        cmd.append("--keep-ragas-examples")
    if args.llm_model:
        cmd.extend(["--llm-model", args.llm_model])
    if args.llm_base_url:
        cmd.extend(["--llm-base-url", args.llm_base_url])
    if args.llm_timeout is not None:
        cmd.extend(["--llm-timeout", str(args.llm_timeout)])
    return cmd


def _terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def load_thresholds(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("阈值文件必须是非空对象")

    thresholds: dict[str, float] = {}
    for metric in RAGAS_METRICS:
        key = f"{metric}_min"
        if key in payload:
            thresholds[metric] = float(payload[key])
    if not thresholds:
        raise ValueError("阈值文件缺少 *_min 配置")
    return thresholds


def evaluate_scores(scores: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    overall_pass = True
    for metric, minimum in thresholds.items():
        actual = float(scores.get(metric, 0.0))
        passed = actual >= minimum
        checks.append(
            {
                "metric": metric,
                "actual": actual,
                "minimum": minimum,
                "passed": passed,
            }
        )
        overall_pass = overall_pass and passed

    return {
        "overall_pass": overall_pass,
        "checks": checks,
    }


def run_eval(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    run = RunRecorder.start(
        pipeline="eval_generation_baseline",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "qa_file": args.qa_file,
            "thresholds_file": args.thresholds_file,
            "fail_on_threshold": args.fail_on_threshold,
            "spawn_api": args.spawn_api,
            "base_url": args.base_url,
            "host": args.host,
            "port": args.port,
            "uvicorn_app": args.uvicorn_app,
            "startup_timeout": args.startup_timeout,
            "request_timeout": args.request_timeout,
            "llm_model": args.llm_model,
            "llm_base_url": args.llm_base_url,
            "llm_timeout": args.llm_timeout,
            "llm_max_attempts": args.llm_max_attempts,
            "llm_backoff": args.llm_backoff,
            "ragas_timeout": args.ragas_timeout,
            "ragas_max_workers": args.ragas_max_workers,
            "ragas_max_retries": args.ragas_max_retries,
            "answer_relevancy_strictness": args.answer_relevancy_strictness,
            "keep_ragas_examples": args.keep_ragas_examples,
            "api_timeout": args.api_timeout,
            "output": args.output,
        },
    )
    setup_logging(cfg, component="eval_generation_baseline", run_log_path=run.run_dir / "run.log")

    api_proc: subprocess.Popen | None = None
    api_log_handle = None
    api_log_path = run.run_dir / "api_process.log"
    ragas_output = run.run_dir / "ragas_results.json"
    baseline_report_path = run.run_dir / "generation_baseline_report.json"
    effective_base_url = _normalize_base_url(args.base_url)

    try:
        with log_context(run_id=run.run_id, pipeline="eval_generation_baseline"):
            if args.spawn_api:
                effective_base_url = f"http://{args.host}:{args.port}"
                cmd = _build_api_command(args)
                env = os.environ.copy()
                env["NEV_CONFIG_PATH"] = args.config
                if args.local_config:
                    env["NEV_LOCAL_CONFIG_PATH"] = args.local_config

                api_log_handle = api_log_path.open("w", encoding="utf-8")
                logger.info("启动 API 进程: {}", " ".join(cmd))
                api_proc = subprocess.Popen(
                    cmd,
                    cwd=repo_root,
                    env=env,
                    stdout=api_log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                run.update_stats(spawned_api_pid=api_proc.pid, api_command=cmd)

                ready, ready_error = _probe_api_ready(
                    effective_base_url,
                    timeout_sec=args.startup_timeout,
                    request_timeout_sec=args.request_timeout,
                )
                if not ready:
                    raise RuntimeError(f"API 启动超时或不可达: {ready_error or 'unknown'}")

            ragas_cmd = _build_ragas_command(
                args,
                ragas_output=ragas_output,
                api_base=f"{effective_base_url}/api/v1",
            )
            logger.info("执行 RAGAS 命令: {}", " ".join(ragas_cmd))
            subprocess.run(
                ragas_cmd,
                cwd=repo_root,
                env=os.environ.copy(),
                check=True,
            )

            scores = json.loads(ragas_output.read_text(encoding="utf-8"))
            threshold_result: dict[str, Any] | None = None
            thresholds: dict[str, float] | None = None
            if args.thresholds_file:
                thresholds = load_thresholds(args.thresholds_file)
                threshold_result = evaluate_scores(scores, thresholds)

            report = {
                "run_id": run.run_id,
                "ragas_output": str(ragas_output),
                "scores": {metric: scores.get(metric) for metric in RAGAS_METRICS},
                "thresholds": thresholds,
                "threshold_result": threshold_result,
                "meta": {
                    "dataset_path": scores.get("dataset_path"),
                    "dataset_sha256": scores.get("dataset_sha256"),
                    "dataset_source": scores.get("dataset_source"),
                    "dataset_count": scores.get("dataset_count"),
                    "llm_model": scores.get("llm_model"),
                    "llm_base_url": scores.get("llm_base_url"),
                    "embedding_model": scores.get("embedding_model"),
                },
            }
            baseline_report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            run.add_output(
                {
                    "name": "ragas_results.json",
                    "artifact_path": str(ragas_output),
                    "active_path": None,
                    "activate_on_publish": False,
                }
            )
            run.add_output(
                {
                    "name": "generation_baseline_report.json",
                    "artifact_path": str(baseline_report_path),
                    "active_path": None,
                    "activate_on_publish": False,
                }
            )
            if args.spawn_api and api_log_path.exists():
                run.add_output(
                    {
                        "name": "api_process.log",
                        "artifact_path": str(api_log_path),
                        "active_path": None,
                        "activate_on_publish": False,
                    }
                )

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
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
                base_url=effective_base_url,
                scores={metric: scores.get(metric) for metric in RAGAS_METRICS},
                threshold_pass=(
                    None
                    if threshold_result is None
                    else bool(threshold_result["overall_pass"])
                ),
            )

            if threshold_result and args.fail_on_threshold and not threshold_result["overall_pass"]:
                run.mark_failure("生成基线阈值校验未通过")
                return 1

            run.mark_success()
            return 0
    except Exception as exc:
        logger.exception("eval_generation_baseline_failed")
        run.mark_failure(str(exc))
        return 1
    finally:
        _terminate_process(api_proc)
        if api_log_handle is not None:
            api_log_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行固定集 RAGAS 并执行阈值门禁")
    parser.add_argument("--qa-file", default="tests/eval_qa_set.json", help="RAGAS 评测集路径")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件（API 与 embedding 共用）")
    parser.add_argument("--local-config", default=None, help="可选本地覆盖配置（spawn API 时生效）")
    parser.add_argument("--thresholds-file", default="tests/eval_ragas_thresholds.json", help="阈值文件路径")
    parser.add_argument("--fail-on-threshold", action="store_true", help="阈值未达标时返回非 0")
    parser.add_argument("--output", default=None, help="可选报告输出路径")

    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="已运行 API 地址")
    parser.add_argument("--spawn-api", action="store_true", help="自动拉起 API 再评估")
    parser.add_argument("--host", default="127.0.0.1", help="spawn API host")
    parser.add_argument("--port", type=int, default=8000, help="spawn API port")
    parser.add_argument("--uvicorn-app", default="src.api.main:app", help="uvicorn app 路径")
    parser.add_argument("--startup-timeout", type=int, default=300, help="API 启动超时（秒）")
    parser.add_argument("--request-timeout", type=int, default=20, help="API 就绪探活超时（秒）")

    parser.add_argument("--llm-model", default="gpt-4o-mini", help="评估 LLM 模型")
    parser.add_argument("--llm-base-url", default=None, help="评估 LLM Base URL")
    parser.add_argument("--api-timeout", type=int, default=180, help="单次被评估 API 请求超时（秒）")
    parser.add_argument("--llm-timeout", type=int, default=None, help="评估 LLM 请求超时（秒）")
    parser.add_argument("--llm-max-attempts", type=int, default=3, help="评估 LLM 最大重试次数")
    parser.add_argument("--llm-backoff", type=float, default=2.0, help="评估 LLM 重试退避")
    parser.add_argument("--ragas-timeout", type=int, default=600, help="RAGAS 单任务超时（秒）")
    parser.add_argument("--ragas-max-workers", type=int, default=4, help="RAGAS 并发任务数")
    parser.add_argument("--ragas-max-retries", type=int, default=3, help="RAGAS 最大重试次数")
    parser.add_argument(
        "--answer-relevancy-strictness",
        type=int,
        default=1,
        help="Answer relevancy 的问题生成次数",
    )
    parser.add_argument("--keep-ragas-examples", action="store_true", help="保留 RAGAS 原始示例")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_eval(args))


if __name__ == "__main__":
    main()
