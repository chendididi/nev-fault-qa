"""
运行时自动验收脚本。

验证目标：
1) /health 存活契约
2) /ready 就绪契约
3) /api/v1/chat 对话契约

支持直接校验已有 API，也支持 --spawn-api 临时拉起 uvicorn 再验收。
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


def validate_health_payload(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "health 响应不是对象"
    if payload.get("status") != "ok":
        return False, "health.status 必须为 ok"
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        return False, "health.version 必须为非空字符串"
    return True, None


def validate_ready_payload(payload: Any) -> tuple[bool, str | None]:
    required_checks = {
        "chunks_file_present",
        "bm25_loaded",
        "qwen_loaded",
        "milvus_connected",
        "neo4j_connected",
    }
    if not isinstance(payload, dict):
        return False, "ready 响应不是对象"
    status = payload.get("status")
    if status not in {"ok", "degraded"}:
        return False, "ready.status 必须为 ok 或 degraded"
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        return False, "ready.version 必须为非空字符串"
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return False, "ready.checks 必须为对象"
    missing = required_checks - set(checks.keys())
    if missing:
        return False, f"ready.checks 缺失字段: {sorted(missing)}"
    for key in required_checks:
        if not isinstance(checks.get(key), bool):
            return False, f"ready.checks.{key} 必须为布尔值"
    return True, None


def validate_chat_payload(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "chat 响应不是对象"
    session_id = payload.get("session_id")
    answer = payload.get("answer")
    sources = payload.get("sources")
    if not isinstance(session_id, str) or not session_id.strip():
        return False, "chat.session_id 必须为非空字符串"
    if not isinstance(answer, str):
        return False, "chat.answer 必须为字符串"
    if not isinstance(sources, list):
        return False, "chat.sources 必须为数组"
    return True, None


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


def _run_http_check(
    *,
    name: str,
    method: str,
    url: str,
    timeout_sec: int,
    payload: dict | None = None,
    validator=None,
) -> dict[str, Any]:
    started_at = time.time()
    started_perf = time.perf_counter()
    result: dict[str, Any] = {
        "name": name,
        "method": method,
        "url": url,
        "started_at": started_at,
        "finished_at": None,
        "latency_ms": None,
        "status_code": None,
        "ok": False,
        "schema_ok": False,
        "error": None,
        "schema_error": None,
        "response_json": None,
    }
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=timeout_sec)
        else:
            response = requests.post(url, json=payload, timeout=timeout_sec)

        result["status_code"] = response.status_code
        result["latency_ms"] = int((time.perf_counter() - started_perf) * 1000)

        body = response.json()
        result["response_json"] = body

        schema_ok = True
        schema_error = None
        if validator is not None:
            schema_ok, schema_error = validator(body)
        result["schema_ok"] = bool(schema_ok)
        result["schema_error"] = schema_error
        result["ok"] = response.status_code == 200 and bool(schema_ok)
    except requests.RequestException as exc:
        result["latency_ms"] = int((time.perf_counter() - started_perf) * 1000)
        result["error"] = str(exc)
    except ValueError as exc:
        result["latency_ms"] = int((time.perf_counter() - started_perf) * 1000)
        result["error"] = f"响应不是合法 JSON: {exc}"
    except Exception as exc:  # pragma: no cover - 兜底保护
        result["latency_ms"] = int((time.perf_counter() - started_perf) * 1000)
        result["error"] = str(exc)
    finally:
        result["finished_at"] = time.time()

    return result


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


def run_validation(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    run = RunRecorder.start(
        pipeline="validate_runtime",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "base_url": args.base_url,
            "spawn_api": args.spawn_api,
            "host": args.host,
            "port": args.port,
            "uvicorn_app": args.uvicorn_app,
            "startup_timeout": args.startup_timeout,
            "request_timeout": args.request_timeout,
            "chat_timeout": args.chat_timeout,
            "skip_chat": args.skip_chat,
            "require_ready_ok": args.require_ready_ok,
            "message_length": len(args.message),
        },
    )
    setup_logging(cfg, component="validate_runtime", run_log_path=run.run_dir / "run.log")

    api_proc: subprocess.Popen | None = None
    api_log_handle = None
    api_log_path = run.run_dir / "api_process.log"
    effective_base_url = _normalize_base_url(args.base_url)
    check_results: list[dict[str, Any]] = []

    try:
        with log_context(run_id=run.run_id, pipeline="validate_runtime"):
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

            health_result = _run_http_check(
                name="health",
                method="GET",
                url=f"{effective_base_url}/health",
                timeout_sec=args.request_timeout,
                validator=validate_health_payload,
            )
            check_results.append(health_result)

            ready_result = _run_http_check(
                name="ready",
                method="GET",
                url=f"{effective_base_url}/ready",
                timeout_sec=args.request_timeout,
                validator=validate_ready_payload,
            )
            check_results.append(ready_result)

            if not args.skip_chat:
                chat_result = _run_http_check(
                    name="chat",
                    method="POST",
                    url=f"{effective_base_url}/api/v1/chat",
                    timeout_sec=args.chat_timeout,
                    payload={"message": args.message},
                    validator=validate_chat_payload,
                )
                check_results.append(chat_result)

        checks_path = run.run_dir / "runtime_checks.json"
        checks_payload = {
            "base_url": effective_base_url,
            "checks": check_results,
        }
        checks_path.write_text(
            json.dumps(checks_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run.add_output(
            {
                "name": "runtime_checks.json",
                "artifact_path": str(checks_path),
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

        all_ok = all(item.get("ok") for item in check_results)
        ready_status = None
        if ready_result := next((item for item in check_results if item["name"] == "ready"), None):
            response_json = ready_result.get("response_json") or {}
            ready_status = response_json.get("status")

        if args.require_ready_ok and ready_status != "ok":
            all_ok = False

        run.update_stats(
            base_url=effective_base_url,
            check_count=len(check_results),
            passed_count=sum(1 for item in check_results if item.get("ok")),
            failed_count=sum(1 for item in check_results if not item.get("ok")),
            ready_status=ready_status,
        )

        if all_ok:
            run.mark_success()
            return 0

        run.mark_failure("存在未通过的运行时校验项")
        return 1
    except Exception as exc:
        logger.exception("validate_runtime_failed")
        run.mark_failure(str(exc))
        return 1
    finally:
        _terminate_process(api_proc)
        if api_log_handle is not None:
            api_log_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校验 NEV API 运行时契约")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="已运行 API 的地址")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径（spawn API 时生效）")
    parser.add_argument("--local-config", default=None, help="可选本地覆盖配置路径（spawn API 时生效）")
    parser.add_argument("--spawn-api", action="store_true", help="自动拉起 uvicorn，校验结束后关闭")
    parser.add_argument("--host", default="127.0.0.1", help="spawn API 的 host")
    parser.add_argument("--port", type=int, default=8000, help="spawn API 的 port")
    parser.add_argument("--uvicorn-app", default="src.api.main:app", help="uvicorn app 路径")
    parser.add_argument("--startup-timeout", type=int, default=180, help="spawn API 启动超时（秒）")
    parser.add_argument("--request-timeout", type=int, default=20, help="health/ready 请求超时（秒）")
    parser.add_argument("--chat-timeout", type=int, default=120, help="chat 请求超时（秒）")
    parser.add_argument("--skip-chat", action="store_true", help="跳过 /api/v1/chat 契约校验")
    parser.add_argument("--require-ready-ok", action="store_true", help="要求 /ready.status 必须为 ok")
    parser.add_argument("--message", default="P0300 故障码的含义是什么？", help="chat 测试问题")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_validation(args))


if __name__ == "__main__":
    main()
