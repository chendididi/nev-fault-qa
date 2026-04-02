"""
一键编排离线自动化流程（下载 -> 索引 -> 可选向量/图谱 -> handoff）。

默认遵循仓库 harness 的轻量优先路径：
1) make check
2) 官方文档下载 + Tesla HTML 抓取
3) build_index --skip-embedding
4) handoff 快照

可通过参数开启重依赖阶段（Milvus / Neo4j）以及 hook。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


HOOK_WHENS = {"pre", "post"}


@dataclass
class Stage:
    name: str
    command: list[str] | None = None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _parse_hook_point(raw: str) -> tuple[str, str]:
    stage, sep, when = raw.rpartition(".")
    if not sep or not stage:
        raise ValueError(f"非法 hook 键: {raw}，格式应为 <stage>.<pre|post>")
    when = when.strip()
    if when not in HOOK_WHENS:
        raise ValueError(f"非法 hook 时机: {raw}，仅支持 pre/post")
    return stage.strip(), when


def _append_hook(hooks: dict[str, list[str]], stage: str, when: str, command: str) -> None:
    command = command.strip()
    if not command:
        raise ValueError(f"hook 命令不能为空: {stage}.{when}")
    hooks.setdefault(f"{stage}.{when}", []).append(command)


def _parse_hook_spec(spec: str) -> tuple[str, str, str]:
    key, sep, command = spec.partition("=")
    if not sep:
        raise ValueError(f"非法 --hook 参数: {spec}，格式应为 <stage>.<pre|post>=<command>")
    stage, when = _parse_hook_point(key.strip())
    return stage, when, command.strip()


def load_hooks(path: str | None, inline_specs: list[str] | None) -> dict[str, list[str]]:
    hooks: dict[str, list[str]] = {}

    if path:
        hook_path = Path(path)
        payload = json.loads(hook_path.read_text(encoding="utf-8"))
        raw_hooks = payload.get("hooks", payload)
        if not isinstance(raw_hooks, dict):
            raise ValueError("hooks 文件格式错误：顶层应为对象或包含 hooks 对象")

        for key, value in raw_hooks.items():
            if isinstance(value, dict):
                stage = key.strip()
                for when in sorted(HOOK_WHENS):
                    commands = value.get(when, [])
                    if isinstance(commands, str):
                        _append_hook(hooks, stage, when, commands)
                    elif isinstance(commands, list):
                        for command in commands:
                            _append_hook(hooks, stage, when, str(command))
                    elif commands not in (None, []):
                        raise ValueError(f"hooks[{key!r}][{when!r}] 必须是字符串或字符串数组")
                continue

            stage, when = _parse_hook_point(str(key).strip())
            commands = value if isinstance(value, list) else [value]
            for command in commands:
                _append_hook(hooks, stage, when, str(command))

    for spec in inline_specs or []:
        stage, when, command = _parse_hook_spec(spec)
        _append_hook(hooks, stage, when, command)

    return hooks


def _tcp_check(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _service_checks(args: argparse.Namespace, cfg: dict) -> dict[str, object]:
    checks: dict[str, object] = {}

    if args.with_embedding:
        milvus_cfg = cfg.get("milvus", {})
        host = str(milvus_cfg.get("host", "localhost"))
        port = int(milvus_cfg.get("port", 19530))
        ok = _tcp_check(host, port)
        checks["milvus"] = {"host": host, "port": port, "ok": ok}
        if args.require_services and not ok:
            raise RuntimeError(f"Milvus 不可达: {host}:{port}")

    if args.with_graph:
        neo4j_cfg = cfg.get("neo4j", {})
        uri = str(neo4j_cfg.get("uri", "bolt://localhost:7687"))
        parsed = urlparse(uri)
        host = parsed.hostname or "localhost"
        port = int(parsed.port or 7687)
        ok = _tcp_check(host, port)
        checks["neo4j"] = {"uri": uri, "host": host, "port": port, "ok": ok}
        if args.require_services and not ok:
            raise RuntimeError(f"Neo4j 不可达: {host}:{port} ({uri})")

    return checks


def run_preflight(args: argparse.Namespace, cfg: dict) -> dict[str, object]:
    input_states: list[dict[str, object]] = []
    for raw_input in args.input:
        input_path = Path(raw_input)
        existed = input_path.exists()

        if not existed and args.skip_download:
            raise FileNotFoundError(
                f"输入路径不存在且你启用了 --skip-download: {input_path}. "
                "请先准备数据，或去掉 --skip-download。"
            )

        if not existed:
            if input_path.suffix:
                input_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                input_path.mkdir(parents=True, exist_ok=True)

        input_states.append(
            {
                "path": str(input_path),
                "exists_before": existed,
                "exists_after": input_path.exists(),
            }
        )

    checks = _service_checks(args, cfg)

    return {
        "input_paths": input_states,
        "service_checks": checks,
    }


def _build_stage_plan(args: argparse.Namespace, python_bin: str) -> list[Stage]:
    stages: list[Stage] = [Stage("preflight", None)]

    if not args.skip_check:
        stages.append(Stage("check", ["make", "check"]))

    if not args.skip_download:
        download_cmd = [python_bin, "scripts/download_official_docs.py", "--config", args.config]
        for brand in args.brand or []:
            download_cmd.extend(["--brand", brand])
        if args.include_html:
            download_cmd.append("--include-html")
        if args.force_download:
            download_cmd.append("--force")
        stages.append(Stage("download_docs", download_cmd))

    if args.with_nhtsa:
        nhtsa_cmd = [
            python_bin,
            "scripts/download_nhtsa_data.py",
            "--config",
            args.config,
        ]
        for year_range in args.nhtsa_year_range or []:
            nhtsa_cmd.extend(["--year-range", year_range])
        for make in args.nhtsa_make or []:
            nhtsa_cmd.extend(["--make", make])
        if args.nhtsa_max_chunks:
            nhtsa_cmd.extend(["--max-chunks", str(args.nhtsa_max_chunks)])
        if args.force_download:
            nhtsa_cmd.append("--force")
        stages.append(Stage("download_nhtsa", nhtsa_cmd))

    if not args.skip_fetch_html:
        fetch_cmd = [
            python_bin,
            "scripts/fetch_tesla_service_manual.py",
            "--config",
            args.config,
            "--manual",
            args.manual,
            "--max-pages",
            str(args.max_pages),
        ]
        if args.seed_file:
            fetch_cmd.extend(["--seed-file", args.seed_file])
        stages.append(Stage("fetch_manual_html", fetch_cmd))

    build_index_cmd = [
        python_bin,
        "scripts/build_index.py",
        "--config",
        args.config,
        "--input",
        *args.input,
        "--skip-embedding",
    ]
    if args.with_circuit_ocr:
        build_index_cmd.append("--with-circuit-ocr")
    if args.disable_table_extraction:
        build_index_cmd.append("--disable-table-extraction")
    stages.append(Stage("build_chunks", build_index_cmd))

    if args.with_embedding:
        stages.append(
            Stage(
                "build_vector",
                [
                    python_bin,
                    "scripts/load_embeddings.py",
                    "--config",
                    args.config,
                    "--input",
                    args.chunks_file,
                    "--batch-size",
                    str(args.embedding_batch_size),
                ],
            )
        )

    if args.with_graph:
        build_graph_cmd = [
            python_bin,
            "scripts/build_graph.py",
            "--config",
            args.config,
            "--chunks-file",
            args.chunks_file,
        ]
        if args.entities_file:
            build_graph_cmd.extend(["--entities-file", args.entities_file])
        stages.append(Stage("build_graph", build_graph_cmd))

    if not args.skip_handoff:
        stages.append(
            Stage(
                "handoff",
                [python_bin, "scripts/handoff_snapshot.py", "--config", args.config],
            )
        )

    return stages


def _run_hook_commands(
    hooks: dict[str, list[str]],
    *,
    stage: str,
    when: str,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    for hook_cmd in hooks.get(f"{stage}.{when}", []):
        logger.info("执行 hook: {}.{} -> {}", stage, when, hook_cmd)
        if dry_run:
            continue
        subprocess.run(hook_cmd, cwd=cwd, env=env, shell=True, check=True)


def _run_stage_command(command: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    logger.info("执行阶段命令: {}", shlex.join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _stage_record(stage: Stage, status: str, started_at: str, finished_at: str, error: str | None = None) -> dict:
    return {
        "stage": stage.name,
        "status": status,
        "command": shlex.join(stage.command) if stage.command else None,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
    }


def run_pipeline(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    hooks = load_hooks(args.hooks_file, args.hook)

    run = RunRecorder.start(
        pipeline="auto_pipeline",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "input_paths": args.input,
            "brands": args.brand or ["tesla", "byd"],
            "include_html": args.include_html,
            "with_nhtsa": args.with_nhtsa,
            "nhtsa_year_ranges": args.nhtsa_year_range or [],
            "nhtsa_makes": args.nhtsa_make or [],
            "nhtsa_max_chunks": args.nhtsa_max_chunks,
            "manual": args.manual,
            "max_pages": args.max_pages,
            "skip_download": args.skip_download,
            "skip_fetch_html": args.skip_fetch_html,
            "skip_check": args.skip_check,
            "with_embedding": args.with_embedding,
            "with_graph": args.with_graph,
            "skip_handoff": args.skip_handoff,
            "continue_on_error": args.continue_on_error,
            "dry_run": args.dry_run,
            "require_services": args.require_services,
            "hooks_file": args.hooks_file,
            "hook_count": sum(len(values) for values in hooks.values()),
        },
    )
    setup_logging(cfg, component="auto_pipeline", run_log_path=run.run_dir / "run.log")

    env = os.environ.copy()
    env["NEV_AUTO_PIPELINE_RUN_ID"] = run.run_id
    env["NEV_AUTO_PIPELINE_CONFIG"] = args.config

    stages = _build_stage_plan(args, sys.executable or "python")
    stage_results: list[dict] = []
    failures: list[str] = []

    logger.info("自动化阶段计划: {}", [stage.name for stage in stages])
    run.update_stats(planned_stage_count=len(stages))
    run.update_stats(planned_stages=[stage.name for stage in stages])

    try:
        with log_context(run_id=run.run_id, pipeline="auto_pipeline"):
            _run_hook_commands(
                hooks,
                stage="pipeline",
                when="pre",
                cwd=repo_root,
                env=env,
                dry_run=args.dry_run,
            )

            for stage in stages:
                started_at = _utc_now()
                try:
                    _run_hook_commands(
                        hooks,
                        stage=stage.name,
                        when="pre",
                        cwd=repo_root,
                        env=env,
                        dry_run=args.dry_run,
                    )

                    if stage.name == "preflight":
                        preflight_stats = run_preflight(args, cfg)
                        run.update_stats(preflight=preflight_stats)
                        logger.info("preflight 完成: {}", preflight_stats)
                    elif stage.command:
                        _run_stage_command(stage.command, cwd=repo_root, env=env, dry_run=args.dry_run)

                    _run_hook_commands(
                        hooks,
                        stage=stage.name,
                        when="post",
                        cwd=repo_root,
                        env=env,
                        dry_run=args.dry_run,
                    )

                    finished_at = _utc_now()
                    stage_results.append(_stage_record(stage, "succeeded", started_at, finished_at))
                except Exception as exc:
                    finished_at = _utc_now()
                    error = str(exc)
                    stage_results.append(_stage_record(stage, "failed", started_at, finished_at, error))
                    failures.append(stage.name)
                    logger.exception("stage_failed")
                    if not args.continue_on_error:
                        raise

            _run_hook_commands(
                hooks,
                stage="pipeline",
                when="post",
                cwd=repo_root,
                env=env,
                dry_run=args.dry_run,
            )

        stage_results_path = run.run_dir / "stage_results.json"
        stage_results_path.write_text(
            json.dumps(stage_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run.add_output(
            {
                "name": "stage_results.json",
                "artifact_path": str(stage_results_path),
                "active_path": None,
                "activate_on_publish": False,
            }
        )

        run.update_stats(
            succeeded_stage_count=sum(1 for item in stage_results if item["status"] == "succeeded"),
            failed_stage_count=sum(1 for item in stage_results if item["status"] == "failed"),
            failed_stages=failures,
        )

        if failures:
            message = f"自动化流程存在失败阶段: {failures}"
            run.mark_failure(message)
            return 1

        run.mark_success()
        return 0
    except Exception as exc:
        run.mark_failure(str(exc))
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键执行 NEV 项目离线自动化流程")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/raw/official"],
        help="build_index 输入路径（支持多个，默认 data/raw/official）",
    )
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--brand", action="append", choices=["tesla", "byd"], help="下载官方文档时按品牌过滤")
    parser.add_argument("--include-html", action="store_true", help="下载官方清单中的 HTML 入口页")
    parser.add_argument("--force-download", action="store_true", help="覆盖已存在官方文档文件")
    parser.add_argument("--skip-download", action="store_true", help="跳过官方文档下载阶段")
    parser.add_argument("--with-nhtsa", action="store_true", help="启用 NHTSA Manufacturer Communications 数据下载")
    parser.add_argument(
        "--nhtsa-year-range",
        action="append",
        default=[],
        help="NHTSA 数据区间（YYYY-YYYY），可重复；默认由下载脚本使用最新区间",
    )
    parser.add_argument(
        "--nhtsa-make",
        action="append",
        default=[],
        help="NHTSA chunks 生成时按 Make 过滤，可重复（例如 TESLA）",
    )
    parser.add_argument("--nhtsa-max-chunks", type=int, default=0, help="限制 NHTSA chunks 最大条数（0=不限制）")
    parser.add_argument("--skip-fetch-html", action="store_true", help="跳过 Tesla HTML 抓取阶段")
    parser.add_argument("--manual", default="model3_2024_en_us", help="Tesla 手册版本（传给 fetch 脚本）")
    parser.add_argument("--max-pages", type=int, default=200, help="Tesla HTML 最多抓取页数")
    parser.add_argument("--seed-file", default=None, help="Tesla HTML 抓取 seed 文件路径")
    parser.add_argument("--skip-check", action="store_true", help="跳过 make check")
    parser.add_argument("--with-circuit-ocr", action="store_true", help="build_index 启用电路图 OCR")
    parser.add_argument("--disable-table-extraction", action="store_true", help="build_index 禁用表格抽取")

    parser.add_argument("--with-embedding", action="store_true", help="启用 Milvus 向量索引阶段")
    parser.add_argument("--embedding-batch-size", type=int, default=256, help="load_embeddings 批大小")
    parser.add_argument("--with-graph", action="store_true", help="启用 Neo4j 图谱构建阶段")
    parser.add_argument("--entities-file", default=None, help="可选：build_graph 复用已有实体结果")
    parser.add_argument("--chunks-file", default="data/processed/chunks.json", help="向量/图谱阶段使用的 chunks 文件")
    parser.add_argument("--require-services", action="store_true", help="preflight 严格检查 Milvus/Neo4j 连通性")
    parser.add_argument("--skip-handoff", action="store_true", help="跳过 handoff 快照阶段")

    parser.add_argument("--hooks-file", default=None, help="hook JSON 配置文件")
    parser.add_argument(
        "--hook",
        action="append",
        default=[],
        help="临时 hook，格式: <stage>.<pre|post>=<command>，可重复",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="阶段失败后继续执行后续阶段")
    parser.add_argument("--dry-run", action="store_true", help="只展示执行计划，不真正执行命令")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
