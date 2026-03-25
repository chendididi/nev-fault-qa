"""
Generate a versioned handoff snapshot for seamless session continuation.

Usage:
    python scripts/handoff_snapshot.py
    python scripts/handoff_snapshot.py --output data/artifacts/handoff/latest.md
    python scripts/handoff_snapshot.py --max-runs 5 --max-commits 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Project root for local imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.ops.artifact_store import ArtifactStore
from src.ops.run_manifest import RunRecorder


def _run_command(args: list[str], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = result.stdout.strip()
    return output if output else None


def _run_git(args: list[str], repo_root: Path) -> str | None:
    return _run_command(["git", *args], repo_root)


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sort_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple:
        started_at = item.get("started_at")
        return (started_at or "", item.get("run_id") or "")

    return sorted(runs, key=sort_key, reverse=True)


def _collect_pipeline_runs(pipeline_dir: Path, max_runs: int) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not pipeline_dir.exists():
        return runs

    for child in pipeline_dir.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _load_json(manifest_path)
        if not manifest:
            continue
        runs.append(
            {
                "run_id": manifest.get("run_id", child.name),
                "status": manifest.get("status"),
                "started_at": manifest.get("started_at"),
                "finished_at": manifest.get("finished_at"),
                "error": manifest.get("error"),
                "run_log_path": manifest.get("run_log_path"),
                "manifest_path": str(manifest_path),
            }
        )
    runs = _sort_runs(runs)
    return runs[:max_runs]


def _format_list(lines: list[str], indent: str = "") -> str:
    return "\n".join(f"{indent}{line}" for line in lines)


def _collect_processes(repo_root: Path) -> list[str]:
    process_output = _run_command(
        [
            "pgrep",
            "-af",
            "uvicorn src.api.main:app|streamlit run src/frontend/streamlit_app.py|python tests/eval_ragas.py",
        ],
        repo_root,
    )
    if not process_output:
        return []
    return [line for line in process_output.splitlines() if line.strip()]


def _probe_ready(cfg: dict[str, Any]) -> dict[str, Any]:
    api_cfg = cfg.get("api", {})
    port = api_cfg.get("port", 8000)
    url = f"http://127.0.0.1:{port}/ready"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "url": url,
            "status": "ok",
            "payload": payload,
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "url": url,
            "status": "http_error",
            "code": exc.code,
            "body": body[:400],
        }
    except Exception as exc:
        return {
            "url": url,
            "status": "unavailable",
            "error": str(exc),
        }


def _load_eval_results(repo_root: Path) -> dict[str, Any] | None:
    results_path = repo_root / "tests" / "ragas_results.json"
    payload = _load_json(results_path)
    if isinstance(payload, dict):
        payload["results_path"] = str(results_path)
        return payload
    return None


def _summarize_eval_log(repo_root: Path, max_lines: int = 12) -> dict[str, Any] | None:
    log_path = repo_root / "logs" / "ragas_eval.log"
    if not log_path.exists():
        return None

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    interesting = [
        line
        for line in lines
        if any(
            marker in line
            for marker in (
                "WARNING",
                "ERROR",
                "Exception raised in Job",
                "Failed to parse output",
                "RAGAS 评估结果",
            )
        )
    ]
    counts = {
        "call_failed": sum("ragas_llm_call_failed" in line for line in lines),
        "fallback_empty": sum("ragas_llm_fallback_empty" in line for line in lines),
        "thread_timeout": sum("ragas_llm_thread_timeout" in line for line in lines),
        "structured_disable": sum(
            "ragas_llm_disable_structured_output" in line for line in lines
        ),
        "job_exceptions": sum("Exception raised in Job" in line for line in lines),
        "parse_failures": sum("Failed to parse output" in line for line in lines),
    }
    return {
        "log_path": str(log_path),
        "counts": counts,
        "recent_lines": interesting[-max_lines:],
    }


def _collect_repo_state(
    *,
    repo_root: Path,
    artifacts_root: Path,
    max_runs: int,
    max_commits: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).astimezone()
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root) or "unknown"
    commit = _run_git(["rev-parse", "HEAD"], repo_root) or "unknown"
    status = _run_git(["status", "--short"], repo_root)
    diffstat = _run_git(["diff", "--stat"], repo_root)
    recent_commits = _run_git(["log", "--oneline", f"-n{max_commits}"], repo_root)

    artifacts_root = artifacts_root.resolve()
    pipelines = sorted(
        [p for p in artifacts_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    ) if artifacts_root.exists() else []

    runtime = {
        "ready": _probe_ready(cfg),
        "processes": _collect_processes(repo_root),
    }
    evaluation = {
        "results": _load_eval_results(repo_root),
        "log_summary": _summarize_eval_log(repo_root),
    }

    return {
        "generated_at": now.isoformat(),
        "repo_root": str(repo_root),
        "branch": branch,
        "commit": commit,
        "status": status,
        "diffstat": diffstat,
        "recent_commits": recent_commits,
        "pipelines": pipelines,
        "runtime": runtime,
        "evaluation": evaluation,
        "artifacts_root": str(artifacts_root),
        "max_runs": max_runs,
    }


def _render_snapshot(state: dict[str, Any]) -> str:
    repo_root = Path(state["repo_root"])
    artifacts_root = Path(state["artifacts_root"])
    max_runs = state["max_runs"]
    lines: list[str] = []

    lines.append("# Handoff Snapshot")
    lines.append("")
    lines.append(f"- Generated at: {state['generated_at']}")
    lines.append(f"- Repo root: {repo_root}")
    lines.append(f"- Branch: {state['branch']}")
    lines.append(f"- Commit: {state['commit']}")
    lines.append("")

    lines.append("## Working Tree")
    if state["status"]:
        lines.append("```")
        lines.append(state["status"])
        lines.append("```")
    else:
        lines.append("Clean")
    lines.append("")

    lines.append("## Diff Summary")
    if state["diffstat"]:
        lines.append("```")
        lines.append(state["diffstat"])
        lines.append("```")
    else:
        lines.append("No local diff")
    lines.append("")

    lines.append("## Recent Commits")
    if state["recent_commits"]:
        lines.append("```")
        lines.append(state["recent_commits"])
        lines.append("```")
    else:
        lines.append("No git log available")
    lines.append("")

    lines.append("## Runtime")
    ready = state["runtime"]["ready"]
    lines.append(f"- readiness url: {ready.get('url')}")
    lines.append(f"- readiness status: {ready.get('status')}")
    if ready.get("payload") is not None:
        lines.append("```json")
        lines.append(json.dumps(ready["payload"], ensure_ascii=False, indent=2))
        lines.append("```")
    elif ready.get("error"):
        lines.append(f"- readiness error: {ready['error']}")
    elif ready.get("body"):
        lines.append("```")
        lines.append(ready["body"])
        lines.append("```")

    processes = state["runtime"]["processes"]
    lines.append("- active processes:")
    if processes:
        lines.append("```")
        lines.extend(processes)
        lines.append("```")
    else:
        lines.append("none")
    lines.append("")

    lines.append("## Evaluation")
    results = state["evaluation"]["results"]
    if results:
        lines.append(f"- results path: {results.get('results_path')}")
        metric_lines = []
        for key in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ):
            value = results.get(key)
            if value is None:
                continue
            metric_lines.append(f"{key}: {value}")
        metric_lines.extend(
            [
                f"llm_model: {results.get('llm_model')}",
                f"llm_base_url: {results.get('llm_base_url')}",
                f"llm_error_count: {results.get('llm_error_count')}",
                f"llm_error_types: {results.get('llm_error_types')}",
            ]
        )
        lines.append(_format_list(metric_lines, indent="  - "))
    else:
        lines.append("- results: none")

    log_summary = state["evaluation"]["log_summary"]
    if log_summary:
        lines.append(f"- eval log: {log_summary.get('log_path')}")
        lines.append("  - error counters:")
        counter_lines = [
            f"{key}: {value}" for key, value in log_summary["counts"].items()
        ]
        lines.append(_format_list(counter_lines, indent="    - "))
        recent_lines = log_summary.get("recent_lines") or []
        lines.append("  - recent warnings/errors:")
        if recent_lines:
            lines.append("```")
            lines.extend(recent_lines)
            lines.append("```")
        else:
            lines.append("    none")
    else:
        lines.append("- eval log: none")
    lines.append("")

    lines.append("## Artifacts")
    pipelines = state["pipelines"]
    if not pipelines:
        lines.append(f"No artifact pipelines found under {artifacts_root}")
    else:
        for pipeline_dir in pipelines:
            lines.append(f"### {pipeline_dir.name}")
            current_path = pipeline_dir / "current.json"
            if current_path.exists():
                current = _load_json(current_path) or {}
                run_id = current.get("run_id", "unknown")
                activated_at = current.get("activated_at", "unknown")
                lines.append(f"- current run: {run_id}")
                lines.append(f"- activated at: {activated_at}")
                outputs = current.get("outputs", [])
                if outputs:
                    lines.append("- outputs:")
                    output_lines = [
                        f"{item.get('name')}: {item.get('artifact_path')} -> {item.get('active_path')}"
                        for item in outputs
                    ]
                    lines.append(_format_list(output_lines, indent="  - "))
                else:
                    lines.append("- outputs: none")
            else:
                lines.append("- current run: none")

            runs = _collect_pipeline_runs(pipeline_dir, max_runs)
            if not runs:
                lines.append("- recent runs: none")
                lines.append("")
                continue

            lines.append("- recent runs:")
            for run in runs:
                run_line = (
                    f"  - {run.get('run_id')} | {run.get('status')} | "
                    f"started={run.get('started_at')} | finished={run.get('finished_at')}"
                )
                lines.append(run_line)
                if run.get("error"):
                    lines.append(f"    error: {run.get('error')}")
                if run.get("run_log_path"):
                    lines.append(f"    log: {run.get('run_log_path')}")
                if run.get("manifest_path"):
                    lines.append(f"    manifest: {run.get('manifest_path')}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_snapshot_run(
    *,
    output_path: Path,
    repo_root: Path,
    artifacts_root: Path,
    max_runs: int,
    max_commits: int,
    cfg: dict[str, Any],
    config_path: str,
) -> Path:
    resolved_output_path = output_path
    if not resolved_output_path.is_absolute():
        resolved_output_path = repo_root / resolved_output_path
    resolved_summary_path = resolved_output_path.with_suffix(".json")

    recorder = RunRecorder.start(
        pipeline="handoff",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=config_path,
        repo_root=repo_root,
        inputs={
            "output_path": str(resolved_output_path),
            "summary_path": str(resolved_summary_path),
            "max_runs": max_runs,
            "max_commits": max_commits,
        },
    )
    store = ArtifactStore(artifacts_root)

    try:
        state = _collect_repo_state(
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            max_runs=max_runs,
            max_commits=max_commits,
            cfg=cfg,
        )
        snapshot_text = _render_snapshot(state)

        snapshot_artifact = recorder.run_dir / "snapshot.md"
        snapshot_artifact.write_text(snapshot_text, encoding="utf-8")

        summary_artifact = recorder.run_dir / "summary.json"
        summary_artifact.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        run_log = recorder.run_dir / "run.log"
        run_log.write_text(
            "\n".join(
                [
                    f"generated_at={state['generated_at']}",
                    f"branch={state['branch']}",
                    f"commit={state['commit']}",
                    f"ready_status={state['runtime']['ready'].get('status')}",
                    f"active_processes={len(state['runtime']['processes'])}",
                    f"eval_results_present={bool(state['evaluation']['results'])}",
                    f"eval_log_present={bool(state['evaluation']['log_summary'])}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        outputs = store.activate(
            pipeline="handoff",
            run_id=recorder.run_id,
            outputs=[
                {
                    "name": "snapshot.md",
                    "artifact_path": str(snapshot_artifact),
                    "active_path": str(resolved_output_path),
                    "activate_on_publish": True,
                },
                {
                    "name": "summary.json",
                    "artifact_path": str(summary_artifact),
                    "active_path": str(resolved_summary_path),
                    "activate_on_publish": True,
                },
            ],
        )

        recorder.update_stats(
            ready_status=state["runtime"]["ready"].get("status"),
            active_processes=len(state["runtime"]["processes"]),
            eval_results_present=bool(state["evaluation"]["results"]),
            eval_log_present=bool(state["evaluation"]["log_summary"]),
        )
        for output in outputs:
            recorder.add_output(output)
        recorder.mark_success()
    except Exception as exc:
        recorder.mark_failure(str(exc))
        raise

    return resolved_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate handoff snapshot")
    parser.add_argument(
        "--output",
        default="data/artifacts/handoff/latest.md",
        help="Markdown output path",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=5,
        help="Max recent runs per pipeline",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=8,
        help="Max recent commits to list",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file for artifacts_root",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    try:
        cfg = load_config(args.config)
    except Exception:
        cfg = {}
    artifacts_root = Path(cfg.get("ops", {}).get("artifacts_root", "data/artifacts"))

    output_path = write_snapshot_run(
        output_path=Path(args.output),
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        max_runs=args.max_runs,
        max_commits=args.max_commits,
        cfg=cfg,
        config_path=args.config,
    )
    print(output_path)


if __name__ == "__main__":
    main()
