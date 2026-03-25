"""
Generate a handoff snapshot for seamless session continuation.

Usage:
    python scripts/handoff_snapshot.py
    python scripts/handoff_snapshot.py --output data/artifacts/handoff/latest.md
    python scripts/handoff_snapshot.py --max-runs 5 --max-commits 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys


# Project root for local imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config


def _run_git(args: list[str], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = result.stdout.strip()
    return output if output else None


def _load_json(path: Path) -> dict | None:
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


def _write_snapshot(
    *,
    output_path: Path,
    repo_root: Path,
    artifacts_root: Path,
    max_runs: int,
    max_commits: int,
) -> Path:
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
    )

    lines: list[str] = []
    lines.append("# Handoff Snapshot")
    lines.append("")
    lines.append(f"- Generated at: {now.isoformat()}")
    lines.append(f"- Repo root: {repo_root}")
    lines.append(f"- Branch: {branch}")
    lines.append(f"- Commit: {commit}")
    lines.append("")

    lines.append("## Working Tree")
    if status:
        lines.append("```")
        lines.append(status)
        lines.append("```")
    else:
        lines.append("Clean")
    lines.append("")

    lines.append("## Diff Summary")
    if diffstat:
        lines.append("```")
        lines.append(diffstat)
        lines.append("```")
    else:
        lines.append("No local diff")
    lines.append("")

    lines.append("## Recent Commits")
    if recent_commits:
        lines.append("```")
        lines.append(recent_commits)
        lines.append("```")
    else:
        lines.append("No git log available")
    lines.append("")

    lines.append("## Artifacts")
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


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

    output_path = _write_snapshot(
        output_path=Path(args.output),
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        max_runs=args.max_runs,
        max_commits=args.max_commits,
    )
    print(output_path)


if __name__ == "__main__":
    main()
