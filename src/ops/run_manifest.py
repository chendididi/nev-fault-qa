"""Versioned run manifest handling for offline pipelines."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def new_run_id(pipeline: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{pipeline}-{timestamp}-{suffix}"


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


@dataclass
class RunRecorder:
    """Persist structured metadata for a pipeline run."""

    pipeline: str
    run_id: str
    run_dir: Path
    manifest_path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        *,
        pipeline: str,
        artifacts_root: str | Path,
        config: dict | None = None,
        config_path: str | Path | None = None,
        repo_root: str | Path | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> "RunRecorder":
        repo_root_path = Path(repo_root or Path(__file__).resolve().parents[2])
        run_id = new_run_id(pipeline)
        run_dir = Path(artifacts_root) / pipeline / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"

        data: dict[str, Any] = {
            "pipeline": pipeline,
            "run_id": run_id,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "repo_root": str(repo_root_path),
            "git_commit": _git_commit(repo_root_path),
            "config_path": str(config_path) if config_path else None,
            "config_snapshot_path": None,
            "run_log_path": str(run_dir / "run.log"),
            "inputs": inputs or {},
            "outputs": [],
            "stats": {},
            "error": None,
            "notes": [],
        }

        recorder = cls(
            pipeline=pipeline,
            run_id=run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
            data=data,
        )

        if config is not None:
            config_snapshot = run_dir / "config.snapshot.json"
            config_snapshot.write_text(
                json.dumps(config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            recorder.data["config_snapshot_path"] = str(config_snapshot)

        recorder.write()
        return recorder

    def write(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_output(self, output: dict[str, Any]) -> None:
        self.data["outputs"].append(output)
        self.write()

    def update_inputs(self, **inputs: Any) -> None:
        self.data["inputs"].update(inputs)
        self.write()

    def update_stats(self, **stats: Any) -> None:
        self.data["stats"].update(stats)
        self.write()

    def note(self, message: str) -> None:
        self.data["notes"].append(message)
        self.write()

    def mark_success(self) -> None:
        self.data["status"] = "succeeded"
        self.data["finished_at"] = utc_now()
        self.write()

    def mark_failure(self, error: str) -> None:
        self.data["status"] = "failed"
        self.data["finished_at"] = utc_now()
        self.data["error"] = error
        self.write()

