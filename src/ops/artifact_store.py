"""Helpers for versioned artifact publishing and rollback."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.ops.run_manifest import utc_now


class ArtifactStore:
    """Manage pipeline artifacts and the active version pointer."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def pipeline_root(self, pipeline: str) -> Path:
        return self.root_dir / pipeline

    def run_dir(self, pipeline: str, run_id: str) -> Path:
        return self.pipeline_root(pipeline) / run_id

    def current_path(self, pipeline: str) -> Path:
        return self.pipeline_root(pipeline) / "current.json"

    def get_current(self, pipeline: str) -> dict[str, Any] | None:
        current_path = self.current_path(pipeline)
        if not current_path.exists():
            return None
        return json.loads(current_path.read_text(encoding="utf-8"))

    def list_runs(self, pipeline: str) -> list[str]:
        pipeline_root = self.pipeline_root(pipeline)
        if not pipeline_root.exists():
            return []
        runs = [
            child.name
            for child in pipeline_root.iterdir()
            if child.is_dir() and (child / "manifest.json").exists()
        ]
        return sorted(runs)

    def publish_json(
        self,
        *,
        pipeline: str,
        run_id: str,
        artifact_name: str,
        payload: Any,
        active_path: str | Path | None = None,
    ) -> dict[str, Any]:
        output = self.stage_json(
            pipeline=pipeline,
            run_id=run_id,
            artifact_name=artifact_name,
            payload=payload,
            active_path=active_path,
        )
        return self.activate(
            pipeline=pipeline,
            run_id=run_id,
            outputs=[output],
        )[0]

    def stage_json(
        self,
        *,
        pipeline: str,
        run_id: str,
        artifact_name: str,
        payload: Any,
        active_path: str | Path | None = None,
    ) -> dict[str, Any]:
        run_dir = self.run_dir(pipeline, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = run_dir / artifact_name
        artifact_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "name": artifact_name,
            "artifact_path": str(artifact_path),
            "active_path": str(active_path) if active_path else None,
            "activate_on_publish": bool(active_path),
        }

    def activate(
        self,
        *,
        pipeline: str,
        run_id: str,
        outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        current = self.get_current(pipeline) or {}
        activated_outputs: list[dict[str, Any]] = []

        for output in outputs:
            artifact_path = Path(output["artifact_path"])
            active_path_value = output.get("active_path")
            active_path = Path(active_path_value) if active_path_value else None
            if active_path is not None:
                active_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(artifact_path, active_path)
            activated_outputs.append(
                {
                    **output,
                    "artifact_path": str(artifact_path),
                    "active_path": str(active_path) if active_path else None,
                    "activated_at": utc_now(),
                }
            )

        current_payload = {
            "pipeline": pipeline,
            "run_id": run_id,
            "previous_run_id": current.get("run_id"),
            "activated_at": utc_now(),
            "outputs": activated_outputs,
        }
        current_path = self.current_path(pipeline)
        current_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.write_text(
            json.dumps(current_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return activated_outputs

    def activate_run(self, pipeline: str, run_id: str) -> dict[str, Any]:
        manifest_path = self.run_dir(pipeline, run_id) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"未找到 run manifest: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = [
            output
            for output in manifest.get("outputs", [])
            if output.get("activate_on_publish")
        ]
        if not outputs:
            raise ValueError(f"pipeline {pipeline} 的 run {run_id} 没有可激活产物")

        activated_outputs = self.activate(
            pipeline=pipeline,
            run_id=run_id,
            outputs=outputs,
        )
        return {
            "pipeline": pipeline,
            "run_id": run_id,
            "outputs": activated_outputs,
        }
