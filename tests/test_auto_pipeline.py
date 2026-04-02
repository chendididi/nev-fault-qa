import argparse
import json
from pathlib import Path

import scripts.auto_pipeline as auto_pipeline


def test_load_hooks_supports_nested_and_inline_specs(tmp_path):
    hooks_file = tmp_path / "hooks.json"
    hooks_file.write_text(
        json.dumps(
            {
                "hooks": {
                    "pipeline.pre": "echo pipeline_pre",
                    "build_chunks": {
                        "pre": ["echo chunks_pre_a", "echo chunks_pre_b"],
                        "post": "echo chunks_post",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    hooks = auto_pipeline.load_hooks(
        str(hooks_file),
        inline_specs=["build_chunks.pre=echo chunks_pre_inline"],
    )

    assert hooks["pipeline.pre"] == ["echo pipeline_pre"]
    assert hooks["build_chunks.post"] == ["echo chunks_post"]
    assert hooks["build_chunks.pre"] == [
        "echo chunks_pre_a",
        "echo chunks_pre_b",
        "echo chunks_pre_inline",
    ]


def test_build_stage_plan_default_and_heavy_flags():
    parser = auto_pipeline.build_parser()

    default_args = parser.parse_args([])
    default_plan = auto_pipeline._build_stage_plan(default_args, "python")
    assert [stage.name for stage in default_plan] == [
        "preflight",
        "check",
        "download_docs",
        "fetch_manual_html",
        "build_chunks",
        "handoff",
    ]

    heavy_args = parser.parse_args(["--with-embedding", "--with-graph"])
    heavy_plan = auto_pipeline._build_stage_plan(heavy_args, "python")
    assert [stage.name for stage in heavy_plan] == [
        "preflight",
        "check",
        "download_docs",
        "fetch_manual_html",
        "build_chunks",
        "build_vector",
        "build_graph",
        "handoff",
    ]


def test_build_stage_plan_with_nhtsa_stage():
    parser = auto_pipeline.build_parser()
    args = parser.parse_args(["--with-nhtsa", "--nhtsa-year-range", "2025-2026", "--nhtsa-make", "TESLA"])
    plan = auto_pipeline._build_stage_plan(args, "python")

    assert [stage.name for stage in plan] == [
        "preflight",
        "check",
        "download_docs",
        "download_nhtsa",
        "fetch_manual_html",
        "build_chunks",
        "handoff",
    ]
    nhtsa_stage = next(stage for stage in plan if stage.name == "download_nhtsa")
    assert "--year-range" in nhtsa_stage.command
    assert "2025-2026" in nhtsa_stage.command
    assert "--make" in nhtsa_stage.command
    assert "TESLA" in nhtsa_stage.command


def test_run_pipeline_executes_hooks_and_stage_commands_in_order(tmp_path, monkeypatch):
    artifacts_root = tmp_path / "artifacts"
    cfg = {
        "ops": {"artifacts_root": str(artifacts_root)},
        "logging": {"file": str(tmp_path / "logs" / "auto_pipeline.log")},
    }
    monkeypatch.setattr(auto_pipeline, "load_config", lambda _path: cfg)

    calls: list[tuple[str, str]] = []

    def _fake_subprocess_run(command, **kwargs):
        if isinstance(command, list) and command[:2] == ["git", "rev-parse"]:
            class _GitResult:
                returncode = 0
                stdout = "abc123\n"

            return _GitResult()

        command_text = command if isinstance(command, str) else " ".join(command)
        call_type = "hook" if kwargs.get("shell") else "stage"
        calls.append((call_type, command_text))

        class _Result:
            returncode = 0
            stdout = ""

        return _Result()

    monkeypatch.setattr(auto_pipeline.subprocess, "run", _fake_subprocess_run)

    parser = auto_pipeline.build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(tmp_path / "dummy.yaml"),
            "--skip-check",
            "--skip-download",
            "--skip-fetch-html",
            "--skip-handoff",
            "--hook",
            "pipeline.pre=echo pipeline_pre",
            "--hook",
            "build_chunks.pre=echo chunks_pre",
            "--hook",
            "build_chunks.post=echo chunks_post",
        ]
    )

    exit_code = auto_pipeline.run_pipeline(args)
    assert exit_code == 0

    assert calls == [
        ("hook", "echo pipeline_pre"),
        ("hook", "echo chunks_pre"),
        (
            "stage",
            f"{Path(auto_pipeline.sys.executable or 'python')} scripts/build_index.py --config "
            f"{tmp_path / 'dummy.yaml'} --input data/raw/official --skip-embedding",
        ),
        ("hook", "echo chunks_post"),
    ]

    auto_pipeline_runs = artifacts_root / "auto_pipeline"
    run_dirs = [path for path in auto_pipeline_runs.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    stage_results = json.loads((run_dirs[0] / "stage_results.json").read_text(encoding="utf-8"))
    assert [item["stage"] for item in stage_results] == ["preflight", "build_chunks"]
    assert all(item["status"] == "succeeded" for item in stage_results)
