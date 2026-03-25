import json
from pathlib import Path

from scripts.handoff_snapshot import write_snapshot_run


def test_handoff_snapshot_publishes_versioned_artifacts(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "tests").mkdir()
    (repo_root / "logs").mkdir()
    artifacts_root = repo_root / "data" / "artifacts"

    (repo_root / "tests" / "ragas_results.json").write_text(
        json.dumps(
            {
                "faithfulness": 0.5,
                "answer_relevancy": 0.8,
                "context_precision": 0.6,
                "context_recall": 0.7,
                "llm_model": "gpt-5.4",
                "llm_base_url": "https://cmdme.cn/v1",
                "llm_error_count": 2,
                "llm_error_types": {"APITimeoutError": 2},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (repo_root / "logs" / "ragas_eval.log").write_text(
        "\n".join(
            [
                "2026-03-25 WARNING ragas_llm_call_failed attempt=1",
                "2026-03-25 ERROR ragas_llm_fallback_empty",
                "Exception raised in Job[1]: TimeoutError()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    build_index_run = artifacts_root / "build_index" / "build_index-1"
    build_index_run.mkdir(parents=True)
    (build_index_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "build_index-1",
                "status": "succeeded",
                "started_at": "2026-03-25T00:00:00+00:00",
                "finished_at": "2026-03-25T00:01:00+00:00",
                "run_log_path": str(build_index_run / "run.log"),
                "outputs": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    git_outputs = {
        "rev-parse --abbrev-ref HEAD": "feat/test",
        "rev-parse HEAD": "abc123",
        "status --short": " M local.txt",
        "diff --stat": " local.txt | 1 +\n 1 file changed, 1 insertion(+)",
        "log --oneline -n4": "abc123 test commit",
    }

    monkeypatch.setattr(
        "scripts.handoff_snapshot._run_git",
        lambda args, repo: git_outputs.get(" ".join(args)),
    )
    monkeypatch.setattr(
        "scripts.handoff_snapshot._probe_ready",
        lambda cfg: {
            "url": "http://127.0.0.1:8000/ready",
            "status": "ok",
            "payload": {"status": "ok"},
        },
    )
    monkeypatch.setattr(
        "scripts.handoff_snapshot._collect_processes",
        lambda repo: ["111 uvicorn src.api.main:app --port 8000"],
    )

    output_path = write_snapshot_run(
        output_path=Path("data/artifacts/handoff/latest.md"),
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        max_runs=3,
        max_commits=4,
        cfg={"ops": {"artifacts_root": str(artifacts_root)}, "api": {"port": 8000}},
        config_path="config/config.yaml",
    )

    assert output_path == repo_root / "data" / "artifacts" / "handoff" / "latest.md"
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()

    current = json.loads((artifacts_root / "handoff" / "current.json").read_text())
    run_id = current["run_id"]
    assert run_id.startswith("handoff-")

    manifest = json.loads(
        (artifacts_root / "handoff" / run_id / "manifest.json").read_text()
    )
    assert manifest["status"] == "succeeded"
    output_names = {item["name"] for item in manifest["outputs"]}
    assert output_names == {"snapshot.md", "summary.json"}

    snapshot_text = output_path.read_text(encoding="utf-8")
    assert "## Runtime" in snapshot_text
    assert "## Evaluation" in snapshot_text
    assert "answer_relevancy: 0.8" in snapshot_text
    assert "APITimeoutError" in snapshot_text
