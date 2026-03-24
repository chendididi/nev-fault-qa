"""Run manifest 与版本化产物测试。"""

import json

from src.ops.artifact_store import ArtifactStore
from src.ops.run_manifest import RunRecorder


def test_run_manifest_persists_config_and_outputs(tmp_path):
    recorder = RunRecorder.start(
        pipeline="build_index",
        artifacts_root=tmp_path,
        config={"logging": {"level": "INFO"}},
        config_path="config/config.yaml",
        repo_root=tmp_path,
        inputs={"input_dir": "data/sample"},
    )
    recorder.update_stats(chunk_count=5)
    recorder.add_output(
        {
            "name": "chunks.json",
            "artifact_path": str(recorder.run_dir / "chunks.json"),
            "active_path": "data/processed/chunks.json",
            "activate_on_publish": True,
        }
    )
    recorder.mark_success()

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["stats"]["chunk_count"] == 5
    assert manifest["config_snapshot_path"]
    assert manifest["outputs"][0]["name"] == "chunks.json"


def test_artifact_store_can_publish_and_rollback(tmp_path):
    store = ArtifactStore(tmp_path)
    active_path = tmp_path / "processed" / "chunks.json"

    first = store.publish_json(
        pipeline="build_index",
        run_id="build_index-20260325T000000Z-aaaa1111",
        artifact_name="chunks.json",
        payload=[{"chunk_id": "c1"}],
        active_path=active_path,
    )
    second = store.publish_json(
        pipeline="build_index",
        run_id="build_index-20260325T000100Z-bbbb2222",
        artifact_name="chunks.json",
        payload=[{"chunk_id": "c2"}],
        active_path=active_path,
    )

    first_manifest = {
        "outputs": [{**first, "activate_on_publish": True}],
    }
    second_manifest = {
        "outputs": [{**second, "activate_on_publish": True}],
    }
    (tmp_path / "build_index" / "build_index-20260325T000000Z-aaaa1111" / "manifest.json").write_text(
        json.dumps(first_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tmp_path / "build_index" / "build_index-20260325T000100Z-bbbb2222" / "manifest.json").write_text(
        json.dumps(second_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert json.loads(active_path.read_text(encoding="utf-8"))[0]["chunk_id"] == "c2"

    store.activate_run("build_index", "build_index-20260325T000000Z-aaaa1111")

    restored = json.loads(active_path.read_text(encoding="utf-8"))
    current = store.get_current("build_index")
    assert restored[0]["chunk_id"] == "c1"
    assert current["run_id"] == "build_index-20260325T000000Z-aaaa1111"
