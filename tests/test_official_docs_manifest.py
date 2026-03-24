"""官方文档 manifest 结构测试。"""

import json
from pathlib import Path


def test_official_docs_manifest_has_expected_shape():
    manifest_path = Path("data/raw/official/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["entries"]
    ids = {entry["id"] for entry in manifest["entries"]}
    assert len(ids) == len(manifest["entries"])
    assert any(entry["brand"] == "tesla" and entry["format"] == "html" for entry in manifest["entries"])
    assert any(entry["brand"] == "byd" and entry["format"] == "pdf" for entry in manifest["entries"])
    for entry in manifest["entries"]:
        assert entry["official"] is True
        assert entry["target_path"].startswith("data/raw/official/")
