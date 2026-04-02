from pathlib import Path

from scripts.fetch_tesla_service_manual import _load_seed_html, _resolve_seed_file


def test_resolve_seed_file_prefers_existing_candidate(tmp_path):
    missing = tmp_path / "missing.html"
    existing = tmp_path / "existing.html"
    existing.write_text("<html>seed</html>", encoding="utf-8")

    manual = {"seed_candidates": [str(missing), str(existing)]}
    assert _resolve_seed_file(manual) == existing


def test_load_seed_html_downloads_when_missing(tmp_path, monkeypatch):
    seed_file = tmp_path / "seed" / "index.html"
    monkeypatch.setattr(
        "scripts.fetch_tesla_service_manual._download_text",
        lambda _url: "<html>downloaded</html>",
    )

    html_text, source = _load_seed_html(seed_file, "https://example.com/index.html")

    assert source == "downloaded_from_base_url"
    assert html_text == "<html>downloaded</html>"
    assert seed_file.read_text(encoding="utf-8") == "<html>downloaded</html>"


def test_load_seed_html_uses_local_file_when_present(tmp_path):
    seed_file = tmp_path / "seed.html"
    seed_file.write_text("<html>local</html>", encoding="utf-8")

    html_text, source = _load_seed_html(seed_file, "https://example.com/index.html")

    assert source == "local"
    assert html_text == "<html>local</html>"

