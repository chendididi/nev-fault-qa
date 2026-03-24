"""
下载官方公开文档到 data/raw/official。

默认只下载 PDF。Tesla 官方 service manual 为 HTML 站点，需显式传 --include-html
才会保存入口页面快照；要真正用于维修手册建库，后续还需要 HTML ingestion / crawler。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


MANIFEST_PATH = Path("data/raw/official/manifest.json")


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_entries(manifest: dict, brands: set[str] | None, include_html: bool) -> list[dict]:
    entries = manifest["entries"]
    entries = [entry for entry in entries if entry.get("download_enabled", True)]
    if brands:
        entries = [entry for entry in entries if entry["brand"] in brands]
    if not include_html:
        entries = [entry for entry in entries if entry["format"] == "pdf"]
    return entries


def download_entry(entry: dict, *, dry_run: bool = False, force: bool = False) -> dict:
    target_path = Path(entry["target_path"])
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        return {
            "id": entry["id"],
            "status": "skipped",
            "target_path": str(target_path),
            "reason": "exists",
        }

    if dry_run:
        return {
            "id": entry["id"],
            "status": "planned",
            "target_path": str(target_path),
        }

    request = Request(
        entry["url"],
        headers={
            "User-Agent": "nev-fault-qa-downloader/0.1",
        },
    )
    with urlopen(request, timeout=60) as response:
        target_path.write_bytes(response.read())

    metadata_path = target_path.with_suffix(target_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "id": entry["id"],
                "url": entry["url"],
                "brand": entry["brand"],
                "kind": entry["kind"],
                "format": entry["format"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "id": entry["id"],
        "status": "downloaded",
        "target_path": str(target_path),
    }


def main():
    parser = argparse.ArgumentParser(description="下载官方 Tesla/BYD 文档到 data/raw/official")
    parser.add_argument("--brand", action="append", choices=["tesla", "byd"], help="按品牌过滤，可重复传参")
    parser.add_argument("--include-html", action="store_true", help="同时保存 HTML 入口页快照")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行下载")
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    run = RunRecorder.start(
        pipeline="download_official_docs",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "brands": args.brand or ["tesla", "byd"],
            "include_html": args.include_html,
            "dry_run": args.dry_run,
            "force": args.force,
        },
    )
    setup_logging(cfg, component="download_official_docs", run_log_path=run.run_dir / "run.log")

    try:
        manifest = load_manifest()
        brands = set(args.brand) if args.brand else None
        entries = iter_entries(manifest, brands, args.include_html)
        run.update_stats(entry_count=len(entries))

        results = []
        failures = []
        with log_context(run_id=run.run_id, pipeline="download_official_docs"):
            for entry in entries:
                try:
                    result = download_entry(entry, dry_run=args.dry_run, force=args.force)
                    results.append(result)
                    logger.info(f"{result['status']}: {entry['id']} -> {result['target_path']}")
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    failures.append({"id": entry["id"], "error": str(exc)})
                    logger.warning(f"下载失败 {entry['id']}: {exc}")

            result_path = run.run_dir / "download_results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "results": results,
                        "failures": failures,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            run.add_output(
                {
                    "name": "download_results.json",
                    "artifact_path": str(result_path),
                    "active_path": None,
                    "activate_on_publish": False,
                }
            )
            run.update_stats(downloaded=len(results), failed=len(failures))

        if failures and not args.dry_run:
            run.mark_failure(f"{len(failures)} 个文档下载失败")
            sys.exit(1)

        run.mark_success()
    except Exception as exc:
        logger.exception("download_official_docs_failed")
        run.mark_failure(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
