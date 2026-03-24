"""
抓取 Tesla 官方 service manual HTML 页面到本地目录。

使用方法：
    python scripts/fetch_tesla_service_manual.py --manual model3_2024_en_us --max-pages 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data_processing.html_manual_parser import extract_manual_links
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


MANUALS = {
    "model3_2024_en_us": {
        "seed_file": "data/raw/official/tesla/html/model3-service-manual-2024-index.html",
        "base_url": "https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/index.html",
        "output_dir": "data/raw/official/tesla/model3_2024_en_us",
    }
}


def _download_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "nev-fault-qa-tesla-crawler/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def main():
    parser = argparse.ArgumentParser(description="抓取 Tesla service manual HTML 页面")
    parser.add_argument("--manual", choices=sorted(MANUALS), default="model3_2024_en_us")
    parser.add_argument("--max-pages", type=int, default=200, help="最多下载的 HTML 页面数")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    run = RunRecorder.start(
        pipeline="fetch_tesla_service_manual",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={"manual": args.manual, "max_pages": args.max_pages},
    )
    setup_logging(cfg, component="fetch_tesla_service_manual", run_log_path=run.run_dir / "run.log")

    manual = MANUALS[args.manual]
    seed_file = Path(manual["seed_file"])
    output_dir = Path(manual["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        seed_html = seed_file.read_text(encoding="utf-8")
        links = extract_manual_links(seed_html, manual["base_url"])
        if manual["base_url"] not in links:
            links.insert(0, manual["base_url"])
        selected_links = links[: args.max_pages] if args.max_pages else links
        run.update_stats(discovered_links=len(links), selected_links=len(selected_links))

        downloaded: list[dict] = []
        with log_context(run_id=run.run_id, pipeline="fetch_tesla_service_manual"):
            for index, url in enumerate(selected_links, start=1):
                file_name = url.rsplit("/", 1)[-1]
                local_path = output_dir / file_name
                if local_path.exists():
                    downloaded.append({"url": url, "path": str(local_path), "status": "skipped"})
                    continue

                html_text = _download_text(url)
                local_path.write_text(html_text, encoding="utf-8")
                downloaded.append({"url": url, "path": str(local_path), "status": "downloaded"})
                if index % 25 == 0:
                    logger.info(f"抓取进度 {index}/{len(selected_links)}")

            crawl_manifest = output_dir / "crawl_manifest.json"
            crawl_manifest.write_text(
                json.dumps(downloaded, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            run.add_output(
                {
                    "name": "crawl_manifest.json",
                    "artifact_path": str(crawl_manifest),
                    "active_path": None,
                    "activate_on_publish": False,
                }
            )
            run.update_stats(
                downloaded_count=sum(1 for item in downloaded if item["status"] == "downloaded"),
                skipped_count=sum(1 for item in downloaded if item["status"] == "skipped"),
            )
            run.mark_success()
    except Exception as exc:
        logger.exception("fetch_tesla_service_manual_failed")
        run.mark_failure(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
