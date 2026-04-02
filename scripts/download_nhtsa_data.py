"""
下载 NHTSA Manufacturer Communications 数据，并可转成 chunks JSON 供 build_index 直接摄取。

默认下载最新 5 年区间（当前为 2025-2026）的 CSV/TSV 压缩包，并生成：
data/raw/official/nhtsa/chunks/nhtsa_mfr_comms_chunks.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data_processing.chunk_contract import validate_chunks
from src.ops.logging_setup import log_context, setup_logging
from src.ops.run_manifest import RunRecorder


BASE_URL = "https://static.nhtsa.gov/odi/ffdd/tsbs"
DEFAULT_YEAR_RANGE = "2025-2026"
YEAR_RANGE_PATTERN = re.compile(r"^\d{4}-\d{4}$")


def _validate_year_ranges(year_ranges: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for year_range in year_ranges:
        value = year_range.strip()
        if not YEAR_RANGE_PATTERN.fullmatch(value):
            raise ValueError(f"非法 --year-range: {year_range}，格式应为 YYYY-YYYY")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _download_file(url: str, target_path: Path, *, force: bool = False, dry_run: bool = False) -> dict:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        return {"status": "skipped", "url": url, "path": str(target_path), "reason": "exists"}
    if dry_run:
        return {"status": "planned", "url": url, "path": str(target_path)}

    request = Request(url, headers={"User-Agent": "nev-fault-qa-nhtsa-downloader/0.1"})
    with urlopen(request, timeout=120) as response:
        target_path.write_bytes(response.read())
    return {"status": "downloaded", "url": url, "path": str(target_path)}


def _truncate_text(text: str, max_length: int = 4500) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 3].rstrip() + "..."


def _iter_csv_rows_from_zip(zip_path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"{zip_path} 中未找到 CSV 文件")
        member = members[0]
        with archive.open(member) as f:
            text_stream = io.TextIOWrapper(f, encoding="utf-8", errors="ignore", newline="")
            reader = csv.DictReader(text_stream)
            return list(reader)


def build_chunks_from_csv_archives(
    csv_archives: list[Path],
    *,
    makes: set[str] | None = None,
    max_chunks: int = 0,
) -> list[dict]:
    chunks: list[dict] = []
    page = 1
    makes_upper = {item.upper() for item in makes} if makes else None

    for archive_path in csv_archives:
        rows = _iter_csv_rows_from_zip(archive_path)
        source_name = archive_path.name

        for row in rows:
            make = (row.get("Make") or "").strip()
            if makes_upper and make.upper() not in makes_upper:
                continue

            tsb_id = (row.get("TSB/Document ID") or "").strip()
            model = (row.get("Model") or "").strip()
            model_year = (row.get("Model Year") or "").strip()
            summary = _truncate_text((row.get("Concise Summary") or "").strip())
            if not summary:
                continue

            text = "\n".join(
                [
                    "来源: NHTSA Manufacturer Communications",
                    f"TSB/Document ID: {tsb_id or 'N/A'}",
                    f"Make: {make or 'N/A'}",
                    f"Model: {model or 'N/A'}",
                    f"Model Year: {model_year or 'N/A'}",
                    f"Concise Summary: {summary}",
                ]
            )
            raw = f"{source_name}|{tsb_id}|{make}|{model}|{model_year}|{page}"
            chunk_id = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": source_name,
                    "page": page,
                    "chapter": "nhtsa_manufacturer_communication",
                }
            )
            page += 1
            if max_chunks and len(chunks) >= max_chunks:
                return chunks

    return chunks


def _build_download_plan(
    year_ranges: list[str],
    *,
    with_csv: bool,
    with_tsv: bool,
) -> list[tuple[str, Path]]:
    plan: list[tuple[str, Path]] = []
    root = Path("data/raw/official/nhtsa")
    if with_csv:
        for year_range in year_ranges:
            file_name = f"MFR_COMMS_RECEIVED_{year_range}.zip"
            plan.append((f"{BASE_URL}/{file_name}", root / "csv" / file_name))
    if with_tsv:
        for year_range in year_ranges:
            file_name = f"TSBS_RECEIVED_{year_range}.zip"
            plan.append((f"{BASE_URL}/{file_name}", root / "tsv" / file_name))
    plan.append((f"{BASE_URL}/TSBS.txt", root / "TSBS.txt"))
    return plan


def main():
    parser = argparse.ArgumentParser(description="下载 NHTSA Manufacturer Communications 数据")
    parser.add_argument(
        "--year-range",
        action="append",
        dest="year_ranges",
        help="NHTSA 5 年分段，如 2025-2026；可重复传参",
    )
    parser.add_argument("--make", action="append", dest="makes", help="生成 chunks 时按 Make 过滤，可重复")
    parser.add_argument("--max-chunks", type=int, default=0, help="生成 chunks 的最大条数（0=不限制）")
    parser.add_argument("--skip-csv", action="store_true", help="不下载 MFR_COMMS CSV 压缩包")
    parser.add_argument("--skip-tsv", action="store_true", help="不下载 TSBS TSV 压缩包")
    parser.add_argument("--no-build-chunks", action="store_true", help="只下载，不生成 chunks JSON")
    parser.add_argument(
        "--chunks-output",
        default="data/raw/official/nhtsa/chunks/nhtsa_mfr_comms_chunks.json",
        help="chunks JSON 输出路径",
    )
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    parser.add_argument("--dry-run", action="store_true", help="只输出计划，不执行下载与转换")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    year_ranges = _validate_year_ranges(args.year_ranges or [DEFAULT_YEAR_RANGE])
    with_csv = not args.skip_csv
    with_tsv = not args.skip_tsv
    build_chunks = with_csv and not args.no_build_chunks
    if not with_csv and not with_tsv:
        raise ValueError("至少保留一种下载类型（CSV 或 TSV）")

    cfg = load_config(args.config)
    artifacts_root = cfg.get("ops", {}).get("artifacts_root", "data/artifacts")
    run = RunRecorder.start(
        pipeline="download_nhtsa_data",
        artifacts_root=artifacts_root,
        config=cfg,
        config_path=args.config,
        inputs={
            "year_ranges": year_ranges,
            "makes": args.makes or [],
            "max_chunks": args.max_chunks,
            "with_csv": with_csv,
            "with_tsv": with_tsv,
            "build_chunks": build_chunks,
            "chunks_output": args.chunks_output,
            "force": args.force,
            "dry_run": args.dry_run,
        },
    )
    setup_logging(cfg, component="download_nhtsa_data", run_log_path=run.run_dir / "run.log")

    download_plan = _build_download_plan(year_ranges, with_csv=with_csv, with_tsv=with_tsv)
    run.update_stats(planned_downloads=len(download_plan))

    try:
        download_results: list[dict] = []
        failures: list[dict] = []
        with log_context(run_id=run.run_id, pipeline="download_nhtsa_data"):
            for url, target_path in download_plan:
                try:
                    result = _download_file(url, target_path, force=args.force, dry_run=args.dry_run)
                    download_results.append(result)
                    logger.info(f"{result['status']}: {url} -> {target_path}")
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    failures.append({"url": url, "path": str(target_path), "error": str(exc)})
                    logger.warning(f"下载失败: {url} ({exc})")

            download_result_path = run.run_dir / "download_results.json"
            download_result_path.write_text(
                json.dumps({"results": download_results, "failures": failures}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            run.add_output(
                {
                    "name": "download_results.json",
                    "artifact_path": str(download_result_path),
                    "active_path": None,
                    "activate_on_publish": False,
                }
            )
            run.update_stats(
                downloaded=sum(1 for item in download_results if item["status"] == "downloaded"),
                skipped=sum(1 for item in download_results if item["status"] == "skipped"),
                failed=len(failures),
            )

            if failures and not args.dry_run:
                run.mark_failure(f"NHTSA 下载失败 {len(failures)} 项")
                sys.exit(1)

            if build_chunks and not args.dry_run:
                csv_archives = [
                    Path(item["path"])
                    for item in download_results
                    if item["status"] in ("downloaded", "skipped")
                    and Path(item["path"]).name.startswith("MFR_COMMS_RECEIVED_")
                ]
                chunks = build_chunks_from_csv_archives(
                    csv_archives,
                    makes=set(args.makes or []),
                    max_chunks=args.max_chunks,
                )
                if not chunks:
                    raise ValueError("NHTSA CSV 未生成任何 chunks，请检查 --make 过滤条件或输入数据")
                validate_chunks(chunks)

                chunks_path = Path(args.chunks_output)
                chunks_path.parent.mkdir(parents=True, exist_ok=True)
                chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
                run.add_output(
                    {
                        "name": "nhtsa_mfr_comms_chunks.json",
                        "artifact_path": str(chunks_path),
                        "active_path": None,
                        "activate_on_publish": False,
                    }
                )
                run.update_stats(chunk_count=len(chunks), chunks_path=str(chunks_path))
                logger.info(f"NHTSA chunks 已生成: {chunks_path} ({len(chunks)} 条)")

            run.mark_success()
    except Exception as exc:
        logger.exception("download_nhtsa_data_failed")
        run.mark_failure(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
