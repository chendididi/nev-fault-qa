import csv
import io
import zipfile
from pathlib import Path

from scripts.download_nhtsa_data import build_chunks_from_csv_archives


def _write_csv_zip(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=["TSB/Document ID", "Make", "Model", "Model Year", "Concise Summary"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(path.stem + ".csv", csv_buffer.getvalue())


def test_build_chunks_from_csv_archives_filters_make_and_applies_limit(tmp_path):
    archive_path = tmp_path / "MFR_COMMS_RECEIVED_2025-2026.zip"
    _write_csv_zip(
        archive_path,
        [
            {
                "TSB/Document ID": "111",
                "Make": "TESLA",
                "Model": "MODEL 3",
                "Model Year": "2024",
                "Concise Summary": "Rear camera intermittent issue",
            },
            {
                "TSB/Document ID": "222",
                "Make": "BYD",
                "Model": "HAN",
                "Model Year": "2024",
                "Concise Summary": "Battery warning message after OTA",
            },
        ],
    )

    chunks = build_chunks_from_csv_archives(
        [archive_path],
        makes={"TESLA"},
        max_chunks=1,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["source"] == archive_path.name
    assert chunk["chapter"] == "nhtsa_manufacturer_communication"
    assert chunk["page"] == 1
    assert "TESLA" in chunk["text"]
    assert "MODEL 3" in chunk["text"]

