"""Tesla HTML manual parser tests."""

from pathlib import Path

from src.data_processing.html_manual_parser import (
    extract_manual_links,
    parse_html_file,
)


def test_extract_manual_links_filters_same_manual_html():
    html = """
    <html><body>
      <a href="index.html">Index</a>
      <a href="GUID-1.html">Topic 1</a>
      <a href="GUID-2.html">Topic 2</a>
      <a href="https://example.com/outside.html">Outside</a>
      <a href="#fragment">Fragment</a>
    </body></html>
    """
    links = extract_manual_links(
        html,
        "https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/index.html",
    )

    assert len(links) == 3
    assert links[0].endswith("index.html")
    assert all(link.startswith("https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/") for link in links)


def test_parse_html_file_extracts_article_chunks(tmp_path: Path):
    html_file = tmp_path / "GUID-1.html"
    html_file.write_text(
        """
        <html><body>
          <aside>nav</aside>
          <article>
            <h1>Charge Port Voltage Check</h1>
            <div class="body">
              <p>Use a multimeter to verify charge port voltage.</p>
              <p>Inspect the connector and harness before replacement.</p>
            </div>
          </article>
        </body></html>
        """,
        encoding="utf-8",
    )

    chunks = parse_html_file(html_file, chunk_size=80, chunk_overlap=10)

    assert chunks
    assert chunks[0]["chapter"] == "Charge Port Voltage Check"
    assert chunks[0]["source"] == "GUID-1.html"
    assert "multimeter" in chunks[0]["text"]
