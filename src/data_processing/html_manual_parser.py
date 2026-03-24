"""Tesla service manual HTML parsing helpers."""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from loguru import logger


_BLOCK_TAGS = {
    "article",
    "div",
    "section",
    "p",
    "li",
    "ul",
    "ol",
    "table",
    "tr",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
}
_IGNORED_TAGS = {"script", "style", "svg", "noscript"}


class _ArticleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.article_depth = 0
        self.ignored_depth = 0
        self.in_h1 = False
        self.title_parts: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "article":
            self.article_depth += 1
        if self.article_depth and tag in _IGNORED_TAGS:
            self.ignored_depth += 1
        if self.article_depth and tag == "h1":
            self.in_h1 = True
        if self.article_depth and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.article_depth and tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if self.article_depth and tag == "h1":
            self.in_h1 = False
        if self.article_depth and tag in _IGNORED_TAGS and self.ignored_depth:
            self.ignored_depth -= 1
        if tag == "article" and self.article_depth:
            self.article_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.article_depth or self.ignored_depth:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned:
            return
        self.parts.append(cleaned)
        if self.in_h1:
            self.title_parts.append(cleaned)


class _ManualLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if href:
            self.hrefs.append(href)


def extract_manual_links(html_text: str, base_url: str) -> list[str]:
    """Extract same-manual HTML links from a Tesla service manual page."""
    parser = _ManualLinkParser()
    parser.feed(html_text)
    base = urlparse(base_url)
    base_prefix = base.path.rsplit("/", 1)[0] + "/"

    links: list[str] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        if not href or href.startswith("#"):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc != base.netloc:
            continue
        if not parsed.path.endswith(".html"):
            continue
        if not parsed.path.startswith(base_prefix):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)

    return links


def _make_chunk_id(source: str, idx: int) -> str:
    return hashlib.md5(f"{source}_{idx}".encode()).hexdigest()[:12]


def _chunk_text(text: str, source: str, chapter: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunks: list[dict] = []
    buffer = ""
    chunk_idx = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        buffer += line + "\n"
        if len(buffer) >= chunk_size:
            chunks.append(
                {
                    "chunk_id": _make_chunk_id(source, chunk_idx),
                    "text": buffer[:chunk_size].strip(),
                    "source": source,
                    "page": 1,
                    "chapter": chapter,
                }
            )
            chunk_idx += 1
            buffer = buffer[max(0, chunk_size - chunk_overlap):]

    if buffer.strip():
        chunks.append(
            {
                "chunk_id": _make_chunk_id(source, chunk_idx),
                "text": buffer.strip(),
                "source": source,
                "page": 1,
                "chapter": chapter,
            }
        )
    return chunks


def parse_html_file(
    html_path: str | Path,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """Parse a single saved Tesla service-manual HTML page into chunks."""
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML 文件不存在: {html_path}")

    parser = _ArticleTextParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip() or html_path.stem
    body = re.sub(r"\n{2,}", "\n", "\n".join(parser.parts)).strip()
    if not body:
        logger.warning(f"HTML 页面无可用正文: {html_path}")
        return []
    return _chunk_text(body, html_path.name, title, chunk_size, chunk_overlap)


def parse_html_dir(
    dir_path: str | Path,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """Parse all HTML files under a local Tesla service manual directory."""
    dir_path = Path(dir_path)
    html_files = sorted(dir_path.glob("*.html"))
    logger.info(f"发现 {len(html_files)} 个 HTML 文件")
    chunks: list[dict] = []
    for html_file in html_files:
        chunks.extend(
            parse_html_file(
                html_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks

