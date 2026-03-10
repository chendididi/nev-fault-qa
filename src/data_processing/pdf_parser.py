"""
PDF 文本分块解析器

使用 PyMuPDF 提取文本，保留章节层级结构，按配置的 chunk_size 切片。
输出格式：List[dict]，每个 dict 包含 text、page、chapter、chunk_id 等字段。
"""

import re
import hashlib
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from loguru import logger


# 章节标题识别正则（适配常见维修手册格式）
_CHAPTER_PATTERNS = [
    re.compile(r"^第\s*[一二三四五六七八九十百\d]+\s*[章节]"),  # 第X章/节
    re.compile(r"^\d+(\.\d+)*\s+\S"),                           # 1.2.3 标题
    re.compile(r"^[A-Z]\d+\s+-\s+"),                            # P0300 - 故障码标题
]


def _is_chapter_title(line: str) -> bool:
    """判断一行文本是否为章节标题。"""
    line = line.strip()
    if not line:
        return False
    return any(p.match(line) for p in _CHAPTER_PATTERNS)


def _make_chunk_id(source: str, page: int, idx: int) -> str:
    """生成唯一 chunk ID（source文件名 + 页码 + 索引 的哈希前缀）。"""
    raw = f"{source}_{page}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def parse_pdf(
    pdf_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    解析 PDF，返回带结构元数据的文本块列表。

    Args:
        pdf_path: PDF 文件路径
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块的重叠字符数（保证上下文连贯）

    Returns:
        List of dicts:
            - chunk_id: str
            - text: str
            - source: str（文件名）
            - page: int（起始页码，1-based）
            - chapter: str（所属章节标题，无则为""）
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    logger.info(f"开始解析 PDF: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))

    chunks: list[dict] = []
    current_chapter = ""
    buffer = ""
    buffer_page = 1
    chunk_idx = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")

        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # 更新当前章节
            if _is_chapter_title(stripped):
                current_chapter = stripped

            buffer += stripped + "\n"

            # 达到 chunk_size 时切片
            if len(buffer) >= chunk_size:
                chunk = {
                    "chunk_id": _make_chunk_id(pdf_path.stem, buffer_page, chunk_idx),
                    "text": buffer[:chunk_size].strip(),
                    "source": pdf_path.name,
                    "page": buffer_page,
                    "chapter": current_chapter,
                }
                chunks.append(chunk)
                chunk_idx += 1
                # 保留重叠部分
                buffer = buffer[chunk_size - chunk_overlap:]
                buffer_page = page_num + 1

        buffer_page = page_num + 1

    # 处理最后一块（不足 chunk_size）
    if buffer.strip():
        chunks.append({
            "chunk_id": _make_chunk_id(pdf_path.stem, buffer_page, chunk_idx),
            "text": buffer.strip(),
            "source": pdf_path.name,
            "page": buffer_page,
            "chapter": current_chapter,
        })

    doc.close()
    logger.info(f"解析完成：{len(chunks)} 个文本块")
    return chunks


def parse_pdf_dir(
    dir_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Generator[dict, None, None]:
    """
    批量解析目录下所有 PDF，逐个 yield chunk。

    Args:
        dir_path: 包含 PDF 的目录
        chunk_size: 块大小
        chunk_overlap: 重叠大小
    """
    dir_path = Path(dir_path)
    pdfs = list(dir_path.glob("*.pdf"))
    logger.info(f"发现 {len(pdfs)} 个 PDF 文件")
    for pdf in pdfs:
        yield from parse_pdf(pdf, chunk_size, chunk_overlap)
