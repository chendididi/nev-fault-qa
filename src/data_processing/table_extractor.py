"""
故障码定义表提取器

使用 pdfplumber 识别并提取 PDF 中的结构化表格（故障码、DTC描述、可能原因等）。
输出为标准化 JSON，供知识图谱实体抽取使用。
"""

import json
import re
from pathlib import Path

import pdfplumber
from loguru import logger


# 故障码正则：P/B/C/U + 4位十六进制
_FAULT_CODE_RE = re.compile(r"\b([PBCU]\d{4})\b", re.IGNORECASE)

# 表格中常见的故障码相关列名关键词
_FAULT_COL_KEYWORDS = {"故障码", "dtc", "fault code", "代码", "code"}
_DESC_COL_KEYWORDS = {"描述", "definition", "故障描述", "说明", "description"}
_CAUSE_COL_KEYWORDS = {"原因", "cause", "可能原因", "possible cause"}


def _identify_column_roles(headers: list[str]) -> dict[str, int]:
    """
    根据列名关键词识别表格列角色，返回 {role: col_index} 映射。
    role 可为: 'fault_code', 'description', 'cause'
    """
    roles: dict[str, int] = {}
    for i, h in enumerate(headers):
        h_lower = h.lower().strip()
        if any(kw in h_lower for kw in _FAULT_COL_KEYWORDS):
            roles.setdefault("fault_code", i)
        if any(kw in h_lower for kw in _DESC_COL_KEYWORDS):
            roles.setdefault("description", i)
        if any(kw in h_lower for kw in _CAUSE_COL_KEYWORDS):
            roles.setdefault("cause", i)
    return roles


def extract_tables(pdf_path: str | Path) -> list[dict]:
    """
    从 PDF 中提取所有包含故障码的表格。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        List of dicts, 每条记录包含:
            - fault_code: str
            - description: str
            - cause: str（可能为空）
            - source: str（文件名）
            - page: int
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    logger.info(f"提取表格: {pdf_path.name}")
    records: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                if not table or len(table) < 2:
                    continue

                headers = [str(c).strip() if c else "" for c in table[0]]
                roles = _identify_column_roles(headers)

                # 若无法识别故障码列，尝试扫描每行是否含故障码
                for row in table[1:]:
                    row_strs = [str(c).strip() if c else "" for c in row]
                    row_text = " ".join(row_strs)

                    fault_codes = _FAULT_CODE_RE.findall(row_text)
                    if not fault_codes:
                        continue

                    for fc in fault_codes:
                        record = {
                            "fault_code": fc.upper(),
                            "description": row_strs[roles["description"]] if "description" in roles else "",
                            "cause": row_strs[roles["cause"]] if "cause" in roles else "",
                            "source": pdf_path.name,
                            "page": page_num,
                        }
                        records.append(record)

    logger.info(f"提取到 {len(records)} 条故障码定义记录")
    return records


def extract_tables_to_json(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
) -> list[dict]:
    """
    提取表格并可选地保存为 JSON 文件。

    Args:
        pdf_path: PDF 路径
        output_path: 输出 JSON 路径，None 则不保存

    Returns:
        提取的记录列表
    """
    records = extract_tables(pdf_path)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"表格数据已保存至: {output_path}")
    return records
