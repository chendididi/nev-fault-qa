"""
电路图 OCR + 描述生成

对 PDF 中图像占比超过阈值的页面，截取图像后交给 Qwen2-VL-7B 进行：
1. OCR：识别图中的线号、元件编号、信号名称
2. 描述：生成自然语言描述，说明电路连接关系

输出文本块格式与 pdf_parser 保持一致，可直接送入向量索引。
"""

from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger

from src.llm.qwen_client import QwenClient


_OCR_PROMPT = """这是一张新能源汽车维修手册中的电路图或接线图。请完成以下两项任务：

1. **OCR识别**：列出图中所有可见的文字，包括：
   - 元件编号（如 ECU、BMS、继电器编号）
   - 线束颜色和线号
   - 信号名称、端子编号
   - 电压/电阻标注

2. **电路描述**：用简洁的中文描述该电路图的连接关系和功能，格式如：
   "[元件A] 通过 [线号/颜色] 连接至 [元件B]，用于 [功能]"

请直接输出，不要重复题目要求。"""


def _get_page_image_ratio(page: fitz.Page) -> float:
    """计算页面中图像面积占页面总面积的比例。"""
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return 0.0
    image_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 1]
    image_area = sum(
        (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
        for b in image_blocks
    )
    return image_area / page_area


def extract_circuit_descriptions(
    pdf_path: str | Path,
    image_ratio_threshold: float = 0.3,
    qwen_client: QwenClient | None = None,
) -> list[dict]:
    """
    扫描 PDF，对含电路图的页面生成 OCR + 文字描述文本块。

    Args:
        pdf_path: PDF 文件路径
        image_ratio_threshold: 图像面积占页面比例阈值，超过则触发 OCR
        qwen_client: 复用外部 QwenClient 实例，None 则内部创建

    Returns:
        List of chunk dicts（与 pdf_parser 格式一致）:
            - chunk_id, text, source, page, chapter="circuit_diagram"
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    client = qwen_client or QwenClient()
    doc = fitz.open(str(pdf_path))
    chunks: list[dict] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        ratio = _get_page_image_ratio(page)

        if ratio < image_ratio_threshold:
            continue

        logger.info(f"第 {page_num + 1} 页图像占比 {ratio:.1%}，触发电路图 OCR")

        # 将页面渲染为高分辨率图像（300 DPI）
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        try:
            description = client.describe_image(img_bytes, _OCR_PROMPT)
        except Exception as e:
            logger.warning(f"第 {page_num + 1} 页 OCR 失败: {e}")
            continue

        import hashlib
        chunk_id = hashlib.md5(f"{pdf_path.stem}_{page_num}_circuit".encode()).hexdigest()[:12]
        chunks.append({
            "chunk_id": chunk_id,
            "text": description,
            "source": pdf_path.name,
            "page": page_num + 1,
            "chapter": "circuit_diagram",
        })

    doc.close()
    logger.info(f"电路图 OCR 完成，共生成 {len(chunks)} 个文本块")
    return chunks
