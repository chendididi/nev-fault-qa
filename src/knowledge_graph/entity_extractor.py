"""
知识图谱实体抽取

使用 Qwen2-VL-7B 从文本块中抽取结构化实体：
- FaultCode（故障码）：P0300, U0100, ...
- Component（零部件）：电机控制器, 电池管理系统, ...
- Symptom（故障现象）：车辆抖动, 无法启动, ...
- Subsystem（子系统）：驱动系统, 制动系统, ...
- Tool（维修工具）：诊断仪, 万用表, ...
"""

import json
import re
from typing import Any

from loguru import logger

from src.llm.qwen_client import QwenClient


# 故障码正则（预过滤，减少 LLM 调用次数）
_FAULT_CODE_RE = re.compile(r"\b([PBCU]\d{4})\b", re.IGNORECASE)

_EXTRACTION_PROMPT = """请从以下新能源汽车维修手册文本中抽取结构化实体，以 JSON 格式输出。

文本：
{text}

请输出以下格式的 JSON（不存在的类型输出空列表）：
{{
  "fault_codes": ["P0300", ...],           // 故障码
  "components": ["电机控制器", ...],        // 零部件名称
  "symptoms": ["车辆抖动", ...],           // 故障现象描述
  "subsystems": ["驱动系统", ...],          // 子系统
  "tools": ["诊断仪", ...]                 // 维修工具
}}

只输出 JSON，不要其他内容。"""


def extract_entities(text: str, client: QwenClient) -> dict[str, list[str]]:
    """
    从单个文本块中抽取实体。

    Args:
        text: 输入文本
        client: QwenClient 实例

    Returns:
        dict with keys: fault_codes, components, symptoms, subsystems, tools
    """
    # 预过滤：无故障码且文本过短，跳过
    has_fault_code = bool(_FAULT_CODE_RE.search(text))
    if not has_fault_code and len(text) < 50:
        return {"fault_codes": [], "components": [], "symptoms": [], "subsystems": [], "tools": []}

    prompt = _EXTRACTION_PROMPT.format(text=text[:1500])  # 截断防超长

    try:
        response = client.generate(prompt)
        # 提取 JSON 部分（LLM 有时会有额外文字）
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError("未找到 JSON 输出")
        return json.loads(json_match.group())
    except Exception as e:
        logger.warning(f"实体抽取失败: {e}，使用正则兜底")
        # 兜底：仅返回正则匹配的故障码
        fault_codes = _FAULT_CODE_RE.findall(text)
        return {
            "fault_codes": [fc.upper() for fc in fault_codes],
            "components": [],
            "symptoms": [],
            "subsystems": [],
            "tools": [],
        }


def batch_extract_entities(
    chunks: list[dict],
    client: QwenClient,
    progress_callback: Any = None,
) -> list[dict]:
    """
    批量对文本块列表进行实体抽取，结果附加到每个 chunk 的 entities 字段。

    Args:
        chunks: pdf_parser 输出的 chunk 列表
        client: QwenClient 实例
        progress_callback: 可选进度回调 callback(current, total)

    Returns:
        chunks 列表，每个 chunk 新增 entities 字段
    """
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk["entities"] = extract_entities(chunk["text"], client)
        if progress_callback:
            progress_callback(i + 1, total)

    logger.info(f"完成 {total} 个文本块的实体抽取")
    return chunks
