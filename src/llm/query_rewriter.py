"""
Query 重写与意图识别

在检索前对用户输入进行处理：
1. 意图识别：区分"故障码查询"、"故障现象描述"、"操作步骤咨询"、"闲聊"
2. Query 扩展：为故障码查询补充可能的相关术语
3. Query 改写：将口语化描述规范化为维修手册术语
"""

import re
from enum import Enum

from loguru import logger

from src.llm.qwen_client import QwenClient


class QueryIntent(str, Enum):
    FAULT_CODE = "fault_code"          # 精确故障码查询，如 "P0300 是什么"
    SYMPTOM = "symptom"                # 故障现象描述，如 "车子启动抖动"
    PROCEDURE = "procedure"            # 操作步骤，如 "怎么更换电池"
    GENERAL = "general"                # 一般知识问答
    CHITCHAT = "chitchat"              # 闲聊


_FAULT_CODE_RE = re.compile(r"\b([PBCU]\d{4})\b", re.IGNORECASE)

_INTENT_PROMPT = """判断以下用户输入的意图类型，只输出一个单词：
- fault_code：输入包含故障码（如P0300、U0100）
- symptom：描述车辆故障现象
- procedure：询问维修操作步骤
- general：一般知识问答
- chitchat：闲聊

用户输入："{query}"

意图："""

_REWRITE_PROMPT = """你是一位经验丰富的新能源汽车技师。请将以下口语化的故障描述改写为更专业、规范的技术描述，
以便在维修手册中检索到相关信息。保持原意，补充可能的专业术语。

原始描述："{query}"

规范化描述（只输出改写结果，不要解释）："""


def _quick_detect_fault_code(query: str) -> bool:
    """快速检测 query 是否包含故障码（避免不必要的 LLM 调用）。"""
    return bool(_FAULT_CODE_RE.search(query))


def detect_intent(query: str, client: QwenClient | None = None) -> QueryIntent:
    """
    识别用户查询意图。

    优先使用正则快速判断，复杂情况才调用 LLM。

    Args:
        query: 用户输入
        client: QwenClient（仅复杂情况使用）

    Returns:
        QueryIntent 枚举值
    """
    # 快速规则判断
    if _quick_detect_fault_code(query):
        return QueryIntent.FAULT_CODE

    # 操作步骤关键词
    procedure_keywords = ["怎么", "如何", "步骤", "操作", "更换", "拆卸", "安装", "检测方法"]
    if any(kw in query for kw in procedure_keywords):
        return QueryIntent.PROCEDURE

    # 故障现象关键词
    symptom_keywords = ["抖动", "异响", "无法", "不能", "故障", "警告灯", "亮灯", "漏油", "过热"]
    if any(kw in query for kw in symptom_keywords):
        return QueryIntent.SYMPTOM

    # 无法快速判断时调用 LLM
    if client is not None:
        try:
            response = client.generate(_INTENT_PROMPT.format(query=query)).strip().lower()
            for intent in QueryIntent:
                if intent.value in response:
                    return intent
        except Exception as e:
            logger.warning(f"意图识别 LLM 调用失败: {e}")

    return QueryIntent.GENERAL


def rewrite_query(query: str, intent: QueryIntent, client: QwenClient) -> str:
    """
    根据意图对 Query 进行改写扩展。

    Args:
        query: 原始查询
        intent: 已识别的意图
        client: QwenClient 实例

    Returns:
        改写后的查询字符串
    """
    # 故障码查询无需改写
    if intent == QueryIntent.FAULT_CODE:
        codes = _FAULT_CODE_RE.findall(query)
        return " ".join([c.upper() for c in codes]) + " " + query

    # 故障现象改写
    if intent == QueryIntent.SYMPTOM:
        try:
            rewritten = client.generate(_REWRITE_PROMPT.format(query=query)).strip()
            logger.debug(f"Query 改写: {query!r} → {rewritten!r}")
            return rewritten
        except Exception as e:
            logger.warning(f"Query 改写失败: {e}，使用原始查询")
            return query

    return query


class QueryProcessor:
    """Query 预处理流水线：意图识别 + 改写。"""

    def __init__(self, client: QwenClient):
        self._client = client

    def process(self, query: str) -> tuple[str, QueryIntent]:
        """
        完整处理用户查询。

        Args:
            query: 原始用户输入

        Returns:
            (rewritten_query, intent) 元组
        """
        intent = detect_intent(query, self._client)
        rewritten = rewrite_query(query, intent, self._client)
        return rewritten, intent
