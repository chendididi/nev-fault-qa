"""
RAG 答案生成器

将检索到的上下文片段 + 知识图谱信息组装为 Prompt，调用 Qwen2-VL 生成最终答案。
支持：
- 引用溯源（标注来源文件和页码）
- 流式输出
- 多轮对话历史
"""

from typing import Generator

from loguru import logger

from src.llm.qwen_client import QwenClient
from src.llm.query_rewriter import QueryIntent


_SYSTEM_PROMPT = """你是一位专业的新能源汽车故障诊断助手，拥有丰富的维修经验。
请根据提供的维修手册内容和知识图谱信息，准确回答技师的问题。

回答要求：
1. 优先使用提供的参考资料，不要编造信息
2. 给出具体的诊断步骤或操作建议
3. 如果涉及安全操作，请特别提示
4. 在回答末尾标注信息来源（文件名和页码）
5. 使用中文回答"""

_RAG_PROMPT_TEMPLATE = """参考资料：
{context}

{graph_info}

用户问题：{query}

请根据以上资料回答，并在回答末尾注明引用来源。"""

_GRAPH_INFO_TEMPLATE = """知识图谱信息：
- 故障码 {fault_code} 相关零部件：{components}
- 所属子系统：{subsystems}
- 常见故障现象：{symptoms}
{possible_fault_codes_line}
"""


def _build_context(chunks: list[dict], max_chars: int = 3000) -> str:
    """
    将 top-k chunks 拼接为上下文字符串，并附加来源标注。

    Args:
        chunks: reranker 输出的 chunk 列表
        max_chars: 最大字符数（防超长）

    Returns:
        格式化的上下文字符串
    """
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        source_tag = f"[{i}] 来源：{chunk.get('source', '未知')} 第{chunk.get('page', '?')}页"
        part = f"{source_tag}\n{chunk['text']}\n"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n".join(parts)


def _build_graph_section(graph_data: dict | None) -> str:
    """将知识图谱查询结果格式化为文本。"""
    if not graph_data:
        return ""
    components = "、".join(graph_data.get("components", [])) or "无"
    subsystems = "、".join(graph_data.get("subsystems", [])) or "无"
    symptoms = "、".join(graph_data.get("symptoms", [])) or "无"
    fault_code = graph_data.get("fault_code", "")
    possible_fault_codes = graph_data.get("possible_fault_codes", [])

    if not any([graph_data.get("components"), graph_data.get("symptoms"), possible_fault_codes]):
        return ""

    return _GRAPH_INFO_TEMPLATE.format(
        fault_code=fault_code,
        components=components,
        subsystems=subsystems,
        symptoms=symptoms,
        possible_fault_codes_line=(
            f"- 现象匹配到的可能故障码：{'、'.join(possible_fault_codes)}"
            if possible_fault_codes
            else ""
        ),
    )


class AnswerGenerator:
    """RAG 答案生成器，整合检索结果和知识图谱信息。"""

    def __init__(self, client: QwenClient):
        """
        Args:
            client: QwenClient 实例
        """
        self._client = client

    def generate(
        self,
        query: str,
        chunks: list[dict],
        graph_data: dict | None = None,
        chat_history: list[dict] | None = None,
    ) -> str:
        """
        生成完整答案（非流式）。

        Args:
            query: 用户原始问题
            chunks: reranker 返回的 Top-K 上下文片段
            graph_data: neo4j_client.query_fault_code() 返回的知识图谱数据
            chat_history: 多轮对话历史（暂未使用，预留接口）

        Returns:
            生成的答案字符串
        """
        context = _build_context(chunks)
        graph_info = _build_graph_section(graph_data)

        prompt = _RAG_PROMPT_TEMPLATE.format(
            context=context,
            graph_info=graph_info,
            query=query,
        )

        logger.debug(f"Prompt 长度: {len(prompt)} 字符")
        answer = self._client.generate(prompt, system_prompt=_SYSTEM_PROMPT)
        return answer

    def stream_generate(
        self,
        query: str,
        chunks: list[dict],
        graph_data: dict | None = None,
        chat_history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        流式生成答案（适合 FastAPI SSE 输出）。

        Yields:
            逐步生成的文本片段
        """
        context = _build_context(chunks)
        graph_info = _build_graph_section(graph_data)

        prompt = _RAG_PROMPT_TEMPLATE.format(
            context=context,
            graph_info=graph_info,
            query=query,
        )

        yield from self._client.stream_generate(prompt, system_prompt=_SYSTEM_PROMPT)

    def format_sources(self, chunks: list[dict]) -> list[dict]:
        """
        提取并格式化答案的引用来源列表。

        Returns:
            List of dicts: [{source, page, chapter, snippet}, ...]
        """
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            sources.append({
                "ref_id": i,
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
                "chapter": chunk.get("chapter", ""),
                "snippet": chunk["text"][:150] + "...",
            })
        return sources
