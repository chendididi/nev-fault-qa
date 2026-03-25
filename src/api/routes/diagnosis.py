"""
故障诊断专用接口

POST /api/v1/diagnosis/fault-code
  - 输入故障码，返回完整诊断信息（知识图谱 + RAG）

POST /api/v1/diagnosis/symptom
  - 输入故障现象描述，返回可能的故障码和排查建议
"""

from time import perf_counter

from fastapi import APIRouter, Request
from loguru import logger
from pydantic import BaseModel

from src.ops.logging_setup import log_context

router = APIRouter()


class FaultCodeRequest(BaseModel):
    fault_code: str                    # 如 "P0300"


class SymptomRequest(BaseModel):
    symptom: str                       # 如 "发动机抖动，转速不稳"
    top_k: int = 5


class DiagnosisResult(BaseModel):
    fault_code: str
    components: list[str]
    subsystems: list[str]
    symptoms: list[str]
    rag_answer: str
    sources: list[dict]


@router.post("/diagnosis/fault-code", response_model=DiagnosisResult)
async def diagnose_by_fault_code(req: FaultCodeRequest, request: Request):
    """
    按故障码查询：知识图谱 + RAG 联合诊断。
    """
    state = request.app.state
    fc = req.fault_code.upper().strip()
    request_logger = logger.bind(route="/api/v1/diagnosis/fault-code", fault_code=fc)

    with log_context(route="/api/v1/diagnosis/fault-code", fault_code=fc):
        total_start = perf_counter()
        # 知识图谱查询
        graph_start = perf_counter()
        graph_data = state.neo4j.query_fault_code(fc)
        graph_ms = int((perf_counter() - graph_start) * 1000)
        request_logger.bind(graph_components=len(graph_data.get("components", []))).info("fault_code_graph_loaded")

        # RAG 检索
        query = f"故障码 {fc} 的含义、原因和处理方法"
        retrieval_start = perf_counter()
        candidates, retrieval_stats = state.hybrid_retriever.retrieve_with_stats(query, top_k=20)
        retrieve_ms = int((perf_counter() - retrieval_start) * 1000)
        rerank_start = perf_counter()
        top_chunks = state.reranker.rerank(query, candidates, top_k=5)
        rerank_ms = int((perf_counter() - rerank_start) * 1000)

        # 生成答案
        answer_start = perf_counter()
        answer = state.answer_generator.generate(
            query=query,
            chunks=top_chunks,
            graph_data=graph_data,
        )
        answer_ms = int((perf_counter() - answer_start) * 1000)
        sources = state.answer_generator.format_sources(top_chunks)
        request_logger.bind(source_count=len(sources)).info("fault_code_diagnosis_completed")
        total_ms = int((perf_counter() - total_start) * 1000)

        logger.bind(
            fault_code=fc,
            rerank_input_count=len(candidates),
            rerank_output_count=len(top_chunks),
            graph_components=len(graph_data.get("components", [])),
            graph_subsystems=len(graph_data.get("subsystems", [])),
            graph_symptoms=len(graph_data.get("symptoms", [])),
            graph_ms=graph_ms,
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            answer_ms=answer_ms,
            total_ms=total_ms,
            **retrieval_stats,
        ).info("diagnosis_trace")

    return DiagnosisResult(
        fault_code=fc,
        components=graph_data.get("components", []),
        subsystems=graph_data.get("subsystems", []),
        symptoms=graph_data.get("symptoms", []),
        rag_answer=answer,
        sources=sources,
    )


class SymptomResult(BaseModel):
    possible_fault_codes: list[str]
    rag_answer: str
    sources: list[dict]


@router.post("/diagnosis/symptom", response_model=SymptomResult)
async def diagnose_by_symptom(req: SymptomRequest, request: Request):
    """
    按故障现象查询：知识图谱推断可能故障码 + RAG 排查建议。
    """
    state = request.app.state
    request_logger = logger.bind(route="/api/v1/diagnosis/symptom")

    with log_context(route="/api/v1/diagnosis/symptom"):
        total_start = perf_counter()
        # 知识图谱：从故障现象推断可能故障码
        graph_start = perf_counter()
        keywords = req.symptom.split()
        graph_results = state.neo4j.query_symptom_fault_codes(keywords)
        graph_ms = int((perf_counter() - graph_start) * 1000)
        possible_codes = list({r["fault_code"] for r in graph_results})[:req.top_k]
        request_logger.bind(possible_fault_codes=possible_codes).info("symptom_graph_loaded")

        # RAG 检索
        rewritten, _ = state.query_processor.process(req.symptom)
        retrieval_start = perf_counter()
        candidates, retrieval_stats = state.hybrid_retriever.retrieve_with_stats(rewritten, top_k=20)
        retrieve_ms = int((perf_counter() - retrieval_start) * 1000)
        rerank_start = perf_counter()
        top_chunks = state.reranker.rerank(rewritten, candidates, top_k=5)
        rerank_ms = int((perf_counter() - rerank_start) * 1000)

        # 生成答案（不传 graph_data，由 RAG 自行处理）
        answer_start = perf_counter()
        answer = state.answer_generator.generate(
            query=req.symptom,
            chunks=top_chunks,
        )
        answer_ms = int((perf_counter() - answer_start) * 1000)
        sources = state.answer_generator.format_sources(top_chunks)
        request_logger.bind(source_count=len(sources)).info("symptom_diagnosis_completed")

        total_ms = int((perf_counter() - total_start) * 1000)
        logger.bind(
            rerank_input_count=len(candidates),
            rerank_output_count=len(top_chunks),
            graph_possible_fault_codes=len(possible_codes),
            graph_ms=graph_ms,
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            answer_ms=answer_ms,
            total_ms=total_ms,
            **retrieval_stats,
        ).info("diagnosis_trace")

    return SymptomResult(
        possible_fault_codes=possible_codes,
        rag_answer=answer,
        sources=sources,
    )
