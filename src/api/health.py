"""Health and readiness helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request


router = APIRouter()


def _safe_check(component: Any, method_name: str) -> bool:
    if component is None:
        return False
    method = getattr(component, method_name, None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            return False
    return True


def build_liveness_payload(version: str) -> dict[str, str]:
    return {"status": "ok", "version": version}


def build_readiness_payload(app: Any) -> dict[str, Any]:
    state = app.state
    chunks_file = Path(getattr(state, "chunks_file_path", "data/processed/chunks.json"))

    checks = {
        "chunks_file_present": chunks_file.exists(),
        "bm25_loaded": bool(
            getattr(getattr(state, "bm25_retriever", None), "is_built", False)
        ),
        "qwen_loaded": getattr(state, "qwen", None) is not None,
        "milvus_connected": _safe_check(
            getattr(state, "embedding_retriever", None),
            "ping",
        ),
        "neo4j_connected": _safe_check(getattr(state, "neo4j", None), "ping"),
    }
    status = "ok" if all(checks.values()) else "degraded"
    return {
        "status": status,
        "version": app.version,
        "checks": checks,
    }


@router.get("/health")
async def health_check(request: Request) -> dict[str, str]:
    return build_liveness_payload(request.app.version)


@router.get("/ready")
async def readiness_check(request: Request) -> dict[str, Any]:
    return build_readiness_payload(request.app)
