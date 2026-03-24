"""Shared logging bootstrap for API and offline pipelines."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from loguru import logger


_LOG_CONTEXT: ContextVar[dict[str, object]] = ContextVar("log_context", default={})
_DEFAULT_COMPONENT = "app"


def _patch_record(record: dict) -> None:
    extra = dict(_LOG_CONTEXT.get())
    extra.update(record["extra"])
    extra.setdefault("component", _DEFAULT_COMPONENT)
    record["extra"] = extra


def setup_logging(
    config: dict | None = None,
    *,
    component: str = "app",
    run_log_path: str | Path | None = None,
) -> None:
    """Configure loguru once per process."""
    global _DEFAULT_COMPONENT
    _DEFAULT_COMPONENT = component

    logging_cfg = (config or {}).get("logging", {})
    level = logging_cfg.get("level", "INFO")
    file_path = Path(run_log_path or logging_cfg.get("file", "logs/nev_qa.log"))
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.configure(patcher=_patch_record)
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
        enqueue=True,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{extra[component]} | {message} | {extra}"
        ),
    )
    logger.add(
        file_path,
        level=level,
        rotation=logging_cfg.get("rotation", "10 MB"),
        serialize=bool(logging_cfg.get("file_json", True)),
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )


@contextmanager
def log_context(**values: object) -> Iterator[None]:
    """Temporarily attach contextual fields to all logs on the current context."""
    current = dict(_LOG_CONTEXT.get())
    current.update({key: value for key, value in values.items() if value is not None})
    token = _LOG_CONTEXT.set(current)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)

