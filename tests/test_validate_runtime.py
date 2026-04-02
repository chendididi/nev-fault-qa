"""运行时校验脚本的轻量单测。"""

from __future__ import annotations

import scripts.validate_runtime as validate_runtime


def test_validate_health_payload_schema():
    ok, err = validate_runtime.validate_health_payload({"status": "ok", "version": "0.1.0"})
    assert ok is True
    assert err is None

    bad_ok, bad_err = validate_runtime.validate_health_payload({"status": "degraded", "version": "0.1.0"})
    assert bad_ok is False
    assert "status" in bad_err


def test_validate_ready_payload_schema():
    payload = {
        "status": "ok",
        "version": "0.1.0",
        "checks": {
            "chunks_file_present": True,
            "bm25_loaded": True,
            "qwen_loaded": True,
            "milvus_connected": False,
            "neo4j_connected": True,
        },
    }
    ok, err = validate_runtime.validate_ready_payload(payload)
    assert ok is True
    assert err is None

    broken = {"status": "ok", "version": "0.1.0", "checks": {"chunks_file_present": True}}
    bad_ok, bad_err = validate_runtime.validate_ready_payload(broken)
    assert bad_ok is False
    assert "缺失字段" in bad_err


def test_validate_chat_payload_schema():
    ok, err = validate_runtime.validate_chat_payload(
        {"session_id": "abc", "answer": "ok", "sources": []}
    )
    assert ok is True
    assert err is None

    bad_ok, bad_err = validate_runtime.validate_chat_payload(
        {"session_id": "", "answer": "ok", "sources": []}
    )
    assert bad_ok is False
    assert "session_id" in bad_err


def test_build_api_command_uses_host_and_port():
    parser = validate_runtime.build_parser()
    args = parser.parse_args(["--spawn-api", "--host", "0.0.0.0", "--port", "8011"])

    cmd = validate_runtime._build_api_command(args)

    assert cmd[-4:] == ["--host", "0.0.0.0", "--port", "8011"]
