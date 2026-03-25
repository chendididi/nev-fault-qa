"""Validation helpers for chunk artifacts."""

from __future__ import annotations


REQUIRED_FIELDS = ("chunk_id", "text", "source", "page", "chapter")
MAX_TEXT_LENGTH = 5000
MAX_SOURCE_LENGTH = 200
MAX_CHAPTER_LENGTH = 200
MAX_CHUNK_ID_LENGTH = 64


def validate_chunks(chunks: list[dict]) -> None:
    """Raise ValueError when chunks break the repository contract."""
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks 必须是非空列表")

    for index, chunk in enumerate(chunks):
        missing = [field for field in REQUIRED_FIELDS if field not in chunk]
        if missing:
            raise ValueError(f"chunk[{index}] 缺少字段: {', '.join(missing)}")
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError(f"chunk[{index}].text 必须是非空字符串")
        if not isinstance(chunk["source"], str) or not chunk["source"].strip():
            raise ValueError(f"chunk[{index}].source 必须是非空字符串")
        if not isinstance(chunk["page"], int):
            raise ValueError(f"chunk[{index}].page 必须是 int")
        if chunk["page"] < 1:
            raise ValueError(f"chunk[{index}].page 必须是大于 0 的整数")
        if not isinstance(chunk["chapter"], str):
            raise ValueError(f"chunk[{index}].chapter 必须是字符串")
        if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"].strip():
            raise ValueError(f"chunk[{index}].chunk_id 必须是非空字符串")

        if len(chunk["text"]) > MAX_TEXT_LENGTH:
            raise ValueError(f"chunk[{index}].text 超过最大长度 {MAX_TEXT_LENGTH}")
        if len(chunk["source"]) > MAX_SOURCE_LENGTH:
            raise ValueError(f"chunk[{index}].source 超过最大长度 {MAX_SOURCE_LENGTH}")
        if len(chunk["chapter"]) > MAX_CHAPTER_LENGTH:
            raise ValueError(f"chunk[{index}].chapter 超过最大长度 {MAX_CHAPTER_LENGTH}")
        if len(chunk["chunk_id"]) > MAX_CHUNK_ID_LENGTH:
            raise ValueError(f"chunk[{index}].chunk_id 超过最大长度 {MAX_CHUNK_ID_LENGTH}")
