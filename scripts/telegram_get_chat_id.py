"""
获取 Telegram chat_id 的辅助脚本。

使用方法：
    export TELEGRAM_BOT_TOKEN="123456:ABCDEF..."
    python scripts/telegram_get_chat_id.py

步骤：
1. 先在 Telegram 里给你的机器人发一条消息（例如 /start）。
2. 运行本脚本，它会通过 getUpdates 拉取最近消息并打印 chat_id。
"""

from __future__ import annotations

import os
import sys

import httpx


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("缺少环境变量 TELEGRAM_BOT_TOKEN")
        return 1

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        response = httpx.get(url, timeout=20.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"调用 Telegram API 失败: {exc}")
        return 1

    payload = response.json()
    if not payload.get("ok"):
        print(f"Telegram API 返回异常: {payload}")
        return 1

    updates = payload.get("result", [])
    if not updates:
        print("没有获取到消息。请先在 Telegram 里给机器人发一条消息，再重试。")
        return 0

    seen: set[int] = set()
    print("最近 chat_id 列表：")
    for item in updates:
        message = item.get("message") or item.get("edited_message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None or chat_id in seen:
            continue
        seen.add(chat_id)
        title = chat.get("title") or chat.get("username") or chat.get("first_name") or "unknown"
        chat_type = chat.get("type", "unknown")
        print(f"- chat_id={chat_id} | type={chat_type} | name={title}")

    if not seen:
        print("更新里没有可用 chat 信息。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
