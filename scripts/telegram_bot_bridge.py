"""
Telegram 与 NEV QA API 的桥接机器人。

功能：
- 长轮询接收 Telegram 消息
- 调用本地 /api/v1/chat
- 将答案回发到同一个 chat

环境变量：
- TELEGRAM_BOT_TOKEN: 机器人 token（必填）
- TELEGRAM_ALLOWED_CHAT_ID: 限定可访问 chat_id（可选，多个用逗号分隔）
- TELEGRAM_POLL_TIMEOUT: 长轮询超时秒数（可选，默认 30）
- TELEGRAM_SESSION_STORE: 会话映射文件路径（可选，默认 data/artifacts/telegram_sessions.json）
- NEV_API_URL: 问答 API 地址（可选，默认 http://127.0.0.1:8000/api/v1/chat）

运行：
    python scripts/telegram_bot_bridge.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


class TelegramBridge:
    def __init__(self) -> None:
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not self.bot_token:
            raise ValueError("缺少环境变量 TELEGRAM_BOT_TOKEN")

        self.allowed_chat_ids = _split_csv(os.getenv("TELEGRAM_ALLOWED_CHAT_ID"))
        self.poll_timeout = int(os.getenv("TELEGRAM_POLL_TIMEOUT", "30"))
        self.nev_api_url = os.getenv("NEV_API_URL", "http://127.0.0.1:8000/api/v1/chat").strip()
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        store_path = os.getenv("TELEGRAM_SESSION_STORE", "data/artifacts/telegram_sessions.json")
        self.session_store_path = Path(store_path)
        self.session_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.chat_sessions = self._load_sessions()

        self.offset = 0

    def _load_sessions(self) -> dict[str, str]:
        if not self.session_store_path.exists():
            return {}
        try:
            payload = json.loads(self.session_store_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items()}
        except json.JSONDecodeError:
            pass
        return {}

    def _save_sessions(self) -> None:
        self.session_store_path.write_text(
            json.dumps(self.chat_sessions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _telegram_request(self, method: str, payload: dict | None = None, *, timeout: float = 40.0) -> dict:
        url = f"{self.base_url}/{method}"
        response = httpx.post(url, json=payload or {}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API 异常: {data}")
        return data

    def _send_message(self, chat_id: int, text: str) -> None:
        text = text.strip() or "（空响应）"
        max_len = 3900
        chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
        for chunk in chunks:
            self._telegram_request(
                "sendMessage",
                {"chat_id": chat_id, "text": chunk},
            )

    def _call_nev_api(self, message: str, session_id: str | None) -> tuple[str, str | None]:
        payload: dict[str, str] = {"message": message}
        if session_id:
            payload["session_id"] = session_id

        response = httpx.post(self.nev_api_url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        answer = str(data.get("answer", "")).strip()
        next_session_id = data.get("session_id")
        return answer, next_session_id

    def _handle_message(self, message: dict) -> None:
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        text = (message.get("text") or "").strip()
        if chat_id is None or not text:
            return

        chat_id_str = str(chat_id)
        if self.allowed_chat_ids and chat_id_str not in self.allowed_chat_ids:
            self._send_message(chat_id, "当前 chat 未授权访问该机器人。")
            return

        if text in {"/start", "/help"}:
            self._send_message(
                chat_id,
                "机器人已连接到 NEV QA。\n直接发送故障码或现象描述即可开始问答。",
            )
            return

        if text == "/reset":
            if chat_id_str in self.chat_sessions:
                del self.chat_sessions[chat_id_str]
                self._save_sessions()
            self._send_message(chat_id, "会话已重置。")
            return

        if text == "/ping":
            self._send_message(chat_id, "pong")
            return

        self._send_message(chat_id, "处理中，请稍候...")
        try:
            session_id = self.chat_sessions.get(chat_id_str)
            answer, next_session_id = self._call_nev_api(text, session_id)
            if next_session_id:
                self.chat_sessions[chat_id_str] = str(next_session_id)
                self._save_sessions()
            self._send_message(chat_id, answer or "未生成答案，请稍后重试。")
        except Exception as exc:
            self._send_message(chat_id, f"处理失败：{exc}")

    def run(self) -> None:
        print("Telegram bridge 启动成功，开始轮询消息...")
        while True:
            try:
                data = self._telegram_request(
                    "getUpdates",
                    {"offset": self.offset, "timeout": self.poll_timeout},
                    timeout=self.poll_timeout + 15,
                )
                for update in data.get("result", []):
                    update_id = int(update.get("update_id", 0))
                    self.offset = max(self.offset, update_id + 1)
                    message = update.get("message") or update.get("edited_message") or {}
                    self._handle_message(message)
            except Exception as exc:
                print(f"[WARN] 轮询失败: {exc}")
                time.sleep(2)


def main() -> int:
    try:
        bot = TelegramBridge()
    except Exception as exc:
        print(f"初始化失败: {exc}")
        return 1

    bot.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
