# Telegram 对接指南

本文档用于将 Telegram Bot 接入本项目 `/api/v1/chat` 接口。

## 1. 创建机器人并获取 Token

1. 在 Telegram 搜索 `@BotFather`
2. 发送 `/newbot` 并按提示创建机器人
3. 记录返回的 `HTTP API Token`

## 2. 安装依赖

项目已包含 `httpx`，通常无需额外安装。如果你还没装依赖：

```bash
pip install -r requirements.txt
```

## 3. 获取 chat_id

先在 Telegram 给你的机器人发送一条消息（如 `/start`），然后执行：

```bash
export TELEGRAM_BOT_TOKEN="123456:ABCDEF..."
python scripts/telegram_get_chat_id.py
```

脚本会打印最近的 `chat_id`。

## 4. 启动后端 API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## 5. 启动 Telegram Bridge

```bash
export TELEGRAM_BOT_TOKEN="123456:ABCDEF..."
export TELEGRAM_ALLOWED_CHAT_ID="123456789"  # 可选，建议配置
export NEV_API_URL="http://127.0.0.1:8000/api/v1/chat"  # 可选
python scripts/telegram_bot_bridge.py
```

## 6. 可用命令

- `/start` 或 `/help`：显示帮助
- `/ping`：连通性测试
- `/reset`：重置当前 chat 的会话上下文

## 7. 常见问题

- 收不到消息：
  - 检查机器人是否已在 Telegram 收到你发的消息
  - 检查 `TELEGRAM_BOT_TOKEN` 是否正确
- 提示未授权：
  - 检查 `TELEGRAM_ALLOWED_CHAT_ID` 是否包含当前 chat_id
- 回答失败：
  - 确认本地 API 可访问：`curl http://127.0.0.1:8000/health`
