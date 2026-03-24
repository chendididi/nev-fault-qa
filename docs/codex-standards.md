# Codex Standards

这个文件把两篇 OpenAI 官方文章落成仓库内可执行的工作标准，避免依赖聊天记忆。

参考来源：

- `Best practices`：<https://developers.openai.com/codex/learn/best-practices>
- `Harness engineering: Leveraging Codex in an agent-first world`：<https://openai.com/index/harness-engineering/>

## 适用范围

- 任何进入本仓库工作的 agent，都默认按本文件执行。
- 如果聊天指令、README 片段和这里冲突，以 `AGENTS.md` + `docs/` 为准。

## 仓库内落地规则

1. `AGENTS.md` 只做短入口，不堆长篇背景。
   细节放到 `docs/`，避免每次重复解释。
2. 任务描述尽量收敛到四件事：
   `Goal`、`Context`、`Constraints`、`Done when`。
3. 先定位改动子系统，再沿对应 harness 层级验证。
   默认从 `make check` 开始，不先上重依赖。
4. 优先把规则写成可执行约束，而不是只写成说明文字。
   包括结构边界测试、契约测试、smoke tests、Make 入口。
5. API 层只做 orchestration。
   不把检索、图谱、prompt 细节塞进路由。
6. 前端通过 HTTP 调 API。
   不直接 import 核心检索、图谱、LLM 模块。
7. 离线数据链路以 `data/processed/chunks.json` 为事实来源。
   改 chunks 契约时，必须同时更新验证和文档。
8. 默认 harness 要优先保证本地、快速、可重复。
   GPU、大模型、Milvus、Neo4j 相关验证属于更高层级，执行时写清前置条件。
9. 重复出现的人工流程要产品化。
   能写成 Make 目标、脚本、测试、文档入口时，不要只留在聊天记录里。
10. 交付必须报告四类证据：
   改了什么、验证了什么、没验证什么及原因、剩余风险在哪。

## 对 agent 的额外要求

- 不假设仓库知识存在于聊天历史里；先读 `AGENTS.md` 和相关 `docs/`。
- 不把“计划中应该做”的事情说成“已经做到”。
- 文档改造要保持渐进披露：入口短，细节分层，命令可直接执行。
- 新增 harness 时，优先补仓库真实痛点，不为“看起来完整”而堆自动化。
