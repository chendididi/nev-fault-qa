# Docs Index

这套文档按 “agent 先执行、再深入” 组织，不按概念堆叠。

## 阅读顺序

1. `../AGENTS.md`
2. `codex-standards.md`
3. `repo-map.md`
4. `harness.md`
5. `architecture.md`
6. `official-sources.md`

## 按任务找文档

- 想知道代码应该改哪里：看 `repo-map.md`
- 想知道先跑什么、怎么验证、失败后怎么查：看 `harness.md`
- 想知道这个仓库默认按什么 agent 标准执行：看 `codex-standards.md`
- 想理解整体链路和运行时组件：看 `architecture.md`
- 想知道官方数据从哪里来、哪些可以自动下载：看 `official-sources.md`

## 文档职责

- `AGENTS.md`
  仓库级入口，给 agent 和维护者最短执行路径。
- `repo-map.md`
  说明模块边界、允许依赖方向、常见改动的落点。
- `codex-standards.md`
  说明本仓库采用的 Codex / harness engineering 工作标准。
- `harness.md`
  说明本仓库当前可用的验证 harness、命令入口、证据要求和后续改造优先级。
- `architecture.md`
  保留系统级结构图和核心链路说明。
- `official-sources.md`
  说明官方公开文档来源、下载边界和当前可自动化入口。

## Source Of Truth

- 执行流程和约束：`AGENTS.md` + `docs/`
- 项目介绍和部署说明：`README.md`
- Claude 兼容说明：`CLAUDE.md`

如果几处文档冲突，以 `AGENTS.md` 和 `docs/` 为准，再回头修正其他文档。
