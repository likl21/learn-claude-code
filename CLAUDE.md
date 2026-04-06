# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

教学项目，讲授 **harness 工程** -- 构建围绕 AI agent 模型的基础设施。核心理念："模型即 agent，代码即 harness。" 通过 12 个递进式 session（s01-s12）逆向解析 Claude Code 的架构，每个 session 添加一个 harness 机制。

## 常用命令

### Python Agent

```sh
pip install -r requirements.txt
cp .env.example .env   # 填入 ANTHROPIC_API_KEY、MODEL_ID，可选 ANTHROPIC_BASE_URL

# 运行任意 session
python agents/s01_agent_loop.py
python agents/s_full.py          # 总集：所有机制整合

# 单元测试
python tests/test_unit.py

# Session 测试（需设置 TEST_API_KEY、TEST_BASE_URL、TEST_MODEL 环境变量）
python tests/test_v0.py          # 至 test_v9.py
```

### Web 平台（Next.js）

```sh
cd web
npm install
npm run dev          # 开发服务器 localhost:3000
npm run build        # 静态导出（prebuild 自动执行 extract）
npx tsc --noEmit     # 类型检查
```

CI 流程：Web 端先 `npx tsc --noEmit` 再 `npm run build`；Python 端运行 `python tests/test_unit.py` 及矩阵 session 测试。

## 架构

### Session 递进结构

`agents/` 下每个 session 在核心 agent loop 上叠加一个 harness 机制，loop 本身不变：

| Session | 机制 | 新增内容 |
|---------|------|---------|
| s01 | Agent Loop | `while True` + `stop_reason` 检查 -- 核心模式 |
| s02 | Tool Dispatch | `TOOL_HANDLERS = {name: handler}` 分发映射 |
| s03 | TodoWrite | `TodoManager` + nag reminder 注入 |
| s04 | Subagents | 每个子 agent 使用独立 `messages[]` 实现上下文隔离 |
| s05 | Skills | `SkillLoader` 通过 `tool_result` 注入 SKILL.md（非 system prompt） |
| s06 | Context Compact | `microcompact()` + `auto_compact()` 三层压缩策略 |
| s07 | Task System | `TaskManager` 文件级 CRUD + 依赖图 |
| s08 | Background Tasks | `BackgroundManager` + 守护线程 + 通知队列 |
| s09 | Agent Teams | `TeammateManager` + `MessageBus`（JSONL 邮箱） |
| s10 | Team Protocols | 关闭握手 + 计划审批状态机 |
| s11 | Autonomous Agents | 空闲循环 + 从任务板自动认领 |
| s12 | Worktree Isolation | `WorktreeManager` + `EventBus` + git worktree 隔离 |

`s_full.py` 是总集，整合所有机制，包含 25+ 工具和 REPL 命令（`/compact`、`/tasks`、`/team`、`/inbox`）。

### 仓库结构

- `agents/` -- Python 参考实现（s01-s12 + s_full），每个文件独立自包含。
- `docs/{en,zh,ja}/` -- 三语 session 文档，心智模型优先：问题、方案、ASCII 图、最小代码。
- `skills/` -- s05 的 SkillLoader 按需加载的 SKILL.md 文件（agent-builder、code-review、mcp-builder、pdf）。
- `web/` -- Next.js 16 交互学习平台（App Router、静态导出），包含每个 session 的可视化组件（`components/visualizations/s01-s12/`）、agent loop 模拟器、diff 查看器、源码查看器。

### 核心设计模式

- **核心 loop 不变。** 所有 session 通过在 `while stop_reason == "tool_use"` 循环外围添加 tool handler 和 manager 来扩展行为。
- **工具分发是扁平映射。** 新增工具 = 在 `TOOL_HANDLERS` 中增加一个条目。
- **知识通过 `tool_result` 注入**，而非 system prompt。技能按需加载。
- **子 agent 使用独立 `messages[]` 实现上下文隔离** -- 子对话不污染父对话。
- **全程文件级持久化**（JSON 存任务，JSONL 存邮箱），无数据库。

### 环境配置

`.env` 文件配置 LLM 提供商，支持 Anthropic 兼容 API：Anthropic（Claude）、MiniMax、GLM/智谱、Kimi/月之暗面、DeepSeek。通过 `ANTHROPIC_BASE_URL` 切换提供商。

### Web 平台细节

- `npm run extract`（dev/build 前自动执行）从 docs/agents 提取内容用于静态站点生成。
- 国际化位于 `web/src/i18n/`（en、zh、ja）。
- 每个 session 的可视化组件位于 `web/src/components/visualizations/`。
