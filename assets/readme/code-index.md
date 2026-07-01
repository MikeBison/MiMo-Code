# MiMoCode 代码索引

按功能模块索引到具体源码位置，方便快速跳转。整体架构说明见 [architecture.md](./architecture.md)。

## Agent 模块代码索引

以「一个 Agent = plan/skills + memory + tools use + action」的心智模型，对应到具体代码位置。Agent 的运行核心是 `packages/opencode/src/session/` 编排循环，它把下列各块串联起来。

### 1. Plan / Skills（规划与技能）

- [packages/opencode/src/agent/agent.ts](../../packages/opencode/src/agent/agent.ts) — 内置 agent 角色定义：`build`（默认，全工具）、`plan`（规划模式，禁用编辑工具）、`explore`（探索模式）
- [packages/opencode/src/agent/config.ts](../../packages/opencode/src/agent/config.ts) — agent 配置
- [packages/opencode/src/tool/plan.ts](../../packages/opencode/src/tool/plan.ts) — 进入/退出 plan 模式的工具
- [packages/opencode/src/tool/plan-enter.txt](../../packages/opencode/src/tool/plan-enter.txt) · [packages/opencode/src/tool/plan-exit.txt](../../packages/opencode/src/tool/plan-exit.txt) — plan 模式提示词
- [packages/opencode/src/skill/index.ts](../../packages/opencode/src/skill/index.ts) — 技能加载/注册核心
- [packages/opencode/src/skill/discovery.ts](../../packages/opencode/src/skill/discovery.ts) — 从 `.claude/.agents/.codex/.opencode/skill(s)/**/SKILL.md` 发现技能
- [packages/opencode/src/skill/builtin/](../../packages/opencode/src/skill/builtin/) · [packages/opencode/src/skill/compose/](../../packages/opencode/src/skill/compose/) — 内置技能 / Compose 技能
- [packages/opencode/src/tool/skill.ts](../../packages/opencode/src/tool/skill.ts) · [packages/opencode/src/tool/skill.txt](../../packages/opencode/src/tool/skill.txt) — 技能调用工具
- [packages/opencode/src/session/system.ts](../../packages/opencode/src/session/system.ts) — 将技能注入系统提示词（`skills(agent)`）

### 2. Memory（记忆）

- [packages/opencode/src/memory/service.ts](../../packages/opencode/src/memory/service.ts) — 记忆读写服务
- [packages/opencode/src/memory/index.ts](../../packages/opencode/src/memory/index.ts) — 记忆模块入口
- [packages/opencode/src/memory/paths.ts](../../packages/opencode/src/memory/paths.ts) — 记忆路径管理
- [packages/opencode/src/memory/reconcile.ts](../../packages/opencode/src/memory/reconcile.ts) — 记忆合并
- [packages/opencode/src/memory/fts.sql.ts](../../packages/opencode/src/memory/fts.sql.ts) · [packages/opencode/src/memory/fts-query.ts](../../packages/opencode/src/memory/fts-query.ts) — 基于 SQLite FTS 的记忆检索
- [packages/opencode/src/tool/memory.ts](../../packages/opencode/src/tool/memory.ts) · [packages/opencode/src/tool/memory.txt](../../packages/opencode/src/tool/memory.txt) — 记忆工具
- [packages/opencode/src/tool/memory-path-guard.ts](../../packages/opencode/src/tool/memory-path-guard.ts) — 记忆路径保护

### 3. Tools use（工具定义与调度）

- [packages/opencode/src/tool/registry.ts](../../packages/opencode/src/tool/registry.ts) — 工具注册中心
- [packages/opencode/src/tool/tool.ts](../../packages/opencode/src/tool/tool.ts) — 工具基类/接口（`Tool.Def`）
- [packages/opencode/src/tool/schema.ts](../../packages/opencode/src/tool/schema.ts) — 参数校验（Zod）
- 文件类：[read.ts](../../packages/opencode/src/tool/read.ts) · [write.ts](../../packages/opencode/src/tool/write.ts) · [edit.ts](../../packages/opencode/src/tool/edit.ts) · [multiedit.ts](../../packages/opencode/src/tool/multiedit.ts) · [apply_patch.ts](../../packages/opencode/src/tool/apply_patch.ts)
- 搜索类：[glob.ts](../../packages/opencode/src/tool/glob.ts) · [grep.ts](../../packages/opencode/src/tool/grep.ts) · [codesearch.ts](../../packages/opencode/src/tool/codesearch.ts)
- 执行类：[bash.ts](../../packages/opencode/src/tool/bash.ts) · [bash-interactive.ts](../../packages/opencode/src/tool/bash-interactive.ts)
- 高级：[task.ts](../../packages/opencode/src/tool/task.ts) · [actor.ts](../../packages/opencode/src/tool/actor.ts) · [workflow.ts](../../packages/opencode/src/tool/workflow.ts) · [cron.ts](../../packages/opencode/src/tool/cron.ts) · [question.ts](../../packages/opencode/src/tool/question.ts) · [lsp.ts](../../packages/opencode/src/tool/lsp.ts) · [webfetch.ts](../../packages/opencode/src/tool/webfetch.ts) · [websearch/](../../packages/opencode/src/tool/websearch/)
- [packages/opencode/src/permission/](../../packages/opencode/src/permission/) — 工具执行前的权限控制

### 4. Action（执行 / 决策循环）

「LLM 决策 → 执行工具 → 回填结果 → 再决策」的 loop：

- [packages/opencode/src/session/processor.ts](../../packages/opencode/src/session/processor.ts) — 主处理器（识别工具调用、执行、循环、doom-loop 检测、max mode 多候选评审）
- [packages/opencode/src/session/llm.ts](../../packages/opencode/src/session/llm.ts) — LLM 调用
- [packages/opencode/src/session/prompt.ts](../../packages/opencode/src/session/prompt.ts) · [system.ts](../../packages/opencode/src/session/system.ts) · [llm-request-prefix.ts](../../packages/opencode/src/session/llm-request-prefix.ts) — 系统提示词构建
- [packages/opencode/src/session/checkpoint.ts](../../packages/opencode/src/session/checkpoint.ts) · [compaction.ts](../../packages/opencode/src/session/compaction.ts) · [overflow.ts](../../packages/opencode/src/session/overflow.ts) — 上下文/检查点管理与窗口压缩
- [packages/opencode/src/task/registry.ts](../../packages/opencode/src/task/registry.ts) — 任务/子 agent 编排
- [packages/opencode/src/provider/](../../packages/opencode/src/provider/) — LLM Provider 抽象（选模型、发请求）

## CLI / 前端 / 网页端代码索引

| 端 | 路径 | 说明 |
|----|------|------|
| CLI 入口 | [packages/opencode/src/index.ts](../../packages/opencode/src/index.ts) | yargs 解析参数，分发子命令 |
| CLI 子命令 | [packages/opencode/src/cli/cmd/](../../packages/opencode/src/cli/cmd/) | `run` / `serve` / `web` / `agent` / `session` / `mcp` 等 |
| TUI（终端界面，当前核心） | [packages/opencode/src/cli/cmd/tui/](../../packages/opencode/src/cli/cmd/tui/) | SolidJS + OpenTUI 渲染 |
| Web 服务端 | [packages/opencode/src/server/](../../packages/opencode/src/server/) | Hono 服务器（配合 `cli/cmd/serve.ts`） |
| 文档官网（网页端） | [packages/web/](../../packages/web/) | Astro + Starlight |
| Web 应用 UI | [packages/app/](../../packages/app/) | SolidJS 前端应用（DOM 渲染） |
| 桌面端 | [packages/desktop/](../../packages/desktop/) | Electron 封装，复用 `app` |
| 共享 UI 组件库 | [packages/ui/](../../packages/ui/) | 组件 / 主题 / 图标 |
| 客户端 SDK | [packages/sdk/](../../packages/sdk/) | JS 等客户端 SDK |

> 注：当前核心开发重点是 **TUI**（`packages/opencode/src/cli/cmd/tui/`），Web / App 暂不主力维护。查看「界面」优先看 TUI；查看「Agent 大脑」看 `session/` + `tool/` + `agent/` + `skill/` + `memory/`。
