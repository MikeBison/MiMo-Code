import path from "path"
import os from "os"
import z from "zod"
import { SessionID, MessageID, PartID } from "./schema"
import { MessageV2 } from "./message-v2"
import { classifyAssistantStep } from "./classify"
import { Log } from "../util"
import { SessionRevert } from "./revert"
import * as Session from "./session"
import { Agent } from "../agent/agent"
import { decideAskRouting, SYSTEM_SPAWNED_AGENT_TYPES } from "@/agent/config"
import { renderActorNotification } from "@/inbox/render"
import { parseReturnHeader } from "@/actor/return-header"
import { Provider } from "../provider"
import { ModelID, ProviderID } from "../provider/schema"
import {
  type Tool as AITool,
  type ModelMessage,
  tool,
  jsonSchema,
  type ToolExecutionOptions,
  asSchema,
  generateText,
  wrapLanguageModel,
} from "ai"
import { InstallationVersion } from "@/installation/version"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { SessionPrune } from "./prune"
import { SessionCheckpoint } from "./checkpoint"
import { SessionCompaction } from "./compaction"
import { computeLastMessageInfo } from "./last-message-info"
import { pressureLevel, isOverflow as overflowCheck } from "./overflow"
import { Config } from "@/config"
import { Global } from "@/global"
import { Bus } from "../bus"
import { ProviderTransform } from "../provider"
import { SystemPrompt } from "./system"
import { Instruction } from "./instruction"
import { TuiEvent } from "@/cli/cmd/tui/event"
import { Plugin } from "../plugin"
import BUILD_SWITCH from "../session/prompt/build-switch.txt"
import MAX_STEPS from "../session/prompt/max-steps.txt"
import PROMPT_COMPOSE from "../session/prompt/compose.txt"
import {
  RECOVERY_PROMPT_MILD,
  RECOVERY_PROMPT_STRONG,
  TEXT_LOOP_BUFFER_SIZE,
  TEXT_LOOP_TRIGGER_COUNT,
  TEXT_LOOP_MAX_RECOVERY,
  normalizeForLoopDetection,
  detectTextLoop,
} from "../session/prompt/text-loop-recovery"
import {
  TEXT_NGRAM_MAX_RECOVERY,
  TEXT_NGRAM_RECOVERY_REMIND,
  TEXT_NGRAM_RECOVERY_REPLAN,
} from "../session/prompt/text-ngram-detection"
import {
  EMPTY_STEP_MAX_RECOVERY,
  EMPTY_STEP_RECOVERY_REMIND,
  EMPTY_STEP_RECOVERY_REPLAN,
  isEmptyStep,
} from "../session/prompt/empty-step-detection"
import { composeSkillsBlock } from "@/skill/compose/extract"
import { builtinSkillRoot, matchDocumentSkills } from "@/skill/builtin/extract"
import { ToolRegistry } from "../tool"
import { MCP } from "../mcp"
import { normalizeToolResult } from "../mcp/tool-result"
import { LSP } from "../lsp"
import { Flag } from "../flag/flag"
import { ulid } from "ulid"
import { ChildProcess, ChildProcessSpawner } from "effect/unstable/process"
import * as CrossSpawnSpawner from "@/effect/cross-spawn-spawner"
import * as Stream from "effect/Stream"
import { Command } from "../command"
import { pathToFileURL, fileURLToPath } from "url"
import { ConfigMarkdown, ConfigCompose } from "../config"
import { SessionSummary } from "./summary"
import { NamedError } from "@mimo-ai/shared/util/error"
import { SessionProcessor } from "./processor"
import { buildLLMRequestPrefix } from "./llm-request-prefix"
import {
  serializeTrajectoryMessages,
  withAssistantParts,
  userQueryText,
  assistantFinalText,
  sessionErrorText,
} from "./trajectory"
import { prefixCaptureRef } from "./prefix-capture-ref"
import { spawnRef } from "@/actor/spawn-ref"
import { Inbox } from "@/inbox"
import { sessionPromptRef, defaultModelRef } from "@/inbox/inbox-ref"
import { Tool } from "@/tool"
import { Permission } from "@/permission"
import { SessionStatus } from "./status"
import { LLM } from "./llm"
import { MaxMode } from "./max-mode"
import { Shell } from "@/shell/shell"
import { AppFileSystem } from "@mimo-ai/shared/filesystem"
import { Truncate } from "@/tool"
import { decodeDataUrl } from "@/util/data-url"
import { Process } from "@/util"
import { Cause, Effect, Exit, Layer, Option, Scope, Context } from "effect"
import { EffectLogger } from "@/effect"
import { InstanceState } from "@/effect"
import { ActorTool, type ActorPromptOps } from "@/tool/actor"
import { SessionRunState } from "./run-state"
import { Goal } from "./goal"
import { TaskGate, MAX_TASK_GATE_MAIN_REACT } from "@/task/gate"
import { TaskGateState } from "@/task/gate-state"
import { TaskRegistry } from "@/task/registry"
import { EffectBridge } from "@/effect"
import { Team } from "@/team"
import { ActorRegistry } from "@/actor/registry"
import { Metrics } from "@/metrics"
import { resolveInvocationStyle, type ToolStyleConfig } from "../tool/invocation-style"
import { ToolResultError } from "../tool/result-error"
import { shouldAutoDream, shouldAutoDistill, DREAM_TASK, DISTILL_TASK, AUTO_DREAM_TITLE, AUTO_DISTILL_TITLE } from "./auto-dream"

// @ts-ignore
globalThis.AI_SDK_LOG_WARNINGS = false

// 召回提醒的提示行，按每个工具配置的调用风格渲染，这样 shell 模式的会话就永远
// 不会看到 JSON 形态的示例（那会诱导模型输出 JSON 并使 shell 解析器崩溃）。
// `memory` 没有 shell 形态，所以它始终是 JSON。导出以便单元测试。
export function recallHintLines(toolCfg: ToolStyleConfig | undefined): string[] {
  const taskHint =
    resolveInvocationStyle(toolCfg, "task") === "shell" ? "- task list" : `- task({ operation: "list" })`
  const actorHint =
    resolveInvocationStyle(toolCfg, "actor") === "shell"
      ? "- actor status <actor_id>"
      : `- actor({ operation: "status", actor_id: "<id>" })`
  // memory 没有 shell 形态（无 shell.parse）→ 始终是 JSON。
  return [`- memory({ operation: "search", query: "<keyword>" })`, taskHint, actorHint]
}

// The orchestrator root session is PERSISTENT and coordinates many tasks over
// its lifetime, so its title must be stable and task-independent — it must not
// be renamed by the per-first-message auto-title generator as tasks come and
// go. Any root session driven by the orchestrator agent keeps this fixed name.
export const ORCHESTRATOR_TITLE = "Orchestrator"

// Returns the stable, task-independent title a root session should keep instead
// of a per-message auto-generated one, or undefined when normal auto-titling
// applies. Pure + exported for unit testing. `agent` is the triggering agent's
// name (e.g. "orchestrator"); `parentID` distinguishes root from child sessions.
export function stableRootTitle(input: { agent: string | undefined; parentID: string | undefined }): string | undefined {
  if (input.parentID) return undefined
  if (input.agent === "orchestrator") return ORCHESTRATOR_TITLE
  return undefined
}

/**
 * 每一轮 goal 驱动的主循环重入次数上限——防止一个永远无法满足的条件无限烧 token 的
 * 安全阀。比派生 actor 的 MAX_PRE_REACT（=3）更高，因为主会话的 goal 通常更大。
 * TODO: 提升到 mimocode.json 配置（例如 session.maxGoalReact）。
 */
const MAX_GOAL_REACT = 12

/**
 * 连续多少个已完成的助手步骤具有相同的 action 签名时，触发"重复步骤"提示。
 * 连续三次是模型陷入自我重复、而非在取得进展的强烈信号。
 */
const REPEATED_STEP_THRESHOLD = 3

/**
 * 带有排序键的确定性 JSON 序列化，使两个语义相同的工具输入无论模型碰巧以何种键顺序
 * 输出，都产生相同的字符串。`JSON.stringify` 会保留插入顺序，而模型经常以不同的键
 * 顺序重新输出相同的参数（例如 {url,format} 与 {format,url}）——没有这一步，签名
 * 就会不同，重复步骤检查就会漏掉真正的循环。
 */
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null"
  if (Array.isArray(value)) return "[" + value.map(stableStringify).join(",") + "]"
  const keys = Object.keys(value as Record<string, unknown>).sort()
  return (
    "{" +
    keys.map((k) => JSON.stringify(k) + ":" + stableStringify((value as Record<string, unknown>)[k])).join(",") +
    "}"
  )
}

/**
 * 助手步骤中*动作*的稳定签名——即它发起的工具调用（名称 + 与键顺序无关的输入）。
 * 文本和推理被有意排除在外：在 ReAct 循环里，模型会用略有不同的措辞叙述每个步骤，
 * 但采取的却是完全相同的动作；而且有些模型会把推理当作纯文本 part 输出——把这两者
 * 计入都会掩盖我们想捕捉的重复动作。当某步骤没有发起任何工具调用时（例如纯文本 turn）
 * 返回 undefined，因为此时没有可比较的重复*动作*。
 */
function stepSignature(parts: MessageV2.Part[]): string | undefined {
  const segments: string[] = []
  for (const part of parts) {
    if (part.type === "tool") {
      segments.push("tool:" + part.tool + ":" + stableStringify(part.state.input ?? {}))
    }
  }
  if (segments.length === 0) return undefined
  return segments.join("\n")
}

/**
 * 高上下文压力下"内存刷写提示"的防抖判定。
 *
 * 如果在*当前高压力片段*内已经注入过一次提示（一个包含 `marker` 的文本 part），
 * 则返回 true；这里的片段指自上一个 checkpoint 边界以来的消息窗口。
 *
 * 以 checkpoint 边界而非固定消息数作为锚点是有意为之的：一个持续的高压力 turn 可能
 * 发出许多工具调用步骤——每个都是独立的消息——所以固定大小的尾窗会让已提示过的
 * 消息滑出窗口，从而在 turn 中途再次触发提示。只有当 checkpoint/rebuild 真正丢弃
 * 上下文时边界才会前移，而那正是重新提示重新变得有用的时机。
 *
 * 当 `boundaryID` 为 undefined（尚无 checkpoint）或在 `msgs` 中找不到时，
 * 整个会话都被视为当前片段。
 */
export function nudgedSinceBoundary(
  msgs: readonly MessageV2.WithParts[],
  boundaryID: string | undefined,
  marker: string,
): boolean {
  const boundaryIdx = boundaryID ? msgs.findIndex((m) => m.info.id === boundaryID) : -1
  const episode = boundaryIdx >= 0 ? msgs.slice(boundaryIdx) : msgs
  return episode.some((m) => m.parts.some((p) => p.type === "text" && p.text?.includes(marker)))
}

const STRUCTURED_OUTPUT_DESCRIPTION = `Use this tool to return your final response in the requested structured format.

IMPORTANT:
- You MUST call this tool exactly once at the end of your response
- The input must be valid JSON matching the required schema
- Complete all necessary research and tool calls BEFORE calling this tool
- This tool provides your final answer - no further actions are taken after calling it`

const STRUCTURED_OUTPUT_SYSTEM_PROMPT = `IMPORTANT: The user has requested structured output. You MUST use the StructuredOutput tool to provide your final response. Do NOT respond with plain text - you MUST call the StructuredOutput tool with your answer formatted according to the schema.`

const PREDICT_SYSTEM = `You predict the single most likely next message a user will send to a coding assistant, based on the conversation so far. Output only that next message as one short, natural first-person request (what the user would type). No preamble, no quotes, no explanation, no markdown. Keep it under 100 characters.`

const PREDICT_NUDGE = `Based on the conversation above, write the user's most likely next message:`

const OUTPUT_LENGTH_CONTINUATION_LIMIT = Flag.MIMOCODE_OUTPUT_LENGTH_CONTINUATION_LIMIT
const INVALID_OUTPUT_CONTINUATION_LIMIT = Flag.MIMOCODE_INVALID_OUTPUT_CONTINUATION_LIMIT
const TEXT_TOOL_CALL_RETRY_LIMIT = Flag.MIMOCODE_TEXT_TOOL_CALL_RETRY_LIMIT

const log = Log.create({ service: "session.prompt" })

// 这里不列出 Hooks：插件层自己通过 mtime 过期检查来探测 hook 文件变化
//（外部编辑器也能覆盖到），所以只有 tools 和 skills 需要在 write/edit 触发时
// 重新加载注册表。
function isExtensionPath(filePath: string): boolean {
  return /\/\.mimocode\/(tools?|skills?)\//.test(filePath)
}
const elog = EffectLogger.create({ service: "session.prompt" })

export interface Interface {
  // 中断指定会话正在进行的 agent 循环(用户按 Esc / 点停止时调用)。
  readonly cancel: (sessionID: SessionID) => Effect.Effect<void>
  // 【对外主入口】处理一条用户消息:组装请求并驱动 agent 循环,最终返回助手消息。
  readonly prompt: (input: PromptInput) => Effect.Effect<MessageV2.WithParts>
  // 【核心】agent 自主循环本体:调模型 → 执行工具 → 把结果喂回 → 再循环,直到结束。
  readonly loop: (input: z.infer<typeof LoopInput>) => Effect.Effect<MessageV2.WithParts>
  // shell 模式入口:把一条 shell 风格的输入当作 prompt 处理。
  readonly shell: (input: ShellInput) => Effect.Effect<MessageV2.WithParts>
  // 斜杠命令(如 /goal、/voice)入口:解析并执行自定义命令。
  readonly command: (input: CommandInput) => Effect.Effect<MessageV2.WithParts>
  // 把含占位符的模板字符串解析成实际的消息 parts(给 command/shell 复用)。
  readonly resolvePromptParts: (template: string) => Effect.Effect<PromptInput["parts"]>
  // 清理"孤儿"助手消息:上次因崩溃/中断而没写完(无 completed)的残留消息。immediate 为 true 时立即清理。
  readonly sweepOrphanAssistants: (sessionID: SessionID, immediate?: boolean) => Effect.Effect<void>
  // 预测:根据当前会话内容生成一段预测/建议文本(辅助功能)。
  readonly predict: (input: { sessionID: SessionID }) => Effect.Effect<string>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionPrompt") {}

export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    // 定义各个模块变量
    // ============================================================================
    // 第①段:取来本服务干活所需的全部依赖服务(每个 yield* 读作 await)。
    // 这一长串体现了 prompt.ts 是"调度中枢"——它编排下面这些专门服务来完成工作。
    // ============================================================================
    const bus = yield* Bus.Service                    // 事件总线:发布/订阅会话事件(消息更新、出错等)
    const status = yield* SessionStatus.Service        // 会话状态:设置 busy/idle 等运行状态
    const sessions = yield* Session.Service            // 会话存取:创建/读取/更新会话与消息
    const agents = yield* Agent.Service                // agent 注册表:按名字取 agent 定义(build/plan/explore…)
    const provider = yield* Provider.Service           // 模型提供方:解析/获取大模型(provider/model)
    const processor = yield* SessionProcessor.Service  // 处理器:处理模型返回的流式片段、落库
    const prune = yield* SessionPrune.Service          // 裁剪:删除过期/多余的消息数据
    const checkpoint = yield* SessionCheckpoint.Service // 检查点:跨会话记忆的存档(MEMORY/checkpoint)
    const compaction = yield* SessionCompaction.Service // 压缩:上下文接近上限时压缩历史
    const config = yield* Config.Service               // 配置:读取 mimocode 配置项
    const plugin = yield* Plugin.Service               // 插件:触发插件钩子(如 system 提示变换)
    const commands = yield* Command.Service            // 斜杠命令:获取/解析 /xxx 自定义命令
    const permission = yield* Permission.Service       // 权限:工具调用的允许/询问/拒绝判定
    const fsys = yield* AppFileSystem.Service           // 文件系统:读写文件的封装
    const mcp = yield* MCP.Service                      // MCP:外部 Model Context Protocol 服务器的工具/资源
    const lsp = yield* LSP.Service                      // LSP:语言服务器,提供代码符号/诊断等语义信息
    const registry = yield* ToolRegistry.Service        // 工具注册表:汇总所有可用工具(read/edit/bash…)
    const truncate = yield* Truncate.Service            // 截断:把过长的工具输出截断/落盘
    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner // 子进程派生器:启动外部进程
    const scope = yield* Scope.Scope                    // 作用域:管理资源生命周期(配合 finalizer 清理)
    const instruction = yield* Instruction.Service      // 指令:加载项目级 instructions(AGENTS.md 等)
    const state = yield* SessionRunState.Service        // 运行态:记录/取消会话当前的运行循环
    const goal = yield* Goal.Service                    // 目标:/goal 停止条件与裁判判定
    const taskGateState = yield* TaskGateState.Service  // 任务闸:控制任务执行的门控状态
    const taskRegistry = yield* TaskRegistry.Service    // 任务注册表:任务树(T1/T1.1…)的增删查改
    const revert = yield* SessionRevert.Service         // 回退:撤销某轮改动
    const summary = yield* SessionSummary.Service       // 摘要:生成会话/消息摘要
    const sys = yield* SystemPrompt.Service             // 系统提示:拼装 system prompt(环境、技能等)
    const llm = yield* LLM.Service                      // LLM:真正向大模型发请求、收流式响应
    const actorRegistry = yield* ActorRegistry.Service  // actor 注册表:子 agent(actor)的派生与管理
    const inbox = yield* Inbox.Service                  // 收件箱:agent 间消息传递(send/drain)

    // 记录已经展示过"已加载指令"提示的会话，这样每个主会话只提示一次，
    // 而不是在每一轮 run-loop 都提示。
    const instructionsNotified = new Set<SessionID>()

    // ============================================================================
    // capture:重建某个 agent 的"LLM 请求前缀"(系统提示 + 工具 + 历史消息),
    // 供 checkpoint-writer 子 agent 在 fork 时复用,以对齐父 agent 的请求、命中
    // 大模型的 prompt 缓存(省钱 + 提速)。
    //
    // 为什么用 prefixCaptureRef 这个"插槽"来晚绑定:checkpoint.ts 需要调用这里的
    // 逻辑,但直接 import 会形成循环依赖(ToolRegistry → SessionCheckpoint →
    // ToolRegistry)。于是把本函数塞进共享插槽,checkpoint.ts 只从插槽取用、
    // 不直接 import,从而打破循环。详见 prefix-capture-ref.ts。
    // ============================================================================
    const capture: typeof prefixCaptureRef.current = (input) =>
      Effect.gen(function* () {
        // 兜底返回值:任一步骤失败就返回这个"空前缀",绝不让 capture 抛错中断调用方。
        const empty = { system: [] as string[], tools: {} as Record<string, AITool>, inheritedMessages: [] as ModelMessage[], parentPermission: [] as Permission.Ruleset }
        // ① 按名字取 agent 定义(取不到就返回空前缀)。
        const ag = yield* agents.get(input.agentName).pipe(Effect.catch(() => Effect.succeed(undefined)))
        if (!ag) return empty
        // ② 取这次用的模型(取不到也返回空前缀)。
        const model = yield* provider
          .getModel(input.providerID as ProviderID, input.modelID as ModelID)
          .pipe(Effect.catch(() => Effect.succeed(undefined)))
        if (!model) return empty
        // 把 env 日期锚定到会话创建时间，使捕获到的前缀与 runLoop 的（它用 session.time.created）
        // 字节一致，从而保持 Anthropic 缓存命中。如果会话无法加载，就无法保证这种一致性，
        // 因此宁可回退到空前缀，也不发出一个不一致的日期。
        const captureSession = yield* sessions.get(input.sessionID).pipe(Effect.catch(() => Effect.succeed(undefined)))
        if (!captureSession) return empty
        // ③ 并行准备系统提示的三块来源:技能、运行环境信息、全局指令(instructions)。
        const [skills, env, instructions] = yield* Effect.all([
          sys.skills(ag),
          sys.environment(model, captureSession.time.created),
          instruction.system().pipe(Effect.orDie),
        ])
        // (checkpoint-writer 不要求 json_schema 输出,所以这里不含结构化输出的系统提示;
        //  那部分由父 agent 的 runLoop 根据 user.format 按需追加。)
        // ④ 把三块来源拼成"附加的系统内容"。
        const additions = [...env, ...(skills ? [skills] : []), ...instructions.content]
        // ⑤ 真正构建请求前缀;临时把 LLM、ToolRegistry 两个服务注入进去;失败仍回退空前缀。
        const prefix = yield* buildLLMRequestPrefix({
          sessionID: input.sessionID,
          agent: ag,
          model,
          msgs: input.msgs as Parameters<typeof buildLLMRequestPrefix>[0]["msgs"],
          additions,
        }).pipe(
          Effect.provideService(LLM.Service, llm), // 步骤①:注入 LLM 服务
          Effect.provideService(ToolRegistry.Service, registry), // 步骤②:注入工具注册表
          Effect.catch(() => Effect.succeed(empty)), // 步骤③:出错就回退空前缀
        )
        // ⑥ 连同父 agent 的权限一起返回(fork 出来的子 agent 要按父权限过滤工具)。
        return { ...prefix, parentPermission: ag.permission }
      })
    // 把上面这个函数塞进共享插槽,供 checkpoint.ts 取用。
    prefixCaptureRef.current = capture
    // 添加销毁钩子，本服务销毁时,若插槽还指向我们这个 capture,就清空它,避免悬空引用。
    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        if (prefixCaptureRef.current === capture) prefixCaptureRef.current = undefined
      }),
    )

    // runner:造一个"运行器/桥"(EffectBridge),用来把 Effect"计划"真正启动起来,
    // 主要供下面 ops 暴露给 Effect 世界之外的调用方(actor 系统)使用。
    const runner = Effect.fn("SessionPrompt.runner")(function* () {
      return yield* EffectBridge.make()
    })
    // ops:把本服务的几个能力包装成普通可调用对象,交给 actor(子 agent)系统使用。
    // cancel 用 run.fork 在后台启动(不等结束);prompt/resolvePromptParts 直接转调。
    const ops = Effect.fn("SessionPrompt.ops")(function* () {
      const run = yield* runner()
      return {
        cancel: (sessionID: SessionID) => run.fork(cancel(sessionID)),
        resolvePromptParts: (template: string) => resolvePromptParts(template),
        prompt: (input: PromptInput) => prompt(input),
      } satisfies ActorPromptOps
    })

    // cancel:中断指定会话——记日志,然后委托 state 服务去真正取消其运行循环。
    const cancel = Effect.fn("SessionPrompt.cancel")(function* (sessionID: SessionID) {
      yield* elog.info("cancel", { sessionID })
      yield* state.cancel(sessionID)
    })

    // ============================================================================
    // rebuildFromCheckpoint:从 checkpoint 重建上下文的"共享步骤",被两条路径复用——
    // runLoop 里的自动溢出路径 + 手动 /rebuild 命令,保证二者逻辑/边界条件永不漂移。
    // 它在当前 watermark 处插入一个 checkpoint 边界标记(绝不删 DB 消息):下一次 runLoop
    // 迭代会从磁盘上的 checkpoint 重建上下文,而 watermark 之后的实时消息尾巴原样保留。
    // 不阻塞等待在途的 writer(与自动路径同策略——宁可用略旧的 checkpoint,也不等一个
    // 可能永远不来的新的)。仅当成功插入边界(即存在可用 checkpoint)才返回 true;
    // 返回 false 时调用方回退到 compaction。
    // ============================================================================
    const rebuildFromCheckpoint = Effect.fn("SessionPrompt.rebuildFromCheckpoint")(function* (input: {
      sessionID: SessionID
      msgs: MessageV2.WithParts[]
      agentID?: string
      agent: string
      model: { providerID: string; id: string }
    }) {
      const hasCP = yield* checkpoint
        .hasCheckpoint(input.sessionID)
        .pipe(Effect.catch(() => Effect.succeed(false)))
      if (!hasCP) return false

      const boundary = yield* checkpoint
        .lastBoundary(input.sessionID)
        .pipe(Effect.catch(() => Effect.succeed(undefined)))
      if (!boundary) return false

      const boundaryMsg = input.msgs.find((m) => m.info.id === boundary)
      const inserted = yield* checkpoint
        .insertRebuildBoundary({
          sessionID: input.sessionID,
          boundary,
          lastMessageInfo: computeLastMessageInfo(input.msgs.map((m) => m.info)),
          agentID: input.agentID,
          agent: input.agent,
          model: { providerID: input.model.providerID, modelID: input.model.id },
          boundaryCreatedAt: boundaryMsg?.info.time.created,
        })
        .pipe(Effect.catch(() => Effect.succeed(false)))

      if (inserted) yield* prune.resetThresholds(input.sessionID)
      return inserted
    })

    // ============================================================================
    // resolvePromptParts:处理用户输入里的 `@` 引用。
    //
    // 例:用户输入 "看下 @src/app.ts,再让 @explore 去调研"
    //   → 输出 [
    //       { type: "text", text: "看下 @src/app.ts,再让 @explore 去调研" }, // 原话
    //       { type: "file", url: ".../src/app.ts" },  // @src/app.ts 是个文件
    //       { type: "agent", name: "explore" },        // @explore 是个 agent
    //     ]
    //
    // 它只负责"认出每个 @xxx 是文件还是 agent",不会去读文件内容(读内容是后面
    // createUserMessage 那步做的)。斜杠命令、shell 输入都会先过这个函数。
    // ============================================================================
    const resolvePromptParts = Effect.fn("SessionPrompt.resolvePromptParts")(function* (template: string) {
      const ctx = yield* InstanceState.context
      // 先把用户原话原样存进去当第一项。@xxx 不会被删掉,后面认出来的文件/agent
      // 是"额外加"在后面,不是替换掉原文。
      const parts: PromptInput["parts"] = [{ type: "text", text: template }]
      // 用正则把文本里所有 `@xxx` 找出来(match[1] 就是 @ 后面那串,比如 "src/app.ts")。
      const files = ConfigMarkdown.files(template)
      const seen = new Set<string>() // 记下处理过的,同一个 @foo 出现两次也只处理一次
      // 把所有 @xxx 同时(并发)处理。这里不关心每次的返回值,直接往上面的 parts 里塞。
      yield* Effect.forEach(
        files,
        Effect.fnUntraced(function* (match) {
          const name = match[1]
          if (seen.has(name)) return
          seen.add(name)
          // 把 @ 后面的名字拼成完整路径:`~/` 开头就是用户主目录下,否则就是项目目录下。
          const filepath = name.startsWith("~/")
            ? path.join(os.homedir(), name.slice(2))
            : path.resolve(ctx.worktree, name)

          // 看看这个路径在磁盘上到底存不存在。存在就拿到文件信息,不存在就是空。
          const info = yield* fsys.stat(filepath).pipe(Effect.option)
          if (Option.isNone(info)) {
            // 找不到这个文件 → 那 @name 可能是个 agent 名(比如 @explore、@plan)。
            // 是 agent 就记下来;两者都不是就不管它,@name 就当普通文字留在原话里。
            const found = yield* agents.get(name)
            if (found) parts.push({ type: "agent", name: found.name })
            return
          }
          // 文件确实存在 → 记一条文件引用(只记路径和类型,不读里面的内容)。
          // 用 mime 区分它是文件夹还是普通文件。
          // 注意:先查文件、再查 agent,所以万一有同名文件,会被当成文件。
          const stat = info.value
          parts.push({
            type: "file",
            url: pathToFileURL(filepath).href,
            filename: name,
            mime: stat.type === "Directory" ? "application/x-directory" : "text/plain",
          })
        }),
        { concurrency: "unbounded", discard: true },
      )
      return parts
    })

    // ============================================================================
    // title:给对话自动起标题(就像 ChatGPT 左侧列表里那个自动生成的对话名)。
    // 用户发完第一条消息后,后台偷偷调一个便宜的小模型,根据这句话概括出一个标题。
    // 全程静默运行、不阻塞主流程,失败了也只记日志不影响使用(所以是"锦上添花"功能)。
    // ============================================================================
    const title = Effect.fn("SessionPrompt.ensureTitle")(function* (input: {
      session: Session.Info
      agent: string | undefined
      history: MessageV2.WithParts[]
      providerID: ProviderID
      modelID: ModelID
    }) {
      // --- 一堆守卫:只在"该起标题"的时候才干活,否则直接返回 ---
      // 是子会话(被派生出来的小弟,用户看不到)→ 不用起标题
      if (input.session.parentID) return
      // 持久编排器根会话:保持一个稳定、与任务无关的标题。若仍是默认标题就设一次,
      // 并跳过"按首条消息生成标题"的 LLM 流程,这样后续任务永远不会重命名它。
      // Persistent orchestrator root session: keep a stable, task-independent
      // title. Set it once (if still the default) and SKIP the per-first-message
      // LLM title generation so later tasks never rename it.
      const stable = stableRootTitle({ agent: input.agent, parentID: input.session.parentID })
      if (stable) {
        if (Session.isDefaultTitle(input.session.title))
          yield* sessions
            .setTitle({ sessionID: input.session.id, title: stable })
            .pipe(Effect.catchCause((cause) => elog.error("failed to set stable title", { error: Cause.squash(cause) })))
        return
      }

      // 标题已经不是默认值了(用户改过 / 之前已生成过)→ 别覆盖
      if (!Session.isDefaultTitle(input.session.title)) return

      // real:判断"是不是一条真人发的用户消息"。排除掉系统自动塞进去的合成消息(synthetic)。
      const real = (m: MessageV2.WithParts) =>
        m.info.role === "user" && !m.parts.every((p) => "synthetic" in p && p.synthetic)
      const idx = input.history.findIndex(real)
      if (idx === -1) return // 压根没有真人消息 → 不起
      // 只在"恰好只有第一条真人消息"时起标题,保证整个对话只在开头生成这一次、不重复生成。
      if (input.history.filter(real).length !== 1) return

      // 截取"从头到第一条真人消息"这段作为生成标题的上下文。
      const context = input.history.slice(0, idx + 1)
      const firstUser = context[idx]
      if (!firstUser || firstUser.info.role !== "user") return // 顺便帮 TS 确认类型
      const firstInfo = firstUser.info

      // 特殊情况:第一条消息可能全是"子任务"(派活给子 agent)而不是文字,
      // onlySubtasks 标记这种情况,后面取内容的方式不一样。
      const subtasks = firstUser.parts.filter((p): p is MessageV2.SubtaskPart => p.type === "subtask")
      const onlySubtasks = subtasks.length > 0 && firstUser.parts.every((p) => p.type === "subtask")

      // 取专门用来起标题的 agent 配置。
      const ag = yield* agents.get("title")
      if (!ag) return
      // 挑一个模型来干活,从上往下多级 fallback(等价于 a ?? b ?? c ?? d):
      // agent 指定的模型组 → agent 写死的模型 → 这个 provider 的小模型 → 当前对话用的模型。
      // 优先小模型,因为起标题很简单,用便宜快的就够了,省钱。
      const mdl = ag.modelRef
        ? yield* provider.resolveModelRef(ag.modelRef, input.providerID)
        : ag.model
          ? yield* provider.getModel(ag.model.providerID, ag.model.modelID)
          : ((yield* provider.getSmallModel(input.providerID)) ??
            (yield* provider.getModel(input.providerID, input.modelID)))
      // 拼出"喂给模型的消息":全是子任务就把各子任务的 prompt 拼起来;否则把上下文转成模型能吃的格式。
      const msgs = onlySubtasks
        ? [{ role: "user" as const, content: subtasks.map((p) => p.prompt).join("\n") }]
        : yield* MessageV2.toModelMessagesEffect(context, mdl)
      // 流式调模型(像前端 SSE,一个字一个字吐),后面 .pipe(...) 是流处理管道:
      const text = yield* llm
        .stream({
          agent: ag,
          user: firstInfo,
          system: [],
          small: true, // 标记为小任务
          tools: {}, // 不给工具,起标题不需要
          model: mdl,
          sessionID: input.session.id,
          retries: 2, // 失败重试 2 次
          messages: [{ role: "user", content: "Generate a title for this conversation:\n" }, ...msgs],
        })
        .pipe(
          Stream.filter((e): e is Extract<LLM.Event, { type: "text-delta" }> => e.type === "text-delta"), // 只要正文,不要思考
          Stream.map((e) => e.text), // 取出每段文字
          Stream.mkString, // 把碎片拼成完整字符串(相当于 arr.join(""))
          Effect.orDie,
        )
      // 清洗模型输出(模型返回不可信,得洗):
      const cleaned = text
        .replace(/<think>[\s\S]*?<\/think>\s*/g, "") // 去掉混进正文的 <think> 思考标签(对付把思考塞正文的模型)
        .split("\n") // 按行拆
        .map((line) => line.trim()) // 每行去空格
        .find((line) => line.length > 0) // 取第一行非空的当标题
      if (!cleaned) return
      // 太长就截断加省略号(超 100 字符切到 97 + "..."),防止列表标题溢出。
      const t = cleaned.length > 100 ? cleaned.substring(0, 97) + "..." : cleaned
      // 存标题。万一存失败只记错误日志、不往上抛(标题只是辅助功能,失败无所谓)。
      yield* sessions
        .setTitle({ sessionID: input.session.id, title: t })
        .pipe(Effect.catchCause((cause) => elog.error("failed to generate title", { error: Cause.squash(cause) })))
    })

    // ============================================================================
    // predict:预测用户"下一句最可能想说什么",给一个输入建议(类似输入法联想 / Copilot
    // 的灰色预填提示)。
    //
    // ⭐ 关键特征——【旁路调用(bypass)】:它裸调 AI SDK 底层的 generateText,刻意*绕过*
    // 系统封装的 llm.stream 正规通道。因为预测是个"影子请求",不该污染真实对话记录
    // (trajectory),也不该触发与会话耦合的插件钩子。类比前端:正规 request() 带一堆拦截器
    // (记日志/埋点/改历史),而影子请求宁可用底层 fetch 裸调来避开这些副作用。详见下方调用处。
    //
    // 跟 title 是兄弟函数(挑小模型、清洗输出、失败不影响主流程),但目的相反:
    // title 看对话开头起标题,predict 看对话结尾猜下一句。
    // 也是"锦上添花"功能,任何一步不满足就返回空串 ""。
    // ============================================================================
    const predict = Effect.fn("SessionPrompt.predict")(function* (input: { sessionID: SessionID }) {
      // 实验性功能,配置里可以关掉(feature flag)。关了就直接不预测。
      const cfg = yield* config.get()
      if (cfg.experimental?.predict_next_prompt === false) return ""

      // real:同 title,只认真人发的用户消息(排除系统合成的)。
      // 但这里用 findLastIndex 找"最后一条"真人消息——predict 关心的是"你刚说完啥"。
      const history = yield* sessions.messages({ sessionID: input.sessionID, agentID: "main" })
      const real = (m: MessageV2.WithParts) =>
        m.info.role === "user" && !m.parts.every((p) => "synthetic" in p && p.synthetic)
      const userIdx = history.findLastIndex(real)
      if (userIdx === -1) return ""
      const lastUser = history[userIdx]
      if (lastUser.info.role !== "user") return ""

      // 只有真正回答了这条用户消息的那个助手 turn 才算数。
      // 如果那个 turn 仍在运行中（其后跟着一个未完成的助手消息），就直接放弃，
      // 这样我们永远不会把最新的 prompt 和一个陈旧/更早的结果配对。
      // 竞态保护:只有当你最后那句话"已经被完整答完"时才预测下一句。
      // 若这轮还没开始答(没有助手消息)、或还在答(有未 completed 的助手消息),就放弃——
      // 否则会拿一个陈旧/未完成的结果去猜,建议会驴唇不对马嘴。类似前端搜索联想的防抖。
      const assistants = history
        .slice(userIdx + 1)
        .filter((m): m is MessageV2.WithParts & { info: MessageV2.Assistant } => m.info.role === "assistant")
      if (assistants.length === 0) return ""
      if (assistants.some((m) => m.info.time.completed === undefined)) return ""
      const lastAssistant = assistants[assistants.length - 1]

      // 喂给预测的上下文：最近至多 3 条用户查询（按时间顺序）加上最新的助手 turn
      //（它携带了工具输出 + 助手最终文本）。更早的助手 turn 会被丢弃，以保持 prompt 精简。
      const recentUsers = history.filter(real).slice(-3)
      const contextMsgs = [...recentUsers, lastAssistant]

      // 复用 "title" 那个 agent 和同样的多级 fallback 选小模型逻辑(猜下一句也是轻任务,用便宜的就行)。
      const base = yield* agents.get("title")
      if (!base) return ""
      const mdl = base.modelRef
        ? yield* provider.resolveModelRef(base.modelRef, lastAssistant.info.providerID)
        : base.model
          ? yield* provider.getModel(base.model.providerID, base.model.modelID)
          : ((yield* provider.getSmallModel(lastAssistant.info.providerID)) ??
            (yield* provider.getModel(lastAssistant.info.providerID, lastAssistant.info.modelID)))

      // 旁路调用：绕过 llm.stream，使预测不进入会话轨迹，也不会触发与会话耦合的插件钩子
      //（chat.params、chat.headers、system.transform、memory instructions、
      // x-session-affinity）。仍会发布 Metrics.ModelCall，使预测成本体现在分析统计中。
      // ⭐ 与 title 最大的区别:这里裸调 AI SDK 底层的 generateText,而不走系统封装的
      // llm.stream。类比前端:正规的 request() 带一堆拦截器(记日志/埋点/改历史),但预测
      // 是个"影子请求",不想触发那些副作用,于是直接用底层 fetch 裸调。不过 llm.stream 里
      // 那些*必要*的处理(如 ProviderTransform.message 消息格式转换)得自己用中间件补回来。
      const msgs = yield* MessageV2.toModelMessagesEffect(contextMsgs, mdl, { stripMedia: true })
      const language = yield* provider.getLanguage(mdl)
      const wrapped = wrapLanguageModel({
        model: language,
        middleware: [
          {
            specificationVersion: "v3" as const,
            async transformParams(args) {
              if (args.type === "generate" || args.type === "stream") {
                // @ts-expect-error
                args.params.prompt = ProviderTransform.message(args.params.prompt, mdl, {})
              }
              return args.params
            },
          },
        ],
      })
      const started = Date.now()
      const result = yield* Effect.tryPromise(() =>
        generateText({
          model: wrapped,
          system: PREDICT_SYSTEM,
          messages: [...msgs, { role: "user", content: PREDICT_NUDGE }],
          maxOutputTokens: ProviderTransform.maxOutputTokens(mdl),
          temperature: mdl.capabilities.temperature ? 0.7 : undefined,
          providerOptions: ProviderTransform.providerOptions(mdl, ProviderTransform.smallOptions(mdl)),
          headers: {
            ...mdl.headers,
            "User-Agent": `mimocode/${InstallationVersion}`,
          },
          maxRetries: 1,
        }),
      ).pipe(
        Effect.catchCause((cause) =>
          elog.warn("predict failed", { error: Cause.pretty(cause) }).pipe(Effect.as(undefined)),
        ),
      )
      if (!result) return ""

      const u = Session.getUsage({ model: mdl, usage: result.usage, metadata: result.providerMetadata })
      yield* bus
        .publish(Metrics.ModelCall, {
          sessionID: input.sessionID,
          finish_reason: result.finishReason,
          latency_ms: Date.now() - started,
          cached_read_tokens: u.tokens.cache.read,
          model_id: mdl.id,
          provider: mdl.providerID,
          total_tokens_in: u.tokens.input + u.tokens.cache.read + u.tokens.cache.write,
          total_tokens_out: u.tokens.output + u.tokens.reasoning,
        })
        .pipe(Effect.ignore)

      const cleaned = result.text
        .replace(/<think>[\s\S]*?<\/think>\s*/g, "")
        .split("\n")
        .map((line) => line.trim())
        .find((line) => line.length > 0)
      if (!cleaned) return ""
      const stripped = cleaned.replace(quoteTrimRegex, "")
      return stripped.length > 120 ? stripped.substring(0, 117) + "..." : stripped
    })

    // ============================================================================
    // insertReminders:发给模型前的"最后加工车间"。根据当前情境,往对话里偷偷塞一些
    // "系统提醒纸条"(带 synthetic:true 标记的文本 part,界面隐藏、但模型看得到),
    // 引导模型的行为。
    //
    // 为什么要这么做:模型是"无状态"的——它不会自己记住"现在是 plan 模式不能改文件"
    // "用户传了个 Excel 该用某技能"。这些情境规则每轮都得重新、按需塞给它。
    // 类比前端的请求拦截器/AOP:在请求真正发出前,根据上下文往里注入额外内容。
    //
    // 这个函数其实是 4 段互不相关的"要不要塞纸条"判断堆在一起:
    //   ① compose 模式  ② 附件文档→推荐技能  ③ 文本里提及多个技能  ④ plan 模式
    // ============================================================================
    const insertReminders = Effect.fn("SessionPrompt.insertReminders")(function* (input: {
      messages: MessageV2.WithParts[]
      agent: Agent.Info
      session: Session.Info
    }) {
      // 找到最后一条用户消息——纸条基本都往它身上贴。没有就直接原样返回。
      const userMessage = input.messages.findLast((msg) => msg.info.role === "user")
      if (!userMessage) return input.messages

      // ── ① compose 模式:如果这个对话是 compose 模式,把 compose 的技能说明 +
      //    "产物存哪个目录"的指引,塞到消息最前面(unshift=塞头,让模型优先看到)。──
      const composeModeMsg = input.messages.find(
        (msg) => msg.info.role === "user" && msg.info.agent === "compose",
      )
      if (composeModeMsg) {
        const composeModeBlock = composeSkillsBlock()
        const ctx = yield* InstanceState.context
        const composeCfg = (yield* config.get()).compose
        const docsDir = ConfigCompose.resolveDocsDir(ctx.worktree, composeCfg)
        // PROMPT_COMPOSE 主体从 prompt/compose.txt 加载（英文，此处不展开翻译），
        // 里面有两个占位符会被下面替换掉：
        //   {{compose_skills}}  → compose 模式可用技能的说明块
        //   {{compose_docs_dir}} → 下面这句内联英文 prompt，中文对照翻译为：
        //     "保存 compose 技能的产物：规格(specs)存到 `.../specs`、
        //      计划(plans)存到 `.../plans`、报告(reports)存到 `.../reports`。"
        const text = PROMPT_COMPOSE
          .replace("{{compose_skills}}", composeModeBlock)
          .replace("{{compose_docs_dir}}", `Save compose skill outputs: specs in \`${path.join(docsDir, "specs")}\`, plans in \`${path.join(docsDir, "plans")}\`, reports in \`${path.join(docsDir, "reports")}\`.`)
        composeModeMsg.parts.unshift({
          id: PartID.ascending(),
          messageID: composeModeMsg.info.id,
          sessionID: composeModeMsg.info.sessionID,
          type: "text",
          text,
          synthetic: true,
        })
      }

      const assistantMessage = input.messages.findLast((msg) => msg.info.role === "assistant")
      // ── ② 附件文档→推荐技能:如果用户消息里带了办公文档(Excel/Word/PPT 等),
      //    查有没有对应的内置技能能处理它,有就塞张纸条:"用户传了文档,这几个技能可能有用"。
      //    两个 Flag 是开关,可整体关掉这个行为。──
      if (!Flag.MIMOCODE_DISABLE_BUILTIN_SKILLS && !Flag.MIMOCODE_DISABLE_OFFICIAL_SKILLS) {
        // 把用户消息里的文件附件挑出来(只要 file 类型,拿到它的 mime 和文件名)。
        const fileCandidates = userMessage.parts.flatMap((p) => {
          if (p.type !== "file") return []
          const filenameFromSource =
            p.source?.type === "file" && p.source.path ? path.basename(p.source.path) : undefined
          return [{ mime: p.mime, filename: p.filename ?? filenameFromSource }]
        })
        // 根据文件类型匹配出相关技能;匹配到就把它们的 SKILL.md 路径列成一张清单塞进去。
        const skills = matchDocumentSkills(fileCandidates)
        if (skills.length > 0) {
          const root = builtinSkillRoot()
          const entries = skills.map((skill) => `- ${skill}: ${path.join(root, skill, "SKILL.md")}`).join("\n")
          // 下面这段 <system-reminder> 是发给模型的英文 prompt，保持英文原样不动。
          // 中文对照翻译如下（仅供阅读）：
          // ─────────────────────────────────────────────────────────────
          // 【系统提醒】
          // 用户的消息里附带了办公文档文件。下列内置技能可能适用于生成、读取或转换这些文件。
          // 当它契合当前任务时，建议你参考对应的 SKILL.md——在适用的情况下，
          // 优先使用这些技能，而不是自己临时想办法：
          // ${entries：每行一个「- 技能名: 该技能 SKILL.md 的路径」}
          // ─────────────────────────────────────────────────────────────
          const part = yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: userMessage.info.id,
            sessionID: userMessage.info.sessionID,
            type: "text",
            text: `<system-reminder>
The user's message attaches office document file(s). The following built-in skill(s) may be relevant for producing, reading, or transforming these files. You are recommended to consult the SKILL.md when it fits the task — prefer using these skills over ad-hoc approaches when applicable:
${entries}
</system-reminder>`,
            synthetic: true,
          })
          userMessage.parts.push(part)
        }
      }

      // ── ③ 文本里提及多个技能:用户在一句话里手打了多个 /技能名(如
      //    "先用 /research 调研,再用 /summary 总结")。这里把这些技能的说明书(SKILL.md)
      //    自动加载塞给模型;若提到 2 个以上,再额外教模型"怎么把多个技能配合起来用"。──

      // 步骤0——去重保护:如果消息里已经有 <skill_content> 块了(说明走斜杠命令那条路已经
      // 包过一次),整段跳过,别重复加载同一个技能。
      const alreadyWrapped = userMessage.parts.some(
        (p) => p.type === "text" && p.text.startsWith('<skill_content name="'),
      )
      if (!alreadyWrapped) {
        // Use all() to include hidden skills (primarily compose:*) — respect the user's explicit /mention action
        // 用 all() 以包含隐藏技能(主要是 compose:*)——尊重用户显式的 /提及 动作
        const allSkills = yield* sys.all()
        if (allSkills.length > 0) {
          // 步骤1——把用户消息的文本拼起来,并"净化":抠掉 ```代码块``` 和 `行内代码`。
          // 目的:防止用户贴的代码里正好有个 "/research" 字符串被误当成技能引用。
          const bodyText = userMessage.parts
            .flatMap((p) => (p.type === "text" ? [p.text] : []))
            .join("\n")
          const stripped = bodyText
            .replace(/```[\s\S]*?```/g, " ")
            .replace(/`[^`\n]*`/g, " ")
          // 步骤2——正则扫出真正的技能提及。
          // 正则含义:前面是行首或空格 + 斜杠 + 技能名(字母开头);后面必须是非标识符字符或结尾,
          // 保证完整匹配一个词。两个过滤:去重(seen) + 必须是真实存在的技能(availableSkills)。
          const mentioned: string[] = []
          const seen = new Set<string>()
          const mentionRe = /(?:^|\s)\/([A-Za-z][A-Za-z0-9_:-]*)(?=[^A-Za-z0-9_:-]|$)/g
          for (const m of stripped.matchAll(mentionRe)) {
            const name = m[1]
            if (!name || seen.has(name)) continue
            if (!allSkills.some((s) => s.name === name)) continue
            seen.add(name)
            mentioned.push(name)
          }

          if (mentioned.length > 0) {
            // 步骤3——加载技能内容,但设上限:只自动加载前 3 个的完整说明书(每个 SKILL.md
            // 可能几百上千 token,全塞会爆上下文/烧钱)。超过 3 个的(overflow)留给模型按需自取。
            const MAX_AUTOLOAD = 3
            const toLoad = mentioned.slice(0, MAX_AUTOLOAD)
            const overflow = mentioned.slice(MAX_AUTOLOAD)
            for (const name of toLoad) {
              const info = allSkills.find((s) => s.name === name)
              if (!info) continue
              // 用 <skill_content> 标签包裹内容(给模型划清边界,防内容串味)。
              const part = yield* sessions.updatePart({
                id: PartID.ascending(),
                messageID: userMessage.info.id,
                sessionID: userMessage.info.sessionID,
                type: "text",
                text: `<skill_content name="${name}">\n${info.content}\n</skill_content>`,
                synthetic: true,
              })
              userMessage.parts.push(part)
            }

            // 步骤4——提到 >=2 个技能时,额外塞一份"编排方法论"(1 个不需要编排)。
            // 下面的 5 步 checklist 是把"专家怎么组合多个技能"的思路显式教给模型:
            // 先读全文档再规划→判断技能间关系(流水线/并行/约束叠加)→定中间产物格式→
            // 冲突时定优先级→先出工作流再执行。loadedHint/overflowHint 动态告诉模型
            // 哪些已加载好、哪些(超 3 个的)要自己调 Skill 工具去取。
            if (mentioned.length >= 2) {
              const loadedHint = toLoad.length > 0
                ? `SKILL.md for [${toLoad.join(", ")}] has been auto-loaded above.`
                : ""
              const overflowHint = overflow.length > 0
                ? `For [${overflow.join(", ")}], use the Skill tool to load them on demand.`
                : ""
              // 下面这段 <system-reminder> 是发给模型的英文 prompt，保持英文原样不动。
              // 中文对照翻译如下（仅供阅读，勿把注释内容当作真正发送的文本）：
              // ─────────────────────────────────────────────────────────────
              // 【系统提醒】
              // 用户在这条消息里显式引用了多个技能：${mentioned 里的技能名}。
              // ${loadedHint：[已加载的技能] 的 SKILL.md 已在上方自动加载。}
              // ${overflowHint：对于 [超出上限的技能]，请用 Skill 工具按需自行加载。}
              //
              // 动手之前，先完成一份"编排计划"：
              //   1. 先把每一个被引用技能的 SKILL.md 完整读一遍，再做规划
              //      （绝不能只凭技能的简短描述就规划——完整的 SKILL.md 里可能有约束，
              //       会直接推翻你凭空想象出来的工作流）。
              //   2. 判断这些技能之间的"组合关系"属于哪一种：
              //        · 流水线 pipeline —— A 的产出作为 B 的输入
              //        · 并行 parallel   —— 各自负责互不相干的一部分
              //        · 约束叠加 overlay —— 一个技能干活，另一个只提供规则/标准
              //   3. 若是流水线：为中间产物定义"接口契约"——即格式和文件路径。
              //   4. 若两个技能在同一维度上都给了指示（输出格式 / 风格 / 流程），
              //      要显式声明一条"冲突消解规则"：在哪个维度上以哪个技能为准。
              //   5. 输出一份精简的工作流（阶段 → 用到的技能 → 产物），然后照它执行。
              //
              // 规划的详略要与任务复杂度匹配：简单的组合，两三句话就够了。
              // ─────────────────────────────────────────────────────────────
              const part = yield* sessions.updatePart({
                id: PartID.ascending(),
                messageID: userMessage.info.id,
                sessionID: userMessage.info.sessionID,
                type: "text",
                text: `<system-reminder>
The user has explicitly referenced multiple skills in this message: ${mentioned.join(", ")}.
${loadedHint} ${overflowHint}

Before starting work, complete an orchestration plan:
1. Read the SKILL.md of every referenced skill FIRST, then plan (never plan from skill descriptions alone — the full SKILL.md may contain constraints that invalidate an imagined workflow)
2. Classify the composition relationship: pipeline (A's output → B's input) / parallel (each handles a separate part) / constraint overlay (one does the work, the other provides rules or standards)
3. If pipeline: define the interface contract for intermediate artifacts — format and file path
4. If two skills give instructions on the same dimension (output format / style / process), explicitly declare a conflict resolution rule: which skill takes precedence on which dimension
5. Output a concise workflow (phase → skill used → artifact), then execute according to it

Keep planning proportional to task complexity: for simple combinations, two or three sentences suffice.
</system-reminder>`,
                synthetic: true,
              })
              userMessage.parts.push(part)
            }
          }
        }
      }

      // ── ④ plan 模式(计划模式:让 agent 先研究和设计、别急着改代码)。分两种情况: ──

      // 情况A——"刚从 plan 切到执行":当前 agent 不是 plan,但上一条助手消息是 plan 产出的。
      // 说明用户看完计划、切去执行了。若存在 plan 文件,塞张纸条:"有个计划文件,照着它去执行"。
      if (input.agent.name !== "plan" && assistantMessage?.info.agent === "plan") {
        const plan = Session.plan(input.session)
        if (!(yield* fsys.existsSafe(plan))) return input.messages
        // 下面拼给模型的 prompt 保持英文原样不动。中文对照翻译如下（仅供阅读）：
        // ─────────────────────────────────────────────────────────────
        // ${BUILD_SWITCH}：从 prompt/build-switch.txt 加载的"从 plan 切到执行"引导语
        //                （告诉模型：计划阶段已结束，现在开始动手执行）。
        //
        // 在 ${plan} 路径存在一个计划文件。你应当按照文件里定义的计划去执行。
        // ─────────────────────────────────────────────────────────────
        const part = yield* sessions.updatePart({
          id: PartID.ascending(),
          messageID: userMessage.info.id,
          sessionID: userMessage.info.sessionID,
          type: "text",
          text: `${BUILD_SWITCH}\n\nA plan file exists at ${plan}. You should execute on the plan defined within it`,
          synthetic: true,
        })
        userMessage.parts.push(part)
        return input.messages
      }

      // 不在 plan 模式(且上面情况A也不成立)→ 无事可做,原样返回。
      if (input.agent.name !== "plan" || assistantMessage?.info.agent === "plan") return input.messages

      // 情况B——"正处于 plan 模式":塞一大段行为约束纸条,把模型降级成"只读研究"状态:
      // 只能读文件/搜索、只能写 plan 文件,禁止改其他文件、禁止跑 test/lint/build 等有副作用的命令。
      // 类比前端的"只读模式/权限降级"。先确保 plan 文件所在目录存在。
      const plan = Session.plan(input.session)
      const exists = yield* fsys.existsSafe(plan)
      if (!exists) yield* fsys.ensureDir(path.dirname(plan)).pipe(Effect.catch(Effect.die))
      // 下面这段 <system-reminder> 是发给模型的英文 prompt，保持英文原样不动。
      // 中文对照翻译如下（仅供阅读，勿把注释内容当作真正发送的文本）：
      // ═════════════════════════════════════════════════════════════════
      // 【系统提醒】
      // plan（计划）模式已激活。用户希望你先做调研和设计，暂时【不要】动手执行。
      // 本条指令优先级高于你此前收到的任何其它指令。
      //
      // ## 你【应该】做什么（推荐）
      // - 凡是专用只读工具能覆盖的，优先用它们：`read`（看文件）、`grep`（搜内容）、
      //   `glob`（找文件），以及 `lsp` 系列工具（定义、引用、诊断）。这才是探索代码的正道。
      // - 派生 `explore` / `general` 子 agent 来并行调研。
      // - 只有当上述工具确实拿不到你要的信息时，才【可以】用 `bash` 来补这个缺口——
      //   但仅限你确信是【纯只读、无任何副作用】的命令（例如 `git status`/`log`/`diff`、
      //   列依赖）。不要用 `bash` 去干 `read`/`grep`/`glob` 已经能干的事。
      //
      // ## 你【绝对不能】做什么
      // - 不要编辑或创建除下面那个 plan 文件之外的任何文件。对非 plan 文件的写入会被
      //   直接拦截并失败——别尝试，也别让用户去审批。
      // - 不要运行 `test`、`lint`、`typecheck`、`build` 或类似的项目命令。它们默认【不安全】：
      //   `lint` 常配了 `--fix`，`test` 可能写快照或动数据库，`build` 会产出构建产物，
      //   背后的脚本什么都干得出来。唯一的例外：你已经【明确核实过】——通过读取确切的
      //   命令/配置——这一次具体调用没有副作用（没有 `--fix`/`--write`，不改文件/状态/数据库）。
      //   若无法核实，就当它是被禁止的，改为把它记录到计划里。
      // - 不要运行任何其它有副作用的 `bash`：不提交、不 `git push`、不装/卸包、
      //   不写/移动/删除文件、不改配置、不 `change_directory`、不 `workflow`。
      // - 如果你发现自己为了推进而想去改动某个东西，这正是一个信号——把它写进计划，
      //   然后继续用只读方式调研。
      //
      // 请用好的判断力：自己去做只读操作，而不是把本可避免的确认弹窗甩给用户。
      // 只有 plan 文件是可写的。
      //
      // ## 计划文件信息：
      // ${exists ? "计划文件已存在于 ${plan}，你可以读取它并用 edit 工具做增量修改。"
      //          : "还没有计划文件，你应当用 write 工具在 ${plan} 处创建你的计划。"}
      // 你应当通过写入/编辑这个文件来逐步构建你的计划。注意：这是你唯一被允许编辑的文件，
      // 除此之外你只能采取【只读】操作。
      //
      // ## 计划工作流
      //
      // ### 阶段 1：初步理解
      // 目标：通过通读代码 + 向用户提问，全面理解用户的诉求。关键：本阶段你只能用 explore 这种子 agent。
      //   1. 聚焦于理解用户诉求，以及与其诉求相关的代码。
      //   2. 【并行启动至多 3 个 explore agent】（同一条消息里发多个工具调用）来高效探索代码库。
      //      - 任务局限于已知文件、用户给了明确路径、或只做小范围定向改动时，用 1 个就够。
      //      - 范围不确定、涉及代码库多处、或需要先摸清既有模式再规划时，才用多个。
      //      - 重质不重量——最多 3 个，且应尽量用最少的数量（通常 1 个即可）。
      //      - 若用多个：给每个 agent 指定明确的搜索焦点或探索区域。例如：一个查已有实现，
      //        另一个探索相关组件，第三个调查测试模式。
      //   3. 探索完代码后，用 question 工具就用户诉求中的模糊点提前向用户澄清。
      //
      // ### 阶段 2：设计
      // 目标：设计一套实现方案。
      // 基于用户意图和阶段 1 的探索结果，启动 general agent 来设计实现。最多并行 1 个。
      //   【准则】
      //   - 默认：大多数任务都至少启动 1 个 Plan agent——它有助于验证你的理解、权衡备选方案。
      //   - 跳过 agent：仅限真正琐碎的任务（改错别字、单行改动、简单重命名）。
      //   何时用多个 agent 的例子：任务涉及代码库多个部分 / 大型重构或架构变更 /
      //   边界情况很多 / 你能从探索不同方案中获益。
      //   按任务类型的视角举例：
      //     - 新功能：简单性 vs 性能 vs 可维护性
      //     - 修 bug：根因 vs 绕过 vs 预防
      //     - 重构：最小改动 vs 干净架构
      //   在给 agent 的 prompt 里：提供阶段 1 探索得到的完整背景（含文件名和代码路径追踪）；
      //   描述需求与约束；要求它给出详细的实现计划。
      //
      // ### 阶段 3：评审
      // 目标：评审阶段 2 的计划，确保与用户意图对齐。
      //   1. 阅读 agent 指出的关键文件，加深理解。
      //   2. 确保计划与用户最初的诉求一致。
      //   3. 用 question 工具就剩余疑问向用户澄清。
      //
      // ### 阶段 4：最终计划
      // 目标：把你的最终计划写进 plan 文件（你唯一能编辑的文件）。
      //   - 只包含你推荐的方案，不要罗列所有备选。
      //   - 确保计划文件既能快速扫读、又足够详细可执行。
      //   - 包含将被修改的关键文件路径。
      //   - 包含一个"验证"小节，描述如何端到端地测试改动（跑代码、用 MCP 工具、跑测试）。
      //
      // ### 阶段 5：调用 plan_exit 工具
      // 在你这一轮的最末尾，一旦你已向用户提过问且对最终计划文件满意，就总是调用 plan_exit，
      // 向用户表明你计划完毕。这至关重要——你这一轮的结束只能是两种情况之一：向用户提问，
      // 或调用 plan_exit。除这两个理由外不要停下。
      //
      // 【重要】用 question 工具澄清需求/方案；用 plan_exit 请求对计划的批准。
      // 不要用 question 工具去问"这个计划行不行？"——那是 plan_exit 干的事。
      //
      // 注意：这套工作流的任何时刻，你都可以随时向用户提问或澄清。不要对用户意图做大的假设。
      // 目标是给用户呈上一份调研充分的计划，并在动手实现前把各种悬而未决的点收尾。
      // ═════════════════════════════════════════════════════════════════
      const part = yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: userMessage.info.id,
        sessionID: userMessage.info.sessionID,
        type: "text",
        text: `<system-reminder>
Plan mode is active. The user wants you to research and design, NOT to execute yet. This supersedes any other instructions you have received.

## What you SHOULD do (recommended)
- Prefer the dedicated read-only tools for everything they cover — \`read\` (view files), \`grep\` (search contents), \`glob\` (find files), and the \`lsp\` tools (definitions, references, diagnostics). These are the right way to explore the code.
- Spawn \`explore\`/\`general\` subagents for parallel research.
- Only when those tools genuinely can't get what you need, you MAY use \`bash\` for the gap — but ONLY for commands you are certain are a pure read with NO side effects (e.g. \`git status\`/\`log\`/\`diff\`, listing dependencies). Do NOT reach for \`bash\` to do what \`read\`/\`grep\`/\`glob\` already do.

## What you MUST NOT do
- Do NOT edit or create any file other than the plan file below. Writes to non-plan files are blocked outright and will fail — do not attempt them and do not ask the user to approve them.
- Do NOT run \`test\`, \`lint\`, \`typecheck\`, \`build\`, or similar project commands. These are NOT safe by default: \`lint\` is often configured with \`--fix\`, \`test\` may write snapshots or touch a database, \`build\` writes artifacts, and scripts behind them can do anything. The ONLY exception is if you have explicitly verified — by reading the exact command/config — that this specific invocation has no side effects (no \`--fix\`/\`--write\`, no file/state/db mutation). If you cannot verify that, treat it as forbidden and note it in the plan instead.
- Do NOT run any other side-effecting \`bash\`: no commits, no \`git push\`, no installing/removing packages, no writing/moving/deleting files, no changing configs, no \`change_directory\`, no \`workflow\`.
- If you find yourself wanting to mutate something to make progress, that's a signal to write it into the plan instead and continue researching read-only.

Use good judgment: take the read-only action yourself rather than pushing avoidable confirmation prompts onto the user. Only the plan file is writable.

## Plan File Info:
${exists ? `A plan file already exists at ${plan}. You can read it and make incremental edits using the edit tool.` : `No plan file exists yet. You should create your plan at ${plan} using the write tool.`}
You should build your plan incrementally by writing to or editing this file. NOTE that this is the only file you are allowed to edit - other than this you are only allowed to take READ-ONLY actions.

## Plan Workflow

### Phase 1: Initial Understanding
Goal: Gain a comprehensive understanding of the user's request by reading through code and asking them questions. Critical: In this phase you should only use the explore subagent type.

1. Focus on understanding the user's request and the code associated with their request

2. **Launch up to 3 explore agents IN PARALLEL** (single message, multiple tool calls) to efficiently explore the codebase.
 - Use 1 agent when the task is isolated to known files, the user provided specific file paths, or you're making a small targeted change.
 - Use multiple agents when: the scope is uncertain, multiple areas of the codebase are involved, or you need to understand existing patterns before planning.
 - Quality over quantity - 3 agents maximum, but you should try to use the minimum number of agents necessary (usually just 1)
 - If using multiple agents: Provide each agent with a specific search focus or area to explore. Example: One agent searches for existing implementations, another explores related components, a third investigates testing patterns

3. After exploring the code, use the question tool to clarify ambiguities in the user request up front.

### Phase 2: Design
Goal: Design an implementation approach.

Launch general agent(s) to design the implementation based on the user's intent and your exploration results from Phase 1.

You can launch up to 1 agent(s) in parallel.

**Guidelines:**
- **Default**: Launch at least 1 Plan agent for most tasks - it helps validate your understanding and consider alternatives
- **Skip agents**: Only for truly trivial tasks (typo fixes, single-line changes, simple renames)

Examples of when to use multiple agents:
- The task touches multiple parts of the codebase
- It's a large refactor or architectural change
- There are many edge cases to consider
- You'd benefit from exploring different approaches

Example perspectives by task type:
- New feature: simplicity vs performance vs maintainability
- Bug fix: root cause vs workaround vs prevention
- Refactoring: minimal change vs clean architecture

In the agent prompt:
- Provide comprehensive background context from Phase 1 exploration including filenames and code path traces
- Describe requirements and constraints
- Request a detailed implementation plan

### Phase 3: Review
Goal: Review the plan(s) from Phase 2 and ensure alignment with the user's intentions.
1. Read the critical files identified by agents to deepen your understanding
2. Ensure that the plans align with the user's original request
3. Use question tool to clarify any remaining questions with the user

### Phase 4: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- Include only your recommended approach, not all alternatives
- Ensure that the plan file is concise enough to scan quickly, but detailed enough to execute effectively
- Include the paths of critical files to be modified
- Include a verification section describing how to test the changes end-to-end (run the code, use MCP tools, run tests)

### Phase 5: Call plan_exit tool
At the very end of your turn, once you have asked the user questions and are happy with your final plan file - you should always call plan_exit to indicate to the user that you are done planning.
This is critical - your turn should only end with either asking the user a question or calling plan_exit. Do not stop unless it's for these 2 reasons.

**Important:** Use question tool to clarify requirements/approach, use plan_exit to request plan approval. Do NOT use question tool to ask "Is this plan okay?" - that's what plan_exit does.

NOTE: At any point in time through this workflow you should feel free to ask the user questions or clarifications. Don't make large assumptions about user intent. The goal is to present a well researched plan to the user, and tie any loose ends before implementation begins.
</system-reminder>`,
        synthetic: true,
      })
      userMessage.parts.push(part)
      return input.messages
    })

    // ============================================================================
    // resolveTools：把"工具定义"装配成大模型 SDK(AI SDK)能识别、能调用的工具对象。
    //
    // 输入：一个 agent + 一个 model + 当前会话；输出：{ 工具名: AITool } 字典，最终发给模型。
    // 每个 AITool 里封了两样东西：
    //   · inputSchema —— 现在就算好、随请求发给模型的参数 schema(模型据此决定怎么调)
    //   · execute     —— 一个闭包,模型将来决定调用时 AI SDK 才回调它,里面套满了
    //                    "白名单→前置钩子→真正执行→后置钩子→埋点"这条中间件洋葱。
    //
    // 类比前端:很像 tRPC/GraphQL 的 resolver 装配层——把业务函数包装成带 schema、
    // 带鉴权中间件、带日志的可调用端点。
    //
    // 函数结构:
    //   ① 准备阶段:算出本次调用的"规矩"(whitelist / askRouting / rejectionFor / context)
    //   ② 循环A:包装原生工具(read/edit/bash/grep…)
    //   ③ 循环B:包装 MCP 工具(外部 MCP server 提供的)
    // ============================================================================
    const resolveTools = Effect.fn("SessionPrompt.resolveTools")(function* (input: {
      agent: Agent.Info
      model: Provider.Model
      session: Session.Info
      tools?: Record<string, boolean>
      processor: Pick<SessionProcessor.Handle, "message" | "updateToolCall" | "completeToolCall">
      bypassAgentCheck: boolean
      messages: MessageV2.WithParts[]
      agentID?: string
      task_id?: string
    }) {
      using _ = log.time("resolveTools") // 记录本函数耗时(块结束自动结算,using 是显式资源管理)
      const tools: Record<string, AITool> = {} // 最终产物:工具字典,边装配边往里塞
      const run = yield* runner() // Effect→Promise 的桥:AI SDK 的 execute 要 Promise,而我们逻辑写在 Effect 里
      const promptOps = yield* ops() // 暴露给工具用的能力(prompt/cancel 等),放进每个工具的 ctx.extra

      // 按工具的运行时白名单：当 LLM 调用是代表一个已注册的 actor（子 agent 或对等 agent）
      // 发起时，查找该 actor 记录；如果 `actor.tools` 是数组，则拒绝调用不在白名单里的工具。
      // `INHERIT` 和缺失的 actor 记录都表示完全放行。
      const whitelistFor = Effect.fn("SessionPrompt.whitelistFor")(function* () {
        if (!input.agentID) return undefined
        const actor = yield* actorRegistry.get(input.session.id, input.agentID)
        if (!actor || !Array.isArray(actor.tools)) return undefined
        return new Set(actor.tools)
      })
      const whitelist = yield* whitelistFor()
      // 权限询问是否必须是非交互式的（干净失败，绝不挂起）：对系统派生的 actor
      //（checkpoint-writer/dream/distill）以及任何后台 actor（例如以 "general" +
      // background:true 派生的 compose 工作流子 agent）为 true。有意只作用于*本次*权限
      // 决策——不并入共享的 isSystemSpawned，后者还会为用户后台 actor 把关 memory
      // instructions 和 checkpoint 自触发。如果 actor 记录缺失（竞态 / 未注册），
      // 回退到按 agent 名判断，这样系统 actor 就不会被当作交互式而漏过。
      const askActor = input.agentID
        ? yield* actorRegistry.get(input.session.id, input.agentID)
        : undefined
      // 三路权限询问路由（见 decideAskRouting）：系统 agent -> 自动拒绝；
      // 编排器对等 agent -> 转发（FORWARD）以请求审批；其他后台 -> 自动拒绝；
      // 普通 -> 交互式。
      const askRouting = decideAskRouting({
        askActor: askActor
          ? {
              agent: askActor.agent,
              background: askActor.background,
              mode: askActor.mode,
              parentActorID: askActor.parentActorID,
            }
          : undefined,
        sessionParentID: input.session.parentID,
        agentName: input.agent.name,
        orchestratorEnabled: Flag.MIMOCODE_EXPERIMENTAL_ORCHESTRATOR,
      })
      const askInteractive = askRouting.interactive // 权限询问:是否弹窗问真人
      const askForward = askRouting.forward // 权限询问:是否转发给上级审批
      const askInherit = askRouting.inherit // 权限询问:是否继承父级已持有的授权
      // rejectionFor:被白名单拦下时返回的统一"拒绝"输出(附上允许的工具清单,便于排查)。
      const rejectionFor = (toolID: string) => ({
        title: "Tool not permitted",
        output: `The "${toolID}" tool is not in this actor's whitelist. Allowed tools: ${
          whitelist ? [...whitelist].join(", ") : "(none)"
        }.`,
        metadata: { rejected: true, reason: "tool-whitelist" as const },
      })

      // context:工厂函数,每次工具执行时用当次的 args/options 造一个新的 Tool.Context 注入进去。
      // 类比后端框架里每个请求 new 一个 RequestContext。工具通过它拿到会话信息 + 两个关键回调:
      //   · metadata(val) —— 工具执行中回传进度/标题给 UI(如 bash 边跑边刷输出)
      //   · ask(req)      —— 工具要权限时调它(权限判定最终都汇聚到 Agent.runtimePermission)
      // 工具本身不知道"怎么更新UI/怎么问权限",这些实现被注入进来(依赖注入/控制反转)。
      const context = (args: any, options: ToolExecutionOptions): Tool.Context => ({
        sessionID: input.session.id,
        abort: options.abortSignal!,
        messageID: input.processor.message.id,
        callID: options.toolCallId,
        extra: { model: input.model, bypassAgentCheck: input.bypassAgentCheck, promptOps },
        agent: input.agent.name,
        actorID: input.agentID,
        taskId: input.task_id,
        messages: input.messages,
        // metadata:工具执行中回传"标题/进度"→更新对应 tool part 的状态(仅当它还在 running/pending)。
        metadata: (val) =>
          input.processor.updateToolCall(options.toolCallId, (match) => {
            if (!["running", "pending"].includes(match.state.status)) return match
            return {
              ...match,
              state: {
                title: val.title,
                metadata: val.metadata,
                status: "running",
                input: args,
                time: { start: Date.now() },
              },
            }
          }),
        ask: (req) =>
          permission
            .ask(
              {
                ...req,
                sessionID: input.session.id,
                tool: { messageID: input.processor.message.id, callID: options.toolCallId },
                ruleset: Agent.runtimePermission(input.agent, input.session.permission),
                // System-spawned + non-peer background agents have no human to answer
                // → fail clean, don't hang. Orchestrator peers FORWARD for approval;
                // ordinary background subagents INHERIT the parent's held grants.
                // 系统派生 + 非对等的后台 agent 没有人类来回答 → 干净失败,不要挂起。
                // 编排器对等 agent 转发(FORWARD)请求审批;普通后台子 agent 则继承(INHERIT)父级已持有的授权。
                interactive: askInteractive,
                ...(askForward ? { forward: askForward } : {}),
                ...(askInherit ? { inherit: askInherit } : {}),
              },
              options.abortSignal,
            )
            .pipe(Effect.orDie),
      })

      // ── 循环A:包装原生工具 ──────────────────────────────────────────────
      // registry.tools(...) 已按 agent/model 筛选并定制好描述(见 tool/registry.ts),
      // 这里把每个"工具定义"包装成 AI SDK 的 tool({ inputSchema, execute })。
      for (const item of yield* registry.tools({
        modelID: ModelID.make(input.model.api.id),
        providerID: input.model.providerID,
        agent: input.agent,
      })) {
        // Zod 参数 → JSON Schema → 按不同模型商适配。一份定义,三种用途(TS类型/运行时校验/给模型的schema)。
        const schema = ProviderTransform.schema(input.model, z.toJSONSchema(item.parameters))
        tools[item.id] = tool({
          description: item.description, // 工具说明书(给模型看)
          inputSchema: jsonSchema(schema), // 参数 schema(给模型看)——"现在"就发出去
          // execute:闭包,模型"将来"决定调用此工具时 AI SDK 才回调。用 run.promise 把 Effect 转成 Promise。
          execute(args, options) {
            return run.promise(
              Effect.gen(function* () {
                const startTs = Date.now()
                const callID = options?.toolCallId ?? "?"
                log.debug("tool execute start", {
                  tool: item.id,
                  callID,
                  sessionID: input.session.id,
                })
                const ctx = context(args, options) // 造本次执行的上下文
                // 步骤1——白名单拦截:不在这个 actor 的工具白名单里 → 直接拒绝、落库、返回。
                if (whitelist && !whitelist.has(item.id)) {
                  const output = rejectionFor(item.id)
                  log.debug("tool execute rejected", {
                    tool: item.id,
                    callID,
                    durationMs: Date.now() - startTs,
                  })
                  yield* input.processor.completeToolCall(options.toolCallId, output)
                  return output
                }
                // 步骤2——前置插件钩子:插件可在此改参数(beforeOutput.args)或取消调用(cancel)。类比 axios 请求拦截器。
                const beforeOutput: { args: any; cancel?: boolean; cancelReason?: string } = { args }
                yield* plugin.trigger(
                  "tool.execute.before",
                  { tool: item.id, sessionID: ctx.sessionID, callID: ctx.callID },
                  beforeOutput,
                )
                // 若插件要求取消 → 记 cancelled 埋点、落库、返回,不执行工具。
                if (beforeOutput.cancel) {
                  const cancelOutput = {
                    title: "Cancelled",
                    output: beforeOutput.cancelReason || "Tool call cancelled by hook",
                    metadata: { cancelled: true },
                  }
                  yield* bus
                    .publish(Metrics.ToolCall, {
                      sessionID: ctx.sessionID,
                      tool_name: item.id,
                      input_bytes: Metrics.jsonByteLength(beforeOutput.args),
                      output_bytes: 0,
                      tool_call_id: options.toolCallId,
                      tool_call_status: "cancelled",
                    })
                    .pipe(Effect.ignore)
                  yield* input.processor.completeToolCall(options.toolCallId, cancelOutput)
                  return cancelOutput
                }
                // 步骤3——★真正执行工具本体(用可能被插件改过的 beforeOutput.args)。
                const result = yield* item.execute(beforeOutput.args, ctx)
                log.debug("tool execute done", {
                  tool: item.id,
                  callID,
                  durationMs: Date.now() - startTs,
                  ok: true,
                })
                // 给结果里的附件补齐身份字段(id/会话/消息),挂靠到当前消息上。
                const output = {
                  ...result,
                  attachments: result.attachments?.map((attachment) => ({
                    ...attachment,
                    id: PartID.ascending(),
                    sessionID: ctx.sessionID,
                    messageID: input.processor.message.id,
                  })),
                }
                // 步骤4——后置插件钩子:插件可后处理结果。类比 axios 响应拦截器。
                yield* plugin.trigger(
                  "tool.execute.after",
                  { tool: item.id, sessionID: ctx.sessionID, callID: ctx.callID, args: beforeOutput.args },
                  output,
                )
                // 步骤5——自我进化:如果这次是 write/edit 改动了 .mimocode/tools|skills 下的
                // 扩展定义文件,就热重载工具注册表,让新工具/技能立即生效。
                if (
                  (item.id === "write" || item.id === "edit") &&
                  beforeOutput.args?.file_path &&
                  isExtensionPath(beforeOutput.args.file_path)
                ) {
                  yield* registry.reload().pipe(Effect.tapError((err) => Effect.sync(() => log.warn("extension reload failed", { error: err }))), Effect.ignore)
                }
                // 步骤6——发埋点:记录本次工具调用的耗时/输入输出字节数/状态(success)。
                yield* bus
                  .publish(Metrics.ToolCall, {
                    sessionID: ctx.sessionID,
                    tool_name: item.id,
                    input_bytes: Metrics.jsonByteLength(beforeOutput.args),
                    output_bytes: Buffer.byteLength(output.output ?? "", "utf8"),
                    tool_call_id: options.toolCallId,
                    tool_call_status: "success",
                  })
                  .pipe(Effect.ignore)
                // 兜底:若执行完发现已被取消,主动落库结果,避免工具状态永远卡在 running(僵尸态)。
                if (options.abortSignal?.aborted) {
                  yield* input.processor.completeToolCall(options.toolCallId, output)
                }
                return output
              }),
            )
          },
        })
      }

      // ── 循环B:包装 MCP 工具(外部 MCP server 提供的)──────────────────────
      // 中间件结构与循环A几乎一样(白名单→前置钩子→执行→后置钩子→埋点),差异有二:
      //   1. 权限统一走 ctx.ask(粒度粗,一律询问);
      //   2. 返回是 result.content 数组,混着 text/image/resource,需要拆开处理。
      for (const [key, item] of Object.entries(yield* mcp.tools())) {
        const execute = item.execute
        if (!execute) continue // 没有执行体的(如纯提示型)跳过

        // MCP 工具自带 schema,这里取出→按模型适配→写回,保证发给模型的 schema 格式正确。
        const schema = yield* Effect.promise(() => Promise.resolve(asSchema(item.inputSchema).jsonSchema))
        const transformed = ProviderTransform.schema(input.model, schema)
        item.inputSchema = jsonSchema(transformed)
        // 覆写 execute:套上和循环A同款的中间件洋葱(原地改写 item.execute)。
        item.execute = (args, opts) =>
          run.promise(
            Effect.gen(function* () {
              const startTs = Date.now()
              const callID = opts?.toolCallId ?? "?"
              log.debug("tool execute start (mcp)", {
                tool: key,
                callID,
                sessionID: input.session.id,
              })
              const ctx = context(args, opts)
              // 白名单拦截(MCP 版):不在白名单 → 拒绝(注意 MCP 的输出结构多一个 content 字段)。
              if (whitelist && !whitelist.has(key)) {
                const rejection = rejectionFor(key)
                const output = {
                  title: rejection.title,
                  metadata: rejection.metadata,
                  output: rejection.output,
                  attachments: [],
                  content: [{ type: "text" as const, text: rejection.output }],
                }
                log.debug("tool execute rejected (mcp)", {
                  tool: key,
                  callID,
                  durationMs: Date.now() - startTs,
                })
                yield* input.processor.completeToolCall(opts.toolCallId, output)
                return output
              }
              const mcpBeforeOutput: { args: any; cancel?: boolean; cancelReason?: string } = { args }
              yield* plugin.trigger(
                "tool.execute.before",
                { tool: key, sessionID: ctx.sessionID, callID: opts.toolCallId },
                mcpBeforeOutput,
              )
              // 前置钩子要求取消 → 记 cancelled 埋点后直接返回,不执行 MCP 工具。
              if (mcpBeforeOutput.cancel) {
                const cancelResult = {
                  content: [{ type: "text" as const, text: mcpBeforeOutput.cancelReason || "Tool call cancelled by hook" }],
                }
                yield* bus
                  .publish(Metrics.ToolCall, {
                    sessionID: ctx.sessionID,
                    tool_name: key,
                    input_bytes: Metrics.jsonByteLength(mcpBeforeOutput.args),
                    output_bytes: 0,
                    tool_call_id: opts.toolCallId,
                    tool_call_status: "cancelled",
                  })
                  .pipe(Effect.ignore)
                return cancelResult
              }
              // MCP 工具统一走权限询问(粒度粗:patterns/always 都是 "*")。
              yield* ctx.ask({ permission: key, metadata: {}, patterns: ["*"], always: ["*"] })
              // ★真正执行 MCP 工具(它是 Promise 接口,用 Effect.promise 包起来)。
              const result: Awaited<ReturnType<NonNullable<typeof execute>>> = yield* Effect.promise(() =>
                execute(mcpBeforeOutput.args, opts),
              )
              yield* plugin.trigger(
                "tool.execute.after",
                { tool: key, sessionID: ctx.sessionID, callID: opts.toolCallId, args },
                result,
              )

              const normalized = normalizeToolResult(result)
              log.debug("tool execute done (mcp)", {
                tool: key,
                callID,
                durationMs: Date.now() - startTs,
                ok: !normalized.isError,
              })

              const truncated = yield* truncate.output(
                normalized.output,
                { outcome: normalized.isError ? "error" : "success" },
                input.agent,
              )
              const metadata = {
                ...normalized.metadata,
                truncated: truncated.truncated,
                ...(truncated.truncated && { outputPath: truncated.outputPath }),
              }
              const attachments = normalized.attachments.map((attachment) => ({
                type: "file" as const,
                ...attachment,
                id: PartID.ascending(),
                sessionID: ctx.sessionID,
                messageID: input.processor.message.id,
              }))

              if (normalized.isError) {
                return yield* Effect.fail(
                  new ToolResultError(
                    truncated.content.trim() || "MCP tool execution failed",
                    metadata,
                    attachments,
                  ),
                )
              }

              yield* bus
                .publish(Metrics.ToolCall, {
                  sessionID: ctx.sessionID,
                  tool_name: key,
                  input_bytes: Metrics.jsonByteLength(args),
                  output_bytes: Metrics.jsonByteLength({
                    content: normalized.content,
                    structuredContent: normalized.structuredContent,
                  }),
                  tool_call_id: opts.toolCallId,
                  tool_call_status: "success",
                })
                .pipe(Effect.ignore)
              // 组装最终输出:output=给模型看的正文(全文或预览),metadata/attachments 取自上方 normalized 处理。
              const output = {
                title: "",
                metadata,
                output: truncated.content,
                attachments,
              }
              // 兜底:被取消时主动落库,避免僵尸态(同循环A)。
              if (opts.abortSignal?.aborted) {
                yield* input.processor.completeToolCall(opts.toolCallId, output)
              }
              return output
            }),
          )
        tools[key] = item // 把包装好的 MCP 工具塞进字典
      }

      return tools // 返回 { 工具名: AITool } 字典,交给 runLoop 发给模型
    })

    // ============================================================================
    // handleSubtask：处理一个"预定的子任务"——手动派生一个子 agent 去干活,并管理它的
    // 完整生命周期(建骨架 → 执行 → 成功/失败/取消三态 → 更新结果)。
    //
    // 何时触发:用户输入里 @了某 agent(如 "@explore 调研登录逻辑"),或斜杠命令配了 subtask,
    // 会被解析成一个 subtask part。runLoop 检测到就调本函数。
    //
    // 与"模型自己调用 task 工具"的区别:subtask 是【预先就定好】的(用户明确指定了派谁),
    // 不需要问模型"你想派谁",所以这里【手动伪造】出"模型如果调用了 task 工具本该产生的
    // 那条助手消息 + 工具调用记录",然后直接执行。
    //
    // 类比前端:很像手动 dispatch 一个异步 action,并手写它的 loading/success/error 三态 UI。
    // ============================================================================
    const handleSubtask = Effect.fn("SessionPrompt.handleSubtask")(function* (input: {
      task: MessageV2.SubtaskPart      // 子任务描述:派哪个 agent、prompt、命令等
      model: Provider.Model            // 当前模型(子任务没指定模型时用它兜底)
      lastUser: MessageV2.User         // 触发这次子任务的用户消息
      sessionID: SessionID
      session: Session.Info
      msgs: MessageV2.WithParts[]      // 当前对话消息(传给子 agent 当上下文)
    }) {
      const { task, model, lastUser, sessionID, session, msgs } = input
      const ctx = yield* InstanceState.context
      const promptOps = yield* ops()                        // 暴露给子 agent 用的 prompt/cancel 能力
      const { actor: actorTool } = yield* registry.named()  // 取出 task(actor)工具——真正派生子 agent 的执行体
      // 子任务可指定自己的模型;没指定就用主流程的 model。
      const taskModel = task.model ? yield* getModel(task.model.providerID, task.model.modelID, sessionID) : model
      // ── 块①(信封):建一条 assistant 消息,代表子 agent 这一 turn。记录 agent/model/时间等元信息。 ──
      const assistantMessage: MessageV2.Assistant = yield* sessions.updateMessage({
        id: MessageID.ascending(),
        role: "assistant",
        parentID: lastUser.id,
        sessionID,
        agentID: lastUser.agentID,
        mode: task.agent,
        agent: task.agent,
        variant: lastUser.model.variant,
        path: { cwd: ctx.directory, root: ctx.worktree },
        cost: 0,
        tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        modelID: taskModel.id,
        providerID: taskModel.providerID,
        time: { created: Date.now() },
      })
      // 组装 task 工具的调用参数——就是"模型如果调 task 工具本该传的参数"。
      const taskArgs = {
        operation: {
          action: "run" as const,
          prompt: task.prompt,
          description: task.description,
          subagent_type: task.agent,   // 派哪个子 agent
          command: task.command,
        },
      }
      // ── 块①(内容块/loading骨架):在信封里放一个 tool part,初始状态 running。
      // 先写这个骨架,TUI 订阅到就立刻显示"⏳ 正在跑子任务"(乐观更新)。用 let 因为后面要反复改它的状态。 ──
      let part: MessageV2.ToolPart = yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: assistantMessage.id,
        sessionID: assistantMessage.sessionID,
        type: "tool",
        callID: ulid(),                // 唯一且可按时间排序的调用 ID
        tool: ActorTool.id,            // 就是 task 工具
        state: {
          status: "running",
          input: taskArgs,
          time: { start: Date.now() },
        },
      })
      // ── 块②:执行前插件钩子(类比 axios 请求拦截器)。 ──
      yield* plugin.trigger(
        "tool.execute.before",
        { tool: ActorTool.id, sessionID, callID: part.id },
        { args: taskArgs },
      )

      // ── 块③:校验——要派的这个子 agent 存在吗?不存在就发错误事件(UI 弹 toast)并抛错。 ──
      const taskAgent = yield* agents.get(task.agent)
      if (!taskAgent) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${task.agent}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID, error: error.toObject() })
        throw error
      }

      // ── 块④(核心):真正派生并驱动子 agent 跑起来,并处理"成功/失败/取消"三种结局。 ──
      let error: Error | undefined                 // 失败时把错误存这儿(见下方 catchCause)
      const taskAbort = new AbortController()       // 前端老朋友:取消控制器,把"停止"信号传给子 agent
      const result = yield* actorTool
        .execute(taskArgs, {
          agent: task.agent,
          messageID: assistantMessage.id,
          sessionID,
          abort: taskAbort.signal,
          callID: part.callID,
          extra: { bypassAgentCheck: true, promptOps }, // bypassAgentCheck:用户明确指定的,跳过 agent 校验
          messages: msgs,
          // metadata:子 agent 的"进度上报通道"。它执行中不断回调此函数,把最新状态写回 part,
          // TUI 靠 part 变化实时刷新(如"explore 读到哪个文件了")。注意 part= 重新赋值,始终持有最新引用。
          metadata: (val: { title?: string; metadata?: Record<string, any> }) =>
            Effect.gen(function* () {
              part = yield* sessions.updatePart({
                ...part,
                type: "tool",
                state: { ...part.state, ...val },
              } satisfies MessageV2.ToolPart)
            }),
          // ask:子 agent 要权限时走这里,ruleset 按子 agent 自己的权限算(不是主 agent 的)。
          ask: (req: any) =>
            permission
              .ask({
                ...req,
                sessionID,
                ruleset: Agent.runtimePermission(taskAgent, session.permission),
              })
              .pipe(Effect.orDie),
        })
        .pipe(
          // 结局B——失败(≈ try/catch 的 catch):记下错误、写日志,然后 return Effect.void"吞掉"错误,
          // 不让它炸到主会话(result 会变成 undefined,后面靠"有没有 result"判断成败)。
          // 降级而非上炸:一个子任务失败不该让整个会话崩,就像单张卡片加载失败不该白屏整页。
          Effect.catchCause((cause) => {
            const defect = Cause.squash(cause)
            error = defect instanceof Error ? defect : new Error(String(defect))
            log.error("subtask execution failed", { error, agent: task.agent, description: task.description })
            return Effect.void
          }),
          // 结局C——被取消(≈ AbortController 触发时的清理):把信封标记完成、把内容块改成 error:"Cancelled",
          // 避免留下永远转圈的僵尸态。
          Effect.onInterrupt(() =>
            Effect.gen(function* () {
              taskAbort.abort()
              assistantMessage.finish = "tool-calls"
              assistantMessage.time.completed = Date.now()
              yield* sessions.updateMessage(assistantMessage)
              if (part.state.status === "running") {
                yield* sessions.updatePart({
                  ...part,
                  state: {
                    status: "error",
                    error: "Cancelled",
                    time: { start: part.state.time.start, end: Date.now() },
                    metadata: part.state.metadata,
                    input: part.state.input,
                  },
                } satisfies MessageV2.ToolPart)
              }
            }),
          ),
        )

      // 给子 agent 返回的附件补齐身份字段(id/会话/消息),挂靠到这条消息上。
      const attachments = result?.attachments?.map((attachment) => ({
        ...attachment,
        id: PartID.ascending(),
        sessionID,
        messageID: assistantMessage.id,
      }))

      // ── 块⑤:执行后插件钩子(类比 axios 响应拦截器)。 ──
      yield* plugin.trigger(
        "tool.execute.after",
        { tool: ActorTool.id, sessionID, callID: part.id, args: taskArgs },
        result,
      )

      // ── 块⑥:更新最终状态。先封"信封"——标记这条助手消息完成。 ──
      assistantMessage.finish = "tool-calls"
      assistantMessage.time.completed = Date.now()
      yield* sessions.updateMessage(assistantMessage)

      // 成功分支:把内容块从 running → completed,填入子任务的输出/标题/附件。
      if (result && part.state.status === "running") {
        yield* sessions.updatePart({
          ...part,
          state: {
            status: "completed",
            input: part.state.input,
            title: result.title,
            metadata: result.metadata,
            output: result.output,
            attachments,
            time: { ...part.state.time, end: Date.now() },
          },
        } satisfies MessageV2.ToolPart)
      }

      // 失败分支:没有 result(被 catchCause 吞掉了),把内容块改成 error 状态。
      if (!result) {
        yield* sessions.updatePart({
          ...part,
          state: {
            status: "error",
            error: error ? `Tool execution failed: ${error.message}` : "Tool execution failed",
            time: {
              start: part.state.status === "running" ? part.state.time.start : Date.now(),
              end: Date.now(),
            },
            metadata: part.state.status === "pending" ? undefined : part.state.metadata,
            input: part.state.input,
          },
        } satisfies MessageV2.ToolPart)
      }

      // ── 块⑦(收尾):只对"命令触发"的子任务生效。追加一条隐藏的合成用户消息,
      // 让主 agent 总结子任务输出并继续——这是子任务结果"回流"到主对话的方式。
      // 不是命令触发的,到此结束。 ──
      if (!task.command) return

      const summaryUserMsg: MessageV2.User = {
        id: MessageID.ascending(),
        sessionID,
        role: "user",
        agentID: lastUser.agentID,
        time: { created: Date.now() },
        agent: lastUser.agent,
        model: lastUser.model,
      }
      yield* sessions.updateMessage(summaryUserMsg)
      // synthetic:true → 界面隐藏、模型可见的"纸条"。内容:总结上面子任务的输出并继续任务。
      yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: summaryUserMsg.id,
        sessionID,
        type: "text",
        text: "Summarize the actor tool output above and continue with your task.",
        synthetic: true,
      } satisfies MessageV2.TextPart)
    })

    // ============================================================================
    // shellImpl：用户在 TUI 里直接敲 shell 命令(如 !git status)的执行体。
    // 把"用户自己跑的命令 + 输出"也记录进对话历史,做成"assistant 消息 + bash 工具调用 +
    // 结果"的记录,好让模型也能看到——相当于人和 AI 共享同一个终端。
    // 套路同 handleSubtask:命令是用户敲的,不需模型决策,所以手动伪造出"模型若调用 bash
    // 工具本该产生的记录",直接执行,再把结果填回去。
    // 流程:①准备(会话/agent/model) → ②建信封+骨架 → ③选 shell 拼参数 →
    //       ④启动子进程流式收集输出 → ⑤收尾(标记完成/中止)。
    // ============================================================================
    const shellImpl = Effect.fn("SessionPrompt.shellImpl")(function* (input: ShellInput) {
      // ── ①准备 ──
      const ctx = yield* InstanceState.context
      const run = yield* runner()
      const session = yield* sessions.get(input.sessionID)
      // 若会话处于"待回退"状态,先清理掉(开始新命令前把上一轮的回退暂存收拾干净)。
      if (session.revert) {
        yield* revert.cleanup(session)
      }
      // 校验 agent 存在;不存在就发错误事件(UI 弹 toast)并抛错。
      const agent = yield* agents.get(input.agent)
      if (!agent) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${input.agent}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }
      const inputModel = input.modelRef
        ? yield* provider
            .resolveModelRef(input.modelRef)
            .pipe(Effect.map((m) => ({ providerID: m.providerID, modelID: m.id })))
        : input.model
      const agentModel = agent.modelRef
        ? yield* provider
            .resolveModelRef(agent.modelRef)
            .pipe(Effect.map((m) => ({ providerID: m.providerID, modelID: m.id })))
        : agent.model
      // 模型多级 fallback:入参指定 → agent 指定 → 会话上次用的。
      const model = inputModel ?? agentModel ?? (yield* lastModel(input.sessionID))
      // ── ②建"信封+骨架" ──
      // 先建一条 user 消息(代表"用户发起了这次操作")。
      const userMsg: MessageV2.User = {
        id: input.messageID ?? MessageID.ascending(),
        sessionID: input.sessionID,
        time: { created: Date.now() },
        role: "user",
        agent: input.agent,
        model: { providerID: model.providerID, modelID: model.modelID },
      }
      yield* sessions.updateMessage(userMsg)
      // 一条 synthetic 文本纸条(界面隐藏、模型可见):告诉模型"下面这个工具是用户自己执行的"。
      const userPart: MessageV2.Part = {
        type: "text",
        id: PartID.ascending(),
        messageID: userMsg.id,
        sessionID: input.sessionID,
        text: "The following tool was executed by the user",
        synthetic: true,
      }
      yield* sessions.updatePart(userPart)

      // 再建一条 assistant 消息,作为承载 bash 工具调用的信封。
      const msg: MessageV2.Assistant = {
        id: MessageID.ascending(),
        sessionID: input.sessionID,
        parentID: userMsg.id,
        agentID: userMsg.agentID,
        mode: input.agent,
        agent: input.agent,
        cost: 0,
        path: { cwd: ctx.directory, root: ctx.worktree },
        time: { created: Date.now() },
        role: "assistant",
        tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        modelID: model.modelID,
        providerID: model.providerID,
      }
      yield* sessions.updateMessage(msg)
      // 建 bash 工具 part,初始 running。先写骨架,TUI 立刻能显示"⏳ 正在执行命令"(乐观更新)。
      const part: MessageV2.ToolPart = {
        type: "tool",
        id: PartID.ascending(),
        messageID: msg.id,
        sessionID: input.sessionID,
        tool: "bash",
        callID: ulid(),
        state: {
          status: "running",
          time: { start: Date.now() },
          input: { command: input.command },
        },
      }
      yield* sessions.updatePart(part)

      // ── ③选 shell + 拼调用参数 ──
      // 取用户偏好的 shell,拿到它的名字(zsh/bash/fish/...)用于下面按 shell 分派参数。
      const sh = Shell.preferred()
      const shellName = (
        process.platform === "win32" ? path.win32.basename(sh, ".exe") : path.basename(sh)
      ).toLowerCase()
      // 不同 shell 语法不同,用一张表按 shell 名分派各自的调用参数(策略模式)。
      // zsh/bash 用 login shell(-l) + source rc 文件,是为了还原你平时终端的 alias/PATH/环境变量;
      // eval + JSON.stringify(command) 把命令包成带引号的安全字符串,防特殊字符破坏 shell 语法。
      const invocations: Record<string, { args: string[] }> = {
        nu: { args: ["-c", input.command] },
        fish: { args: ["-c", input.command] },
        zsh: {
          args: [
            "-l",
            "-c",
            `
              __oc_cwd=$PWD
              [[ -f ~/.zshenv ]] && source ~/.zshenv >/dev/null 2>&1 || true
              [[ -f "\${ZDOTDIR:-$HOME}/.zshrc" ]] && source "\${ZDOTDIR:-$HOME}/.zshrc" >/dev/null 2>&1 || true
              cd "$__oc_cwd"
              eval ${JSON.stringify(input.command)}
            `,
          ],
        },
        bash: {
          args: [
            "-l",
            "-c",
            `
              __oc_cwd=$PWD
              shopt -s expand_aliases
              [[ -f ~/.bashrc ]] && source ~/.bashrc >/dev/null 2>&1 || true
              cd "$__oc_cwd"
              eval ${JSON.stringify(input.command)}
            `,
          ],
        },
        cmd: { args: ["/c", `${Shell.CMD_UTF8_PREFIX}${input.command}`] },
        powershell: {
          args: ["-NoProfile", "-Command", `${Shell.POWERSHELL_UTF8_PREFIX}${input.command}`],
        },
        pwsh: {
          args: ["-NoProfile", "-Command", `${Shell.POWERSHELL_UTF8_PREFIX}${input.command}`],
        },
        "": { args: ["-c", input.command] },
      }

      // 按 shell 名取参数;不认识的 shell 用兜底("" 那项)。
      const args = (invocations[shellName] ?? invocations[""]).args
      const cwd = ctx.directory
      // 触发插件钩子 shell.env,让插件有机会注入额外环境变量。
      const shellEnv = yield* plugin.trigger(
        "shell.env",
        { cwd, sessionID: input.sessionID, callID: part.callID },
        { env: {} },
      )

      // 组装子进程:注入环境变量;TERM=dumb 让程序不输出颜色控制符(纯文本更干净);
      // stdin=ignore 不接受输入;forceKillAfter 强杀前给 3 秒优雅退出。
      const cmd = ChildProcess.make(sh, args, {
        cwd,
        extendEnv: true,
        env: {
          ...shellEnv.env,
          ...(process.platform === "win32" ? { PYTHONIOENCODING: "utf-8" } : {}),
          TERM: "dumb",
        },
        stdin: "ignore",
        forceKillAfter: "3 seconds",
      })

      let output = ""        // 累积的命令输出
      let aborted = false    // 是否被用户中止(见下方 onInterrupt)

      // ── ⑤收尾(先定义,后面 ensuring 挂上去) ──
      // uninterruptible:收尾本身不可被中断(写库写一半又被取消会导致状态烂掉)。
      const finish = Effect.uninterruptible(
        Effect.gen(function* () {
          // 若被中止,在输出末尾追加一段"用户中止了命令"的元信息。
          if (aborted) {
            output += "\n\n" + ["<metadata>", "User aborted the command", "</metadata>"].join("\n")
          }
          // 标记 assistant 消息完成(若还没标)。
          if (!msg.time.completed) {
            msg.time.completed = Date.now()
            yield* sessions.updateMessage(msg)
          }
          // 把 bash 工具 part 从 running 定稿为 completed,写入最终输出。
          if (part.state.status === "running") {
            part.state = {
              status: "completed",
              time: { ...part.state.time, end: Date.now() },
              input: part.state.input,
              title: "",
              metadata: { output, description: "" },
              output,
            }
            yield* sessions.updatePart(part)
          }
        }),
      )

      // ── ④启动子进程,流式收集输出 ──
      const exit = yield* Effect.gen(function* () {
        const handle = yield* spawner.spawn(cmd)
        // 子进程的 stdout/stderr 是流(handle.all):每来一块 chunk 就累积,并"非阻塞地"(run.fork)
        // 刷新工具 part,使 TUI 像真实终端一样一行行往外冒(类比前端消费 SSE/ReadableStream)。
        yield* Stream.runForEach(Stream.decodeText(handle.all), (chunk) =>
          Effect.sync(() => {
            output += chunk
            if (part.state.status === "running") {
              part.state.metadata = { output, description: "" }
              void run.fork(sessions.updatePart(part))
            }
          }),
        )
        yield* handle.exitCode  // 等进程退出码
      }).pipe(
        Effect.scoped,
        // 被取消时只做一件轻量事:置 aborted=true。真正的收尾交给下面 ensuring 的 finish。
        Effect.onInterrupt(() =>
          Effect.sync(() => {
            aborted = true
          }),
        ),
        Effect.orDie,
        Effect.ensuring(finish),  // ≈ try/finally 的 finally:无论成功/失败/取消,都跑 finish
        Effect.exit,              // 把结果包成 Exit,不直接抛,便于下面判断
      )

      // 只有"非取消"的真实失败才往上抛;纯取消(用户主动 Esc)视为正常收尾,不算错误。
      if (Exit.isFailure(exit) && !Cause.hasInterruptsOnly(exit.cause)) {
        return yield* Effect.failCause(exit.cause)
      }

      return { info: msg, parts: [part] }
    })

    // ============================================================================
    // getModel：provider.getModel 的封装,核心价值是"模型找不到时给用户一条带建议的友好提示"。
    // 它只是顺便发提示,不吞错误——该失败仍失败(最后原样重抛)。
    // ============================================================================
    const getModel = Effect.fn("SessionPrompt.getModel")(function* (
      providerID: ProviderID,
      modelID: ModelID,
      sessionID: SessionID,
    ) {
      // Effect.exit:把"可能失败的 Effect"捕获成一个 Exit 值(Success/Failure),不直接抛。
      // 类比前端把 await fetch() 包进 try/catch,好让下面自己判断成败再决定怎么处理。
      const exit = yield* provider.getModel(providerID, modelID).pipe(Effect.exit)
      if (Exit.isSuccess(exit)) return exit.value // 成功 → 直接返回模型
      // 失败 → Cause.squash 把 Effect 复杂的错误原因"压扁"成一个普通 Error。
      const err = Cause.squash(exit.cause)
      // 只对"模型没找到"这类特定错误做友好处理。
      if (Provider.ModelNotFoundError.isInstance(err)) {
        // 若 provider 给了候选,拼一句 "Did you mean: xxx?"(比如敲错 gpt-4o → 提示 gpt-4o-mini)。
        const hint = err.data.suggestions?.length ? ` Did you mean: ${err.data.suggestions.join(", ")}?` : ""
        // 发一个用户可见的错误事件(UI 弹 toast)。
        yield* bus.publish(Session.Event.Error, {
          sessionID,
          error: new NamedError.Unknown({
            message: `Model not found: ${err.data.providerID}/${err.data.modelID}.${hint}`,
          }).toObject(),
        })
      }
      // 无论如何把原始错误重新抛出去——只负责"顺便提示",不负责"消化错误"。
      return yield* Effect.failCause(exit.cause)
    })

    // ============================================================================
    // lastModel：回答"这个会话上次用的是哪个模型",作为模型解析链的兜底。
    // 用途:入参没指定、agent 也没指定模型时,延续会话上次用的模型,保持一致性。
    // fnUntraced:不生成链路追踪 span——这是个高频廉价小查询,不值得单独记一条 trace。
    // ============================================================================
    const lastModel = Effect.fnUntraced(function* (sessionID: SessionID) {
      // 往回找最近一条"角色是 user 且带了 model 字段"的消息。agentID:"*" 表示跨所有 agent 切片找。
      const match = yield* sessions.findMessage(
        sessionID,
        (m) => m.info.role === "user" && !!m.info.model,
        { agentID: "*" },
      )
      // Option 是 Effect 版的"可空值",isSome=有值。找到就用它的 model。
      if (Option.isSome(match) && match.value.info.role === "user") return match.value.info.model
      // 没找到(如全新会话)→ 回退到系统默认模型。
      return yield* provider.defaultModel()
    })

    // ============================================================================
    // createUserMessage：把用户输入的一堆"轻量引用"(文本、@文件、@agent、data URL、
    // MCP 资源等)【兑现】成一条完整的、能存库、能喂给模型的用户消息。
    //
    // 与 resolvePromptParts 的分工:resolvePromptParts 只"认出"每个 @xxx 是文件还是 agent;
    // createUserMessage 才真正去【读文件、抓 MCP 资源、解码 data URL】,把引用变成实际内容。
    // 类比前端:很像 GraphQL 的 resolver / 数据水合——把引用描述解析成真实数据。
    //
    // 流程:①解析 agent → ②解析 model+variant → ③构建消息头 info →
    //       ④resolvePart 逐个兑现 part(核心) → ⑤并发跑+拍平+补 id → ⑥插件钩子 →
    //       ⑦校验(只记日志不抛) → ⑧存库。
    // ============================================================================
    const createUserMessage = Effect.fn("SessionPrompt.createUserMessage")(function* (input: PromptInput) {
      // ── ①解析 agent:入参指定 → 否则用默认 agent。 ──
      // 获取agent name
      const agentName = input.agent || (yield* agents.defaultAgent())
      const ag = yield* agents.get(agentName)
      if (!ag) {
        // 找不到agent会列出可用的agent
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${agentName}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }

      // ── ②解析 model + variant ──
      // 输入方模型(调用方这次显式指定的;modelRef 是档位/分组名,需 resolveModelRef 解析成具体模型)
      const inputModel = input.modelRef
        ? yield* provider
            .resolveModelRef(input.modelRef)
            .pipe(Effect.map((m) => ({ providerID: m.providerID, modelID: m.id })))
        : input.model
      // agent配置模型(这个 agent 默认用的模型)
      const agentModel = ag.modelRef
        ? yield* provider
            .resolveModelRef(ag.modelRef)
            .pipe(Effect.map((m) => ({ providerID: m.providerID, modelID: m.id })))
        : ag.model
      // 三级优先级:调用方指定 ?? agent 配置 ?? 会话上次用的。
      const model = inputModel ?? agentModel ?? (yield* lastModel(input.sessionID))
      // same:最终选中的模型是不是就等于 agent 配置的那个模型。variant(模型变体,如思考模式)
      // 是绑在 agent 自己的模型上的——只有真用了 agent 的模型(same)时才考虑套用它,
      // 若调用方覆盖了模型就不套(换了模型那 variant 无意义)。
      const same = agentModel && model.providerID === agentModel.providerID && model.modelID === agentModel.modelID
      // full:取完整模型信息以查它支持哪些 variant(仅当需要判断 agent 的 variant 时才取)。
      const full =
        !input.variant && ag.variant && same
          ? yield* provider.getModel(model.providerID, model.modelID).pipe(Effect.catchDefect(() => Effect.void))
          : undefined
      // 最终 variant:调用方显式指定优先;否则当 agent 配了 variant 且模型确实支持时才用。
      const variant = input.variant ?? (ag.variant && full?.variants?.[ag.variant] ? ag.variant : undefined)

      // ── ③构建消息头 info(信封:role/agent/model/时间等元信息,不含具体内容)。 ──
      const info: MessageV2.User = {
        id: input.messageID ?? MessageID.ascending(),
        role: "user",
        sessionID: input.sessionID,
        agentID: input.agentID,
        time: { created: Date.now() },
        tools: input.tools,
        agent: ag.name,
        model: {
          providerID: model.providerID,
          modelID: model.modelID,
          variant,
        },
        system: input.system,
        format: input.format,
        provenance: input.provenance,
      }

      // 作用域结束时清理这条消息相关的临时指令(配合函数末尾的 Effect.scoped)。
      yield* Effect.addFinalizer(() => instruction.clear(info.id))

      // Draft<T>:"id 可选"版的 Part 类型(草稿阶段先不填 id)。分配式条件类型逐个成员处理,
      // 保住判别联合的 type 与各自字段的对应关系(直接 Omit 整个联合会破坏它)。
      type Draft<T> = T extends MessageV2.Part ? Omit<T, "id"> & { id?: string } : never
      // assign:给草稿 part 补 id——有就用(PartID.make),没有就生成递增 id(ascending 保证顺序)。
      const assign = (part: Draft<MessageV2.Part>): MessageV2.Part => ({
        ...part,
        id: part.id ? PartID.make(part.id) : PartID.ascending(),
      })

      // ── ④resolvePart(核心):把"一个输入 part"兑现成"一个或多个真实 part"。
      // 按 part 类型分支:file(MCP资源 / data URL / 本地文件·目录·二进制)、agent、默认透传。 ──
      const resolvePart: (part: PromptInput["parts"][number]) => Effect.Effect<Draft<MessageV2.Part>[]> = Effect.fn(
        "SessionPrompt.resolveUserPart",
      )(function* (part) {
        if (part.type === "file") {
          // 分支A:file 来源是 MCP 资源 → 调 mcp.readResource 抓下来,展开成文本片段/二进制占位。
          if (part.source?.type === "resource") {
            const { clientName, uri } = part.source
            log.info("mcp resource", { clientName, uri, mime: part.mime })
            const pieces: Draft<MessageV2.Part>[] = [
              {
                messageID: info.id,
                sessionID: input.sessionID,
                type: "text",
                synthetic: true,
                text: `Reading MCP resource: ${part.filename} (${uri})`,
              },
            ]
            const exit = yield* mcp.readResource(clientName, uri).pipe(Effect.exit)
            if (Exit.isSuccess(exit)) {
              const content = exit.value
              if (!content) throw new Error(`Resource not found: ${clientName}/${uri}`)
              const items = Array.isArray(content.contents) ? content.contents : [content.contents]
              for (const c of items) {
                if ("text" in c && c.text) {
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: c.text,
                  })
                } else if ("blob" in c && c.blob) {
                  const mime = "mimeType" in c ? c.mimeType : part.mime
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `[Binary content: ${mime}]`,
                  })
                }
              }
              pieces.push({ ...part, messageID: info.id, sessionID: input.sessionID })
            } else {
              const error = Cause.squash(exit.cause)
              log.error("failed to read MCP resource", { error, clientName, uri })
              const message = error instanceof Error ? error.message : String(error)
              pieces.push({
                messageID: info.id,
                sessionID: input.sessionID,
                type: "text",
                synthetic: true,
                text: `Failed to read MCP resource ${part.filename}: ${message}`,
              })
            }
            return pieces
          }
          const url = new URL(part.url)
          switch (url.protocol) {
            // 分支B:data URL 内联数据。文本类型 → 解码出内容,并伪造成"调用了 Read 工具 + 返回内容"
            // 的样子(下面 file 分支同理),让模型以为文件已读好,省一次 read 往返。
            case "data:":
              if (part.mime === "text/plain") {
                return [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify({ file_path: part.filename })}`,
                  },
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: decodeDataUrl(part.url),
                  },
                  { ...part, messageID: info.id, sessionID: input.sessionID },
                ]
              }
              break
            // 分支C:本地文件(@文件)。真正调 read 工具读它,并伪造成"read 工具调用+结果",
            // 让模型直接拿到内容,省一次往返;复用真正的 read 工具保证与模型自己读时行为一致。
            case "file:": {
              log.info("file", { mime: part.mime })
              const filepath = fileURLToPath(part.url)
              if (yield* fsys.isDir(filepath)) part.mime = "application/x-directory"

              // execRead:用真正的 read 工具读文件的小封装(bypassCwdCheck 放宽目录限制,因用户明确指定)。
              const { read } = yield* registry.named()
              const execRead = (args: Parameters<typeof read.execute>[0], extra?: Tool.Context["extra"]) => {
                const controller = new AbortController()
                return read
                  .execute(args, {
                    sessionID: input.sessionID,
                    abort: controller.signal,
                    agent: input.agent!,
                    messageID: info.id,
                    extra: { bypassCwdCheck: true, ...extra },
                    messages: [],
                    metadata: () => Effect.void,
                    ask: () => Effect.void,
                  })
                  .pipe(Effect.onInterrupt(() => Effect.sync(() => controller.abort())))
              }

              // C-1:文本文件。支持行范围(URL 的 ?start=&end= 参数)。
              if (part.mime === "text/plain") {
                let offset: number | undefined
                let limit: number | undefined
                const range = { start: url.searchParams.get("start"), end: url.searchParams.get("end") }
                if (range.start != null) {
                  const filePathURI = part.url.split("?")[0]
                  let start = parseInt(range.start)
                  let end = range.end ? parseInt(range.end) : undefined
                  // LSP 加持:若 start===end(只点了一行),查这行是不是某符号(函数/类)的定义,
                  // 是的话自动扩展到整个符号范围(@file 点在函数名上 → 读整个函数)。
                  if (start === end) {
                    const symbols = yield* lsp.documentSymbol(filePathURI).pipe(Effect.catch(() => Effect.succeed([])))
                    for (const symbol of symbols) {
                      let r: LSP.Range | undefined
                      if ("range" in symbol) r = symbol.range
                      else if ("location" in symbol) r = symbol.location.range
                      if (r?.start?.line && r?.start?.line === start) {
                        start = r.start.line
                        end = r?.end?.line ?? start
                        break
                      }
                    }
                  }
                  offset = Math.max(start, 1)
                  if (end) limit = end - (offset - 1)
                }
                const args = { file_path: filepath, offset, limit }
                const pieces: Draft<MessageV2.Part>[] = [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify(args)}`,
                  },
                ]
                const exit = yield* provider.getModel(info.model.providerID, info.model.modelID).pipe(
                  Effect.flatMap((mdl) => execRead(args, { model: mdl })),
                  Effect.exit,
                )
                if (Exit.isSuccess(exit)) {
                  const result = exit.value
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: result.output,
                  })
                  if (result.attachments?.length) {
                    pieces.push(
                      ...result.attachments.map((a) => ({
                        ...a,
                        synthetic: true,
                        filename: a.filename ?? part.filename,
                        messageID: info.id,
                        sessionID: input.sessionID,
                      })),
                    )
                  } else {
                    pieces.push({ ...part, messageID: info.id, sessionID: input.sessionID })
                  }
                } else {
                  const error = Cause.squash(exit.cause)
                  log.error("failed to read file", { error })
                  const message = error instanceof Error ? error.message : String(error)
                  yield* bus.publish(Session.Event.Error, {
                    sessionID: input.sessionID,
                    error: new NamedError.Unknown({ message }).toObject(),
                  })
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Read tool failed to read ${filepath} with the following error: ${message}`,
                  })
                }
                return pieces
              }

              // C-2:目录。read 工具读出目录列表。
              if (part.mime === "application/x-directory") {
                const args = { file_path: filepath }
                const exit = yield* execRead(args).pipe(Effect.exit)
                if (Exit.isFailure(exit)) {
                  const error = Cause.squash(exit.cause)
                  log.error("failed to read directory", { error })
                  const message = error instanceof Error ? error.message : String(error)
                  yield* bus.publish(Session.Event.Error, {
                    sessionID: input.sessionID,
                    error: new NamedError.Unknown({ message }).toObject(),
                  })
                  return [
                    {
                      messageID: info.id,
                      sessionID: input.sessionID,
                      type: "text",
                      synthetic: true,
                      text: `Read tool failed to read ${filepath} with the following error: ${message}`,
                    },
                  ]
                }
                return [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify(args)}`,
                  },
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: exit.value.output,
                  },
                  { ...part, messageID: info.id, sessionID: input.sessionID },
                ]
              }

              // C-3:其它(二进制,如图片)。读文件 → base64 编码成 data URL 的 file part
              // (模型上下文吃不下原始字节,只能走编码后的 data URL,见二进制/base64 那节)。
              return [
                {
                  messageID: info.id,
                  sessionID: input.sessionID,
                  type: "text",
                  synthetic: true,
                  text: `Called the Read tool with the following input: {"file_path":"${filepath}"}`,
                },
                {
                  id: part.id,
                  messageID: info.id,
                  sessionID: input.sessionID,
                  type: "file",
                  url:
                    `data:${part.mime};base64,` +
                    Buffer.from(yield* fsys.readFile(filepath).pipe(Effect.catch(Effect.die))).toString("base64"),
                  mime: part.mime,
                  filename: part.filename!,
                  source: part.source,
                },
              ]
            }
          }
        }

        // 分支D:@agent。展开成"agent 引用本身 + 一条合成指令(叫模型去调 actor 工具派这个子 agent)"。
        if (part.type === "agent") {
          // 若权限上该 agent 是 deny 的,但用户明确 @了它 → 加一句"用户指定的,保证存在"以绕过(用户优先级最高)。
          const perm = Permission.evaluate("task", part.name, ag.permission)
          const hint = perm.action === "deny" ? " . Invoked by user; guaranteed to exist." : ""
          return [
            { ...part, messageID: info.id, sessionID: input.sessionID },
            {
              messageID: info.id,
              sessionID: input.sessionID,
              type: "text",
              synthetic: true,
              text:
                " Use the above message and context to generate a prompt and call the actor tool with subagent: " +
                part.name +
                hint,
            },
          ]
        }

        // 分支E(默认):纯文本等,原样透传,不加工。
        return [{ ...part, messageID: info.id, sessionID: input.sessionID }]
      })

      // ── ⑤对所有输入 part 并发跑 resolvePart(多个文件同时读)→ 拍平(一个 part 可能兑现成多个)→
      // 逐个 assign 补 id。 ──
      const parts = yield* Effect.forEach(input.parts, resolvePart, { concurrency: "unbounded" }).pipe(
        Effect.map((x) => x.flat().map(assign)),
      )

      // ── ⑥插件钩子 chat.message:让插件在消息落库前介入。 ──
      yield* plugin.trigger(
        "chat.message",
        {
          sessionID: input.sessionID,
          agent: input.agent,
          model: input.model,
          messageID: input.messageID,
          variant: input.variant,
        },
        { message: info, parts },
      )

      // ── ⑦校验:safeParse 校验消息头和每个 part。注意——失败只 log.error 记日志、【不抛错】,
      // 防御性:即便某个 part 格式有问题,也尽量把能存的存进去,不让整条消息崩掉。 ──
      const parsed = MessageV2.Info.safeParse(info)
      if (!parsed.success) {
        log.error("invalid user message before save", {
          sessionID: input.sessionID,
          messageID: info.id,
          agent: info.agent,
          model: info.model,
          issues: parsed.error.issues,
        })
      }
      parts.forEach((part, index) => {
        const p = MessageV2.Part.safeParse(part)
        if (p.success) return
        log.error("invalid user part before save", {
          sessionID: input.sessionID,
          messageID: info.id,
          partID: part.id,
          partType: part.type,
          index,
          issues: p.error.issues,
          part,
        })
      })

      // ── ⑧存库:先存信封(消息头),再逐个存 part。 ──
      yield* sessions.updateMessage(info)
      for (const part of parts) yield* sessions.updatePart(part)

      return { info, parts }
      // Effect.scoped:让上面 addFinalizer 注册的清理(instruction.clear)在此作用域结束时执行。
    }, Effect.scoped)

    // ============================================================================
    // sweepOrphanAssistants：清理"孤儿助手消息"——因硬中断(进程崩溃/被 kill/断连)导致
    // 正常 finish 收尾没跑、永远停在"进行中"(有 created 无 completed)的残留消息。
    //
    // 危害:TUI 靠"最新的未完成助手消息"判断会话是否在忙,陈旧孤儿会让新发的每条消息
    // 假性卡成 QUEUED(最长 1 小时)。所以这里给孤儿强制收尾:补 completed 时间 + 标记 AbortedError。
    //
    // 核心两难:"未完成" ≠ "孤儿"——它也可能是正在重试的在途请求。双策略区分(见 immediate):
    //   · 不确定会话忙不忙 → 用【时间守卫】(等够 ORPHAN_AGE_MS 才敢清)
    //   · 确定会话空闲(idle) → 用【状态判断】(空闲还挂着未完成的必是孤儿,立即清)
    //
    // 类比前端:进页面时重置"僵尸 loading"(请求中途崩了,finally 没跑,loading 永远 true)。
    // ============================================================================
    const sweepOrphanAssistants = Effect.fn("SessionPrompt.sweepOrphanAssistants")(function* (
      sessionID: SessionID,
      // immediate=true 时,不看年龄,直接清所有未完成助手消息。调用方在"会话空闲(无活跃 runner)"时
      // 传 true——此时任何未完成的助手消息都必是硬中断留下的真孤儿,而非在途重试。立即清很重要,
      // 否则新消息会被那个假性 QUEUED 挡住(最长 ORPHAN_AGE_MS=1 小时)。默认 false,让后台调用方
      // (spawn/hook)保留年龄守卫。
      // When true, sweep dangling assistants regardless of age. The caller sets
      // this when the session is idle (no active runner), meaning any assistant
      // without time.completed is definitively orphaned — left behind by a hard
      // interruption (process crash / kill / disconnect) that skipped the normal
      // `finish` effect, not an in-flight retry chain. Sweeping immediately
      // matters because the TUI derives its "pending" marker from the newest
      // incomplete assistant (routes/session/index.tsx `pending`): a stale
      // orphan otherwise makes EVERY newly submitted message on an idle session
      // render as stuck QUEUED for up to ORPHAN_AGE_MS (an hour). Defaults to
      // false so background callers (spawn/hook) keep the age guard.
      immediate = false,
    ) {
      const msgs = yield* sessions.messages({ sessionID, agentID: "*" }) // 取会话所有消息(跨所有 agent 切片,孤儿可能属于子 agent)
      const now = Date.now()
      // 1 小时——必须超过 Task 1 的 chunkMs（300s）加上 Task 2 的
      // PERSISTENT_RETRY 最坏情况退避（10 次尝试 × 5 分钟上限 =
      // 50 分钟），这样一个仍然活跃的在途请求就绝不会在其重试链
      // 正取得进展时被错误清扫。
      const ORPHAN_AGE_MS = 3_600_000
      for (const m of msgs) {
        if (m.info.role !== "assistant") continue          // 只处理助手消息
        if (m.info.time?.completed) continue                // 已完成的不是孤儿,跳过
        const created = m.info.time?.created ?? 0
        // 年龄守卫:非 immediate 模式下,只清"超过 1 小时还没完成的",避免误杀在途重试请求。
        if (!immediate && now - created < ORPHAN_AGE_MS) continue
        // 强制收尾:补完成时间 + 标记中止错误(本来就有错就保留),让界面显示"被中断"而非无声无息。
        m.info.time = { ...m.info.time, completed: now }
        m.info.error =
          m.info.error ??
          new MessageV2.AbortedError({
            message: "Abandoned: previous request interrupted before completion",
          }).toObject()
        // 存库。失败只记 warn 不抛错——清理是"尽力而为"的维护动作,单条失败不该拖垮流程(防御性)。
        yield* sessions.updateMessage(m.info).pipe(
          Effect.catchCause((cause) =>
            elog.warn("orphan-update-failed", {
              sessionID,
              messageID: m.info.id,
              cause,
            }),
          ),
        )
        yield* elog.info("orphan-assistant-cleared", {
          sessionID,
          messageID: m.info.id,
        })
      }
    })

    // ============================================================================
    // prompt：SessionPrompt 对外的【主入口】——用户发一条消息就从这进。
    // 职责:清理 → 兑现用户消息落库 → 启动 agent 循环 → 返回最终助手消息。
    // 调用链枢纽:prompt → loop → runLoop(while 循环) → resolveTools/handleSubtask...
    // 把前面学的所有函数串成一条"用户消息 → AI 回复"的完整链路。
    // ============================================================================
    const prompt: (input: PromptInput) => Effect.Effect<MessageV2.WithParts> = Effect.fn("SessionPrompt.prompt")(
      function* (input: PromptInput) {
        const session = yield* sessions.get(input.sessionID)
        // 前置工作:只对"真人直接发的消息"做清理;排除子 agent 派生(spawn)和定时/钩子(hook)这类系统内部发起的。
        if (input.source !== "spawn" && input.source !== "hook") {
          yield* revert.cleanup(session) // 清理上一轮的回退暂存
          // 会话空闲=无活跃 runner,此时任何未完成的助手消息都是硬中断留下的真孤儿 → 立即清
          //(把 idle 作为 immediate 传入),免得新消息被那个假性 QUEUED 挡住。
          // An idle session has no active runner, so any dangling assistant is a
          // true orphan from a hard interruption — sweep it now (age-independent)
          // so a fresh message is not rendered as stuck QUEUED behind it.
          const idle = (yield* status.get(input.sessionID)).type === "idle"
          yield* sweepOrphanAssistants(input.sessionID, idle)
        }
        // 把输入 parts 兑现成完整用户消息并落库(见 createUserMessage)。
        const message = yield* createUserMessage(input)
        yield* sessions.touch(input.sessionID) // 更新会话"最后活动时间",供列表排序

        // 向后兼容:input.tools 是老的 {工具名:是否启用} 开关,翻译成权限规则集写到会话上(已 @deprecated)。
        const permissions: Permission.Ruleset = []
        for (const [t, enabled] of Object.entries(input.tools ?? {})) {
          permissions.push({ permission: t, action: enabled ? "allow" : "deny", pattern: "*" })
        }
        if (permissions.length > 0) {
          session.permission = permissions
          yield* sessions.setPermission({ sessionID: session.id, permission: permissions })
        }

        // noReply:只存用户消息、不触发 AI 回复(如 /goal clear 那种只想塞条消息的场景)。
        if (input.noReply === true) return message
        // 否则启动 agent 主循环(默认 agentID="main"),这才是"让 AI 干活"。返回最终助手消息。
        return yield* loop({ sessionID: input.sessionID, agentID: input.agentID ?? "main", task_id: input.task_id })
      },
    )

    const lastAssistant = Effect.fnUntraced(function* (sessionID: SessionID, agentID?: string) {
      if (agentID !== undefined) {
        // 按 agent 作用域：返回*该* agent 最新的消息（优先助手消息）。
        // 对同一会话内并发的子 agent 至关重要——会话级的查找会把并发 actor 的
        // 返回值坍缩成"最后完成的那个"。messages() 按从旧到新（末尾最新）返回，
        // 所以 findLast 取到最新的助手消息，最后一个元素则是整体最新的消息。
        const own = yield* sessions.messages({ sessionID, agentID })
        const lastAsst = own.findLast((m) => m.info.role === "assistant")
        if (lastAsst) return lastAsst
        if (own.length > 0) return own[own.length - 1]
        // 如果该 agent 还没有任何消息，则回退到会话级查找
      }
      const match = yield* sessions.findMessage(sessionID, (m) => m.info.role !== "user", { agentID: "*" })
      if (Option.isSome(match)) return match.value
      const msgs = yield* sessions.messages({ sessionID, limit: 1, agentID: "*" })
      if (msgs.length > 0) return msgs[0]
      throw new Error("Impossible")
    })

    const runLoop: (
      sessionID: SessionID,
      agentID?: string,
      task_id?: string,
      notifyParentOnComplete?: boolean,
    ) => Effect.Effect<MessageV2.WithParts> = Effect.fn("SessionPrompt.run")(
      function* (sessionID: SessionID, agentID?: string, task_id?: string, notifyParentOnComplete?: boolean) {
        const ctx = yield* InstanceState.context
        const slog = elog.with({ sessionID })
        let structured: unknown | undefined
        let step = 0
        const session = yield* sessions.get(sessionID)
        let lastFinishedForPrune: MessageV2.Assistant | undefined
        let lastModelForPrune: Provider.Model | undefined
        let outputLengthContinuations = 0
        // "模型完成了但没产出任何可用内容"（仅推理 / 空）的共享本地计数器。
        // T04 的通用 invalid 重试复用这同一个计数器——不要再加第二个。局部于 runLoop，
        // 这样新一轮用户 turn 会重置它（无跨消息污染），与 outputLengthContinuations 一致。
        let invalidContinuations = 0
        // structured-output 专用 retry：上限来自 lastUser.format.retryCount（默认 2），
        // 与 invalidContinuations（generic invalid）分离，互不污染。局部于 runLoop，
        // 新一轮用户 turn 自动归零。
        let structuredRetries = 0
        // 针对文本形式工具调用（模型把工具调用写成了散文文本而非结构化的 tool_use）的
        // 有限次重试。局部于 runLoop，这样每一轮新的用户 turn 都从干净状态开始。
        let textToolCallRetries = 0
        // Consecutive empty/no-op tool-call steps in this turn. Counts steps
        // where the model "called a tool" with empty/invalid input, or produced
        // no valid tool part and no substantive output at all (see isEmptyStep).
        // A single non-empty step resets it. Escalates soft (remind → replan)
        // then hard-halts once it exceeds EMPTY_STEP_MAX_RECOVERY, mirroring the
        // text-ngram ladder. Local to runLoop so a fresh user turn starts clean.
        let emptyStepStreak = 0
        // Set true when a guard hard-halts the turn (currently the empty-step
        // guard). A hard halt is terminal: it must break out immediately and
        // NOT be re-entered by the taskGate / goalGate ReAct gates, which would
        // otherwise inject a fresh user turn and re-drive a still-degraded model
        // into the same loop.
        let hardHalt = false
        const resolvedAgentID = agentID ?? "main"
        // 跟踪由插件驱动的取消（session.pre 或任意 session.userQuery.pre），
        // 使 session.post 报告 outcome="cancelled" 而非 "error"。
        let cancelled = false
        let cancelReason: string | undefined

        // 通过下面主体上的 Effect.onExit 恰好触发一次 session.post。
        // 没有这层包裹，while 循环内任何被 yield 出来的失败（provider 错误、
        // 网络错误、抛出的 defect）都会完全跳过这个钩子。
        //
        // 轨迹一致性：使用 MessageV2.filterCompactedEffect 配合会话的 contextFrom /
        // contextWatermark，使 compaction 边界把历史裁剪到 agent 实际看到的范围，
        // 并且包含子会话的父级前缀——与 session.userQuery.post 语义一致。
        const firePostSession = (exit: Exit.Exit<MessageV2.WithParts, unknown>) =>
          Effect.gen(function* () {
            const sliceMsgs = yield* MessageV2.filterCompactedEffect(sessionID, {
              contextFrom: session.contextFrom,
              contextWatermark: session.contextWatermark,
              agentID: resolvedAgentID,
            }).pipe(Effect.catch(() => Effect.succeed([] as MessageV2.WithParts[])))
            const lastSlice = sliceMsgs.findLast((m) => m.info.role === "assistant")
            const finalAsst =
              lastSlice && lastSlice.info.role === "assistant" ? lastSlice.info : undefined
            const finalParts = lastSlice?.parts ?? []
            const failed = Exit.isFailure(exit)
            const finalIsError = !!finalAsst?.error
            const outcome: "completed" | "error" | "cancelled" = cancelled
              ? "cancelled"
              : failed || finalIsError
                ? "error"
                : "completed"
            const error = cancelled
              ? cancelReason
              : failed
                ? Cause.pretty(exit.cause)
                : finalAsst
                  ? sessionErrorText(finalAsst.error)
                  : undefined
            yield* plugin.trigger(
              "session.post",
              {
                sessionID,
                agentID: resolvedAgentID,
                task_id,
                outcome,
                error,
                finalText: finalAsst ? assistantFinalText(finalAsst, finalParts) : undefined,
                assistantMessageID: finalAsst?.id,
                trajectory: serializeTrajectoryMessages(sliceMsgs),
              },
              {},
            )
          }).pipe(Effect.ignore)

        return yield* Effect.gen(function* () {
          const preSession = { cancel: undefined as boolean | undefined, cancelReason: undefined as string | undefined }
          yield* plugin.trigger(
            "session.pre",
            { sessionID, agentID: resolvedAgentID, task_id },
            preSession,
          )
          if (preSession.cancel) {
            cancelled = true
            cancelReason = preSession.cancelReason
            return yield* Effect.fail(
              new NamedError.Unknown({
                message: preSession.cancelReason ?? "Session cancelled by plugin",
              }),
            )
          }
        const agentMetrics = { tokens_in: 0, tokens_out: 0, files_changed: 0 }
        const trajectoryForStep = (currentMsgs: MessageV2.WithParts[], assistant: MessageV2.Assistant) =>
          serializeTrajectoryMessages(
            withAssistantParts(currentMsgs, assistant, MessageV2.parts(assistant.id)),
          )

        const publishAgentRequest = (phase: string, taskType: string) =>
          bus
            .publish(Metrics.AgentRequest, {
              sessionID,
              phase,
              task_type: taskType,
              surface: Flag.MIMOCODE_CLIENT,
              total_tokens_in: agentMetrics.tokens_in,
              total_tokens_out: agentMetrics.tokens_out,
              files_changed: agentMetrics.files_changed,
              validation_status: "skipped",
            })
            .pipe(Effect.ignore)
        // 裁剪释放了空间，但 `lastFinished.tokens` 仍反映裁剪前的状态。
        // 跳过一次溢出检查，让模型能在裁剪后的上下文上作答；
        // 它新的助手消息会为下一次检查携带准确的 token 数。
        let skipOverflowCheck = false

        const textLoopBuffer: string[] = []
        let textLoopRecoveryAttempts = 0
        let textNgramRecoveryAttempts = 0

        // 契约（T05）：当 finish="length" 时，仅对纯文本注入续写提示。如果存在任何
        // 非 providerExecuted 的客户端工具 part，就放弃（返回 false），交由 classify
        // 走正常的"工具观测再循环"。这保证了"涉及工具时不做 output-length 续写"——
        // 但它*不*保证一个在流式过程中被截断的工具从未执行过，因为 AI SDK 会在得知
        // finish 原因之前就在流中途运行工具。
        const autoContinueOutputLength = Effect.fn("SessionPrompt.autoContinueOutputLength")(function* (input: {
          lastUser: MessageV2.User
          assistant: MessageV2.Assistant
        }) {
          if (input.assistant.finish !== "length" || input.assistant.error || input.assistant.summary) return false
          if (
            MessageV2.parts(input.assistant.id).some((part) => part.type === "tool" && !part.metadata?.providerExecuted)
          ) {
            return false
          }
          if (outputLengthContinuations >= OUTPUT_LENGTH_CONTINUATION_LIMIT) {
            input.assistant.error = new MessageV2.OutputLengthError({}).toObject()
            yield* sessions.updateMessage(input.assistant)
            yield* bus.publish(Session.Event.Error, {
              sessionID: input.assistant.sessionID,
              error: input.assistant.error,
            })
            return false
          }

          outputLengthContinuations++
          yield* slog.info("auto-continuing output length", { attempt: outputLengthContinuations })
          const msg = yield* sessions.updateMessage({
            id: MessageID.ascending(),
            role: "user" as const,
            sessionID: input.lastUser.sessionID,
            agentID: input.lastUser.agentID,
            agent: input.lastUser.agent,
            model: input.lastUser.model,
            tools: input.lastUser.tools,
            format: input.lastUser.format,
            time: { created: Date.now() },
          })
          yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: msg.id,
            sessionID: msg.sessionID,
            type: "text",
            synthetic: true,
            text: [
              "<system-reminder>",
              "The previous assistant response hit the model output token limit before completing.",
              "Continue the same task from the exact point where it stopped.",
              "Do not restart, recap, or repeat prior reasoning. Keep reasoning concise, prefer concrete tool calls or final output, and only stop when the user's task is complete or genuinely blocked.",
              "</system-reminder>",
            ].join("\n"),
          } satisfies MessageV2.TextPart)
          return true
        })

        // 任务停止条件闸门（仅主 agent）。在允许停止之前，列出会话里未终结的任务：
        // 如果还有剩余，就以合成用户 turn 的形式注入一条提示并重入（返回 true），
        // 让模型用 `task done` / `task abandon` 把它们收尾。ReAct 上限 + 计数器
        // 与 goal 闸门相同；超过上限则允许停止并记一条 warn 日志（主 agent 上无
        // reportedStatus）。owner=undefined 会接手那些被达到自身上限的子 agent 闸门
        // 遗弃的任务。在 goalGate *之前*运行，因为任务状态更容易结算，而一块待办任务
        // 板会污染任何 goal 裁决。
        const taskGate = Effect.fn("SessionPrompt.taskGate")(function* (lastUser: MessageV2.User) {
          if ((agentID ?? "main") !== "main") return false
          // 如果主 agent 的 `task` 工具被剥离了（Permission.disabled），那么提示它去
          // 调用 `task done` 是无法满足的，并且会一直重入直到达到上限。此时完全跳过
          // 闸门。这与 actor/spawn.ts 里的 canWrite 跳过（对 forkAgentInfo 做
          // Permission.disabled(["write"], ...) 检查）相对应。按会话解析意味着这里
          // 只检查 agent 的静态权限（对 v1 足够；会话级覆盖在被拒绝的 agent 上重新
          // 启用 task 属于病态情况，超出范围）。
          const mainAgent = yield* agents.get("main").pipe(Effect.orElseSucceed(() => undefined))
          if (mainAgent && Permission.disabled(["task"], mainAgent.permission).has("task")) return false
          // 按消息的 `tools` 是第二层工具剥离（llm.ts:720 的
          // `input.user.tools?.[k] !== false` 过滤），独立于 Permission.disabled。
          // 一个为其 turn 固定了狭窄工具集的斜杠命令，可能在权限允许的情况下仍丢掉
          // `task`；此时提示就无法满足。跳过理由相同，只是作用窗口更窄。
          if (lastUser.tools?.["task"] === false) return false

          const count = yield* taskGateState.get(sessionID)
          // runLoop 被标注为 `R = never`；TaskGate.decide 会引入一个
          // TaskRegistry.Service 依赖需求，我们在本地用已由 layer 解析的绑定把它闭合，
          // 使其不会泄漏进 runLoop 的 R 集合。
          const decision = yield* TaskGate.decide({
            session_id: sessionID,
            owner: undefined,
            reactCount: count,
            maxReact: MAX_TASK_GATE_MAIN_REACT,
            mode: "main",
          }).pipe(Effect.provideService(TaskRegistry.Service, taskRegistry))
          if (!decision.needReentry) {
            if (decision.capExceeded) {
              yield* slog.warn("task gate hit cap; allowing stop", {
                sessionID,
                incompleteTasks: decision.incompleteTasks,
              })
            }
            yield* taskGateState.clear(sessionID)
            return false
          }
          yield* taskGateState.bump(sessionID)
          const reentry = yield* sessions.updateMessage({
            id: MessageID.ascending(),
            role: "user" as const,
            sessionID,
            agentID: lastUser.agentID,
            agent: lastUser.agent,
            model: lastUser.model,
            tools: lastUser.tools,
            format: lastUser.format,
            time: { created: Date.now() },
          })
          yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: reentry.id,
            sessionID,
            type: "text",
            synthetic: true,
            text: decision.reentryText,
          } satisfies MessageV2.TextPart)
          return true
        })

        // Goal 停止条件闸门（仅主 agent）。在允许停止之前，一个独立的裁判模型读取
        // 对话记录并判断当前活跃的 goal 是否已满足。未满足 → 把裁判给出的理由作为
        // 合成用户 turn 注入，并向调用方发出继续工作的信号（返回 true）。这是
        // actor.preStop ReAct 重入的主循环对应物，后者只对派生 actor 触发。对任何
        // 裁判错误采取 fail-open，这样一个不稳定的裁判永远不会困住用户。
        const goalGate = Effect.fn("SessionPrompt.goalGate")(function* (lastUser: MessageV2.User) {
          if ((agentID ?? "main") !== "main") return false
          const active = yield* goal.get(sessionID)
          if (!active) return false

          const transcriptMsgs = yield* MessageV2.filterCompactedEffect(sessionID, {
            contextFrom: session.contextFrom,
            contextWatermark: session.contextWatermark,
            agentID: "main",
          })
          // 把裁决锚定到裁判刚刚评估过的那个助手 turn，这样 TUI 就能渲染一个
          // 按 turn 的标记，供用户回溯定位。
          const judgedMessageID = transcriptMsgs.findLast((m) => m.info.role === "assistant")?.info.id
          const verdict = yield* goal
            .evaluate({
              condition: active.condition,
              msgs: transcriptMsgs,
              model: lastUser.model,
            })
            .pipe(
              Effect.catch((err) =>
                Effect.gen(function* () {
                  yield* slog.warn("goal judge failed; allowing stop", { error: String(err) })
                  return { ok: true, reason: "judge error", judgeFailed: true } as Goal.Verdict & {
                    judgeFailed: true
                  }
                }),
              ),
            )

          if (verdict.ok || verdict.impossible) {
            yield* slog.info("goal satisfied; allowing stop", {
              sessionID,
              impossible: verdict.impossible === true,
            })
            // 发布最终裁决（goal 已清除），使 TUI 能在指示器消失前渲染
            // ✓/⊘ 结果行。goal.clear 也会发布 goal:undefined，但 TUI 会让
            // lastVerdict 保持粘滞。
            yield* bus.publish(Goal.Event.Updated, {
              sessionID,
              goal: undefined,
              lastVerdict: {
                ...verdict,
                attempt: active.react,
                messageID: judgedMessageID,
                error: "judgeFailed" in verdict ? true : undefined,
              },
            })
            yield* goal.clear(sessionID)
            return false
          }

          const count = yield* goal.bumpReact(sessionID)
          if (count > MAX_GOAL_REACT) {
            yield* slog.warn("goal hit MAX_GOAL_REACT cap; allowing stop", {
              sessionID,
              condition: active.condition,
              count,
            })
            yield* bus.publish(Goal.Event.Updated, {
              sessionID,
              goal: undefined,
              lastVerdict: { ...verdict, attempt: count, messageID: judgedMessageID },
            })
            yield* goal.clear(sessionID)
            return false
          }

          yield* slog.info("goal not satisfied; re-entering", { sessionID, attempt: count })
          yield* bus.publish(Goal.Event.Updated, {
            sessionID,
            goal: { condition: active.condition },
            lastVerdict: { ...verdict, attempt: count, messageID: judgedMessageID },
          })
          const reentry = yield* sessions.updateMessage({
            id: MessageID.ascending(),
            role: "user" as const,
            sessionID,
            agentID: lastUser.agentID,
            agent: lastUser.agent,
            model: lastUser.model,
            tools: lastUser.tools,
            format: lastUser.format,
            time: { created: Date.now() },
          })
          yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: reentry.id,
            sessionID,
            type: "text",
            synthetic: true,
            text: [
              "<system-reminder>",
              `Your goal is not yet satisfied: "${active.condition}".`,
              "A judge reviewed the transcript and reported what is still missing:",
              verdict.reason,
              "Keep working toward the goal. Do not stop until it is genuinely met or impossible.",
              "</system-reminder>",
            ].join("\n"),
          } satisfies MessageV2.TextPart)
          return true
        })

        // 仅推理（think-only）/ 空（empty，什么都没有）的步骤以非工具方式停止，
        // 却没有携带任何可用答案。若不干预，循环就会中断，把一个没有最终文本的助手
        // 消息交给用户。提示模型给出最终答案或调用一个真正的工具；一旦共享计数器
        // 耗尽就放弃（写入一个终止性错误），这样我们永远不会无限循环。
        const autoContinueInvalidOutput = Effect.fn("SessionPrompt.autoContinueInvalidOutput")(function* (input: {
          lastUser: MessageV2.User
          assistant: MessageV2.Assistant
          reason: string
        }) {
          if (input.assistant.error || input.assistant.summary || input.assistant.structured !== undefined) return false
          if (invalidContinuations >= INVALID_OUTPUT_CONTINUATION_LIMIT) {
            input.assistant.error = new MessageV2.InvalidOutputError({ message: input.reason }).toObject()
            yield* sessions.updateMessage(input.assistant)
            yield* bus.publish(Session.Event.Error, {
              sessionID: input.assistant.sessionID,
              error: input.assistant.error,
            })
            return false
          }

          invalidContinuations++
          yield* slog.info("auto-continuing invalid output", { attempt: invalidContinuations, reason: input.reason })
          const msg = yield* sessions.updateMessage({
            id: MessageID.ascending(),
            role: "user" as const,
            sessionID: input.lastUser.sessionID,
            agentID: input.lastUser.agentID,
            agent: input.lastUser.agent,
            model: input.lastUser.model,
            tools: input.lastUser.tools,
            format: input.lastUser.format,
            time: { created: Date.now() },
          })
          yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: msg.id,
            sessionID: msg.sessionID,
            type: "text",
            synthetic: true,
            text: [
              "<system-reminder>",
              "Your previous response contained no usable answer (it had only reasoning, or was empty).",
              "Provide a final answer to the user now, or call a valid tool to make progress on the task.",
              "Do not respond with only reasoning/thinking.",
              "</system-reminder>",
            ].join("\n"),
          } satisfies MessageV2.TextPart)
          return true
        })

        // 文本形式工具调用的恢复。模型把工具调用序列化成了散文文本而非结构化的
        // tool_use（大上下文下的一种退化状态）。通过设置 assistant.error 把这个坏的
        // 助手 turn 从历史中*丢弃*（toModelMessages 会跳过 info.error 被设置的消息，
        // 见 message-v2.ts），这样它既不会把对话搁浅在一个助手 turn 上（导致 provider
        // 预填充被拒），也不会污染后续上下文。随后我们重试该请求（调用方执行
        // `continue`，不创建新消息）。耗尽后错误保持终止性。返回 true ⇒ continue；
        // false ⇒ break。
        const autoRetryTextToolCall = Effect.fn("SessionPrompt.autoRetryTextToolCall")(function* (input: {
          lastUser: MessageV2.User
          assistant: MessageV2.Assistant
        }) {
          // 上一趟已经丢弃过——让 classify 落到 `failed`，而不是再次检测并浪费一次重试。
          if (input.assistant.error) return false
          // 把这个坏 turn 从请求历史中丢弃：toModelMessages 会跳过 info.error 被设置的
          // 消息，这样它既不会把对话搁浅在一个助手 turn 上，也不会污染后续上下文。
          input.assistant.error = new MessageV2.TextToolCallError({
            message: "Model emitted a tool call as text instead of a structured tool call.",
          }).toObject()
          yield* sessions.updateMessage(input.assistant)
          if (textToolCallRetries >= TEXT_TOOL_CALL_RETRY_LIMIT) {
            yield* bus.publish(Session.Event.Error, {
              sessionID: input.assistant.sessionID,
              error: input.assistant.error,
            })
            return false
          }
          textToolCallRetries++
          yield* slog.info("retrying text-form tool call", { attempt: textToolCallRetries })
          // 追加一个合成用户 turn，使被丢弃的助手消息变得陈旧（classify 的陈旧性保护），
          // 并让循环推进到生成阶段——与 autoRetryStructuredOutput 一致。没有这一步，
          // 循环会重入、再次检测到同一个 turn，并在零次模型调用的情况下耗尽重试。
          const msg = yield* sessions.updateMessage({
            id: MessageID.ascending(),
            role: "user" as const,
            sessionID: input.lastUser.sessionID,
            agentID: input.lastUser.agentID,
            agent: input.lastUser.agent,
            model: input.lastUser.model,
            tools: input.lastUser.tools,
            format: input.lastUser.format,
            time: { created: Date.now() },
          })
          yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: msg.id,
            sessionID: msg.sessionID,
            type: "text",
            synthetic: true,
            text: [
              "<system-reminder>",
              "Your previous response wrote a tool call as plain text instead of invoking the tool.",
              "Re-issue it through the real tool channel — emit a structured tool call, not text.",
              "Do not paste the tool call as text again.",
              "</system-reminder>",
            ].join("\n"),
          } satisfies MessageV2.TextPart)
          return true
        })

        // json_schema 模式，但模型从未产出结构化输出（纯文本停止、空、仅推理，或任何
        // 其他非工具的终止态）。用一条修复提示重试至多 lastUser.format.retryCount 次；
        // 耗尽后写入一个携带*真实*重试次数的 StructuredOutputError。与
        // invalidContinuations 分离：结构化重试受每次请求的 retryCount 约束，
        // 而非通用的 invalid-output 上限。
        const autoRetryStructuredOutput = Effect.fn("SessionPrompt.autoRetryStructuredOutput")(function* (input: {
          lastUser: MessageV2.User
          assistant: MessageV2.Assistant
        }) {
          if (input.assistant.error || input.assistant.summary || input.assistant.structured !== undefined) return false
          const limit = input.lastUser.format?.type === "json_schema" ? input.lastUser.format.retryCount : 0
          if (structuredRetries >= limit) {
            input.assistant.error = new MessageV2.StructuredOutputError({
              message: "Model did not produce structured output",
              retries: structuredRetries,
            }).toObject()
            yield* sessions.updateMessage(input.assistant)
            yield* bus.publish(Session.Event.Error, {
              sessionID: input.assistant.sessionID,
              error: input.assistant.error,
            })
            return false
          }

          structuredRetries++
          yield* slog.info("retrying structured output", { attempt: structuredRetries })
          const msg = yield* sessions.updateMessage({
            id: MessageID.ascending(),
            role: "user" as const,
            sessionID: input.lastUser.sessionID,
            agentID: input.lastUser.agentID,
            agent: input.lastUser.agent,
            model: input.lastUser.model,
            tools: input.lastUser.tools,
            // 必须携带 format，这样下一次迭代才会重新注册 StructuredOutput 工具。
            format: input.lastUser.format,
            time: { created: Date.now() },
          })
          yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: msg.id,
            sessionID: msg.sessionID,
            type: "text",
            synthetic: true,
            text: [
              "<system-reminder>",
              "Your previous response did not produce valid structured output via the StructuredOutput tool",
              "(it was plain text, empty, or only reasoning).",
              "You MUST call the StructuredOutput tool now, passing JSON that matches the requested schema.",
              "Do not reply with plain text and do not respond with only reasoning/thinking.",
              "</system-reminder>",
            ].join("\n"),
          } satisfies MessageV2.TextPart)
          return true
        })

        // 滑动窗口 n-gram 重复恢复。在主分支和 fork 分支间对称：第 1 次命中注入
        // REMIND，第 2 次命中注入 REPLAN，第 3 次命中（>= TEXT_NGRAM_MAX_RECOVERY）
        // 写入错误并发出 break 信号。
        const handleTextRepeat = Effect.fn("SessionPrompt.handleTextRepeat")(function* (input: {
          lastUser: MessageV2.User
        }) {
          if (textNgramRecoveryAttempts >= TEXT_NGRAM_MAX_RECOVERY) {
            yield* slog.info("text n-gram: max recovery exceeded, terminating")
            yield* bus.publish(Session.Event.Error, {
              sessionID,
              error: new NamedError.Unknown({
                message: `Text repetition detected: repeated n-grams after ${TEXT_NGRAM_MAX_RECOVERY} recovery attempts. Session terminated.`,
              }).toObject(),
            })
            return false
          }
          const recoveryText =
            textNgramRecoveryAttempts === 0 ? TEXT_NGRAM_RECOVERY_REMIND : TEXT_NGRAM_RECOVERY_REPLAN
          const reentry = yield* sessions.updateMessage({
            id: MessageID.ascending(),
            role: "user" as const,
            sessionID,
            agentID: input.lastUser.agentID,
            agent: input.lastUser.agent,
            model: input.lastUser.model,
            tools: input.lastUser.tools,
            format: input.lastUser.format,
            time: { created: Date.now() },
          })
          yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: reentry.id,
            sessionID,
            type: "text",
            synthetic: true,
            text: recoveryText,
          } satisfies MessageV2.TextPart)
          textNgramRecoveryAttempts++
          yield* slog.info("text n-gram: recovery injected", { attempt: textNgramRecoveryAttempts })
          return true
        })

        // Empty/no-op tool-call loop guard. Symmetric across main and fork
        // branches, mirroring handleTextRepeat's soft→hard ladder but keyed on
        // *empty steps* (empty/invalid tool input, or a fully empty terminal)
        // rather than repeated text n-grams — the gap TEXT_NGRAM and
        // stepSignature both miss (an empty tool call has no text to match and
        // is dropped by stepSignature's undefined path).
        //
        // Returns:
        //   "none"     — the step was NOT empty; streak reset, caller continues
        //                normal classification.
        //   "continue" — empty step, still within the soft-nudge budget; a
        //                remind/replan reminder was injected, caller should loop.
        //   "halt"     — empty streak exceeded EMPTY_STEP_MAX_RECOVERY; a
        //                terminal error was published, caller must break.
        const handleEmptyStep = Effect.fn("SessionPrompt.handleEmptyStep")(function* (input: {
          lastUser: MessageV2.User
          assistant: MessageV2.Assistant
        }) {
          // Never mask a genuine terminal outcome as an "empty loop": an errored
          // step, a content-filter/error finish, or an already-resolved
          // structured/summary step must fall through to its own classifier
          // handler (writeContentFilterError / writeModelError / final). Those
          // are terminal safety/error events, not a spinning no-op.
          if (
            input.assistant.error ||
            input.assistant.summary ||
            input.assistant.structured !== undefined ||
            input.assistant.finish === "content-filter" ||
            input.assistant.finish === "error"
          ) {
            return "none" as const
          }
          const parts = MessageV2.parts(input.assistant.id)
          if (!isEmptyStep(parts)) {
            emptyStepStreak = 0
            return "none" as const
          }
          emptyStepStreak++
          if (emptyStepStreak > EMPTY_STEP_MAX_RECOVERY) {
            yield* slog.info("empty step: max recovery exceeded, terminating", { streak: emptyStepStreak })
            hardHalt = true
            // Discard the empty turn from request history so it can neither
            // strand the conversation on an assistant prefill nor poison later
            // context (toModelMessages skips a message whose info.error is set).
            if (!input.assistant.error) {
              input.assistant.error = new NamedError.Unknown({
                message: `Empty tool call loop detected: ${emptyStepStreak} consecutive empty/no-op steps after ${EMPTY_STEP_MAX_RECOVERY} recovery attempts. Session terminated.`,
              }).toObject()
              yield* sessions.updateMessage(input.assistant)
            }
            yield* bus.publish(Session.Event.Error, {
              sessionID,
              error: new NamedError.Unknown({
                message: `Empty tool call loop detected: ${emptyStepStreak} consecutive empty/no-op steps after ${EMPTY_STEP_MAX_RECOVERY} recovery attempts. Session terminated.`,
              }).toObject(),
            })
            return "halt" as const
          }
          const recoveryText =
            emptyStepStreak === 1 ? EMPTY_STEP_RECOVERY_REMIND : EMPTY_STEP_RECOVERY_REPLAN
          const reentry = yield* sessions.updateMessage({
            id: MessageID.ascending(),
            role: "user" as const,
            sessionID,
            agentID: input.lastUser.agentID,
            agent: input.lastUser.agent,
            model: input.lastUser.model,
            tools: input.lastUser.tools,
            format: input.lastUser.format,
            time: { created: Date.now() },
          })
          yield* sessions.updatePart({
            id: PartID.ascending(),
            messageID: reentry.id,
            sessionID,
            type: "text",
            synthetic: true,
            text: recoveryText,
          } satisfies MessageV2.TextPart)
          yield* slog.info("empty step: recovery injected", { streak: emptyStepStreak })
          return "continue" as const
        })


        // content-filter is terminal on first occurrence: re-sending the same
        // turn would just get filtered again, so there is no nudge / counter.
        // Write a user-visible error (rendered via the session.error toast) and
        // let the caller break.
        // content-filter 在首次出现时即为终止性:重发同一个 turn 只会再次被过滤,
        // 所以没有提示 / 计数器。写入一个用户可见的错误(通过 session.error toast 渲染),并让调用方 break。
        const writeContentFilterError = Effect.fn("SessionPrompt.writeContentFilterError")(function* (input: {
          assistant: MessageV2.Assistant
        }) {
          if (input.assistant.error) return
          input.assistant.error = new MessageV2.ContentFilterError({
            message: "The response was withheld by the model provider's content safety filter.",
          }).toObject()
          yield* sessions.updateMessage(input.assistant)
          yield* bus.publish(Session.Event.Error, {
            sessionID: input.assistant.sessionID,
            error: input.assistant.error,
          })
        })

        // `failed` 分类（模型以 "error" 结束，或流错误路径已设置了错误）是终止性的。
        // 如果该步骤已经携带了错误（例如流抛出时写入的 APIError，processor.ts:581），
        // 就保留它；否则写入一个 ModelError，使循环永远不会在没有用户可见失败的情况下
        // 悄无声息地中断。
        const writeModelError = Effect.fn("SessionPrompt.writeModelError")(function* (input: {
          assistant: MessageV2.Assistant
          reason: string
        }) {
          if (input.assistant.error) return
          input.assistant.error = new MessageV2.ModelError({ message: input.reason }).toObject()
          yield* sessions.updateMessage(input.assistant)
          yield* bus.publish(Session.Event.Error, {
            sessionID: input.assistant.sessionID,
            error: input.assistant.error,
          })
        })

        // ╔══════════════════════════════════════════════════════════════════════════╗
        // ║ runLoop 主循环（ReAct 循环）总览                                           ║
        // ╠══════════════════════════════════════════════════════════════════════════╣
        // ║ 每一圈 = 一个 step，本质是「读上下文 → 调模型 → 执行工具 → 结果喂回 → 再来   ║
        // ║ 一圈」，直到模型交出最终答案。一圈的顺序大致是：                            ║
        // ║   1) 标记 busy、drain 收件箱、按 agentID 过滤出本 agent 可见的消息切片       ║
        // ║   2) 先对「已有的」助手消息分类，能提前判定就不白调模型                      ║
        // ║   3) step++；首圈做杂活（生成标题 / auto-dream / distill / cron）           ║
        // ║   4) 特殊路由：compaction 边界标记 / subtask 派生 / 上下文压力提醒           ║
        // ║   5) 溢出检查：主 agent 走 checkpoint 重建、子 agent/回退走 compaction       ║
        // ║   6) 建空助手消息 → 调模型（fork 用冻结快照 / 主用现算前缀）→ 得 result     ║
        // ║   7) 过一连串「控制闸门」把 result 归纳成 outcome（"break" | "continue"）    ║
        // ║   8) outcome==="break" 时仍要过 taskGate / goalGate，可被否决而强制 continue ║
        // ║ 循环外收尾：后台 prune → 取 final → 发指标 → 唤醒型 peer 通知父级 → return   ║
        // ║ 整段被 Effect.onExit(firePostSession) 包裹：无论成功/失败/中断都触发         ║
        // ║ session.post 钩子（try/finally 语义）。                                     ║
        // ╠══════════════════════════════════════════════════════════════════════════╣
        // ║ 控制闸门场景一览（模型不可靠，每个闸门拦一种「异常」，能救则 continue 重试，  ║
        // ║ 救不动 / 达终态则 break；带 * 者仅主 agent）：                              ║
        // ║                                                                            ║
        // ║ ── 让循环「继续」的场景（continue，通常是重试或喂回工具结果）──             ║
        // ║ • 模型调用了工具            : result==="continue" 且未被其它闸门拦截         ║
        // ║ • 输出被 token 上限截断     : autoContinueOutputLength → 注入「接着说」      ║
        // ║ • 只思考没给答案 / 空内容   : autoContinueInvalidOutput（think-only/invalid）║
        // ║ • 工具调用被写成纯文本      : autoRetryTextToolCall（text-tool-call）        ║
        // ║ • 要 json_schema 却没产出   : autoRetryStructuredOutput                      ║
        // ║ • 复读机（可救档）          : handleTextRepeat → 提醒 / 换思路              ║
        // ║ • 空步骤（可救档）          : handleEmptyStep（调了工具但无实质内容）        ║
        // ║ • 上下文溢出               : 主 agent→checkpoint 重建；子 agent→compaction   ║
        // ║ • compaction 边界标记      : 路由到 compaction.process，非 stop 则 continue  ║
        // ║ • subtask 派生             : handleSubtask 后 continue                      ║
        // ║ *• taskGate 否决 break     : 模型想停但仍有未完成 task → 逼继续              ║
        // ║ *• goalGate 否决 break     : 裁判模型判定用户目标未达成 → 逼继续             ║
        // ║                                                                            ║
        // ║ ── 让循环「收工」的场景（break）──                                         ║
        // ║ • 模型正常说完              : result==="stop"                               ║
        // ║ • 拿到结构化输出            : structured !== undefined                       ║
        // ║ • 被内容安全过滤            : classify → filtered，写错误                    ║
        // ║ • 模型报错                  : classify → failed，写错误                      ║
        // ║ • 被插件取消                : session.userQuery.pre 里 cancel===true          ║
        // ║ • fork 快照丢失             : forkCtx 缺失 → actor 失败                       ║
        // ║ • 各「可救」闸门重试超限    : 复读/空转/文本工具调用/结构化重试用尽           ║
        // ║ • 硬熔断                    : hardHalt（连续异常触顶）                        ║
        // ╚══════════════════════════════════════════════════════════════════════════╝
        while (true) {
          // F55：只有主 agent 才把会话状态设为 busy；子 agent 的 runner
          // 不得触碰会话级状态（按 F47，非主 actor 的 Runner.onBusy 是 Effect.void）。
          if (!agentID || agentID === "main") yield* status.set(sessionID, { type: "busy" })
          yield* inbox.drain(sessionID, agentID ?? "main").pipe(Effect.ignore)
          yield* slog.info("loop", { step })

          // F37：按 agentID 过滤，使子 agent 的切片在同一会话内与主 agent 的切片保持
          // 隔离。没有这一步，通过 mimocode 的共享 sessionID 设计派生出来的 actor
          //（explore/general 等）会在这里看到父级的完整对话，从而偏离任务。
          // agentID === "main" => 主 agent 切片（DB 中 agent_id = 'main'），
          // agentID === "explore-1" => 只有 explore-1 的切片。
          let msgs = yield* MessageV2.filterCompactedEffect(sessionID, {
            contextFrom: session.contextFrom,
            contextWatermark: session.contextWatermark,
            agentID: agentID ?? "main",
          })

          let lastUser: MessageV2.User | undefined
          let lastAssistant: MessageV2.Assistant | undefined
          let lastFinished: MessageV2.Assistant | undefined
          let tasks: MessageV2.SubtaskPart[] = []
          for (let i = msgs.length - 1; i >= 0; i--) {
            const msg = msgs[i]
            if (!lastUser && msg.info.role === "user") lastUser = msg.info
            if (!lastAssistant && msg.info.role === "assistant") lastAssistant = msg.info
            if (!lastFinished && msg.info.role === "assistant" && msg.info.finish) lastFinished = msg.info
            if (lastUser && lastFinished) break
            const task = msg.parts.filter((part): part is MessageV2.SubtaskPart => part.type === "subtask")
            if (task && !lastFinished) tasks.push(...task)
          }

          if (!lastUser) throw new Error("No user message found in stream. This should never happen.")

          // 按用户消息的主动召回提醒。一旦会话有了任何 memory 产物（memory 目录已填充
          // 或已记录 task），就追加一段简短的召回协议，使 agent 查询
          // memory.search / task / actor / Read 的反射在许多 rebuild 之后的 turn 里
          // 仍保持活跃。每个 turn 约 120 token，以 hasMemoryOrTasks 为条件。
          const lastUserMsgForRecall = msgs.findLast((m) => m.info.role === "user")
          if (lastUserMsgForRecall) {
            const hasRecallTarget = yield* checkpoint
              .hasMemoryOrTasks(sessionID)
              .pipe(Effect.catch(() => Effect.succeed(false)))
            if (hasRecallTarget) {
              const sessMemDir = path.join(Global.Path.data, "memory", "sessions", sessionID)
              const hints = recallHintLines((yield* config.get()).tool)
              lastUserMsgForRecall.parts.push({
                id: PartID.ascending(),
                messageID: lastUserMsgForRecall.info.id,
                sessionID,
                type: "text" as const,
                synthetic: true,
                text: [
                  "<system-reminder>",
                  `This session has memory at ${sessMemDir}/. Recall content`,
                  "not in your context with:",
                  hints[0],
                  `- Read(file_path="${sessMemDir}/...")`,
                  hints[1],
                  hints[2],
                  "",
                  "Don't ask the user about something memory may already record.",
                  "</system-reminder>",
                ].join("\n"),
              })
            }
          }

          const lastAssistantMsg = msgs.findLast(
            (msg) => msg.info.role === "assistant" && msg.info.id === lastAssistant?.id,
          )
          // 有些 provider 即便助手消息里包含工具调用也会返回 "stop"。
          // 保持循环运行，以便把工具结果送回模型。
          // 跳过 provider 执行的工具 part——那些已在 provider 的流内部完全处理完毕
          //（例如 DWS Agent Platform），不需要再循环一次。
          const hasToolCalls =
            lastAssistantMsg?.parts.some((part) => part.type === "tool" && !part.metadata?.providerExecuted) ?? false

          // ==== 预检查阶段：先看「上一圈已有的助手消息」，能提前判定就不白调一次模型 ====
          if (
            lastAssistant?.finish === "length" &&
            !hasToolCalls &&
            lastUser.id < lastAssistant.id &&
            (yield* autoContinueOutputLength({ lastUser, assistant: lastAssistant }))
          ) {
            // 场景：上一圈的回答撞 token 上限被截断（§2.1）。已注入「接着写」提醒 → 再转一圈续写。
            continue
          }

          if (lastAssistant) {
            const classification = classifyAssistantStep({
              phase: "existing-assistant",
              lastUser,
              assistant: lastAssistant,
              parts: lastAssistantMsg?.parts ?? [],
            })
            if (classification.type === "filtered") {
              // 场景：上一圈被内容安全过滤（§4）。终态，重发只会再被过滤 → 写错误、收工。
              yield* writeContentFilterError({ assistant: lastAssistant })
              yield* slog.info("exiting loop", { classification: classification.type })
              break
            }
            if (classification.type === "failed") {
              // 场景：上一圈模型报错（§4）。终态 → 写错误、收工。
              yield* writeModelError({ assistant: lastAssistant, reason: classification.reason })
              yield* slog.info("exiting loop", { classification: classification.type, reason: classification.reason })
              break
            }
            if (classification.type === "text-tool-call") {
              // 场景：上一圈把工具调用写成了纯文本（§2.3）。能重试则再转一圈；重试用尽则收工。
              if (yield* autoRetryTextToolCall({ lastUser, assistant: lastAssistant })) continue
              yield* slog.info("exiting loop", { classification: classification.type })
              break
            }
            if (classification.type === "think-only" || classification.type === "invalid") {
              // 场景：上一圈只思考没给答案 / 空输出（§2.2）。能续写则再转一圈；续写用尽则收工。
              const reason = classification.type === "invalid" ? classification.reason : "think-only"
              if (yield* autoContinueInvalidOutput({ lastUser, assistant: lastAssistant, reason })) continue
              yield* slog.info("exiting loop", { classification: classification.type })
              break
            }
            if (classification.type === "final" && classification.degraded)
              yield* slog.warn("degraded final on abnormal finish", { finish: lastAssistant.finish })
            if (classification.type !== "continue") {
              // 场景：上一圈已是可收工的终态（final 等）。但收工前先过两道否决闸门（§5）：
              // 还有未完成 task → continue；goal 未达成 → continue；都放行才真正收工。
              if (yield* taskGate(lastUser)) continue
              if (yield* goalGate(lastUser)) continue
              yield* slog.info("exiting loop", { classification: classification.type })
              break
            }
          }

          step++
          // Per-step turn heartbeat: only writer of turn_count; advances last_turn_time/time_updated so the orchestrator can tell progressing children from stalled ones. Safe 0-row no-op when no registry row exists.
          yield* actorRegistry.updateTurn(sessionID, resolvedAgentID).pipe(Effect.ignore)
          if (step === 1)
            yield* title({
              session,
              agent: lastUser.agent,
              modelID: lastUser.model.modelID,
              providerID: lastUser.model.providerID,
              history: msgs,
            }).pipe(Effect.ignore, Effect.forkIn(scope))

          // ┌─ 顺带跑的后台维护，与当前这轮对话无关 ───────────────────────────────
          // │ 触发时机：仅「顶层会话（非子 agent）的第一圈」，即 step === 1 且没有 parentID。
          // │ 用意：借用户开启新对话的时机，在后台捎带跑周期性的自我维护，全程不阻塞、
          // │       不影响用户当前对话。三类维护任务：
          // │   · dream   —— 把历史对话里可靠、已验证的信息沉淀进项目 memory（默认每 7 天）
          // │   · distill —— 把反复出现的手动流程打包成可复用的 skill/agent（默认每 30 天）
          // │   · cron    —— 实验性：给会话挂一个定时任务桥（受 feature flag 保护）
          // │ 是否真的触发由 shouldAutoDream/shouldAutoDistill 决定，内部已带三层节流：
          // │   配置开关 + 10s 进程内防抖 + 距上次运行的时间间隔；查询出错时兜底为 false
          // │   （维护任务绝不能拖垮用户的正常对话）。
          // │ 执行方式：下面每个任务都用 AppRuntime.runPromise(...).catch(...) 各自「发射后
          // │   不管」——新开一个独立会话、在独立 runtime 上运行，出错只记日志，不 await，
          // │   因此不会卡住当前 runLoop。
          // └──────────────────────────────────────────────────────────────────────
          if (step === 1 && !session.parentID) {
            const cfg = yield* config.get()
            const dreamTrigger = yield* shouldAutoDream(cfg).pipe(Effect.catch(() => Effect.succeed(false)))
            const distillTrigger = yield* shouldAutoDistill(cfg).pipe(Effect.catch(() => Effect.succeed(false)))
            const mdl = { providerID: lastUser.model.providerID, modelID: lastUser.model.modelID }
            // AppRuntime 是动态导入的（不在模块顶层），以使 session 层不卷入 app-runtime
            // 的模块初始化循环（prompt → app-runtime → AppLayer → SessionPrompt）。
            // 仅在某个触发器真正触发时才加载。在完整 runtime 上以分离的 fire-and-forget 方式运行。
            const needAppRuntime = dreamTrigger || distillTrigger || Flag.MIMOCODE_EXPERIMENTAL_CRON
            if (needAppRuntime) {
              const { AppRuntime } = yield* Effect.promise(() => import("@/effect/app-runtime"))
              if (dreamTrigger) {
                AppRuntime.runPromise(
                  Session.Service.use((svc) =>
                    Effect.gen(function* () {
                      const s = yield* svc.create({ title: AUTO_DREAM_TITLE })
                      const sp = yield* Service
                      yield* sp.prompt({ sessionID: s.id, agent: "dream", model: mdl, parts: [{ type: "text", text: DREAM_TASK }] })
                    }),
                  ),
                ).catch((err) => log.error("auto-dream prompt failed", { error: String(err) }))
              }
              if (distillTrigger) {
                AppRuntime.runPromise(
                  Session.Service.use((svc) =>
                    Effect.gen(function* () {
                      const s = yield* svc.create({ title: AUTO_DISTILL_TITLE })
                      const sp = yield* Service
                      yield* sp.prompt({ sessionID: s.id, agent: "distill", model: mdl, parts: [{ type: "text", text: DISTILL_TASK }] })
                    }),
                  ),
                ).catch((err) => log.error("auto-distill prompt failed", { error: String(err) }))
              }
              // T18-bridge 挂载：每个新的顶层会话启动时触发一次
              // CronBridge.start(sessionID, workspaceRoot)。当 MIMOCODE_EXPERIMENTAL_CRON
              // 未设置时 bridge 本身会 no-op；外层的门控只是在常见情况下省去解析成本。
              // 与 auto-dream 的分离式动态导入模式一致，使 prompt.ts 不卷入 app-runtime
              // 的模块初始化循环。Bridge.start 通过其 `started` 保护是幂等的，其 Layer
              // finalizer 会在 scope 关闭时处理拆卸。
              if (Flag.MIMOCODE_EXPERIMENTAL_CRON) {
                const workspaceRoot = (yield* InstanceState.context).worktree
                const { CronBridge } = yield* Effect.promise(() => import("@/session/cron-bridge"))
                AppRuntime.runPromise(
                  CronBridge.use((b) => b.start(sessionID, workspaceRoot)),
                ).catch((err) => log.error("cron-bridge start failed", { sessionID, error: String(err) }))
              }
            }
          }

          const model = yield* getModel(lastUser.model.providerID, lastUser.model.modelID, sessionID)
          lastModelForPrune = model
          lastFinishedForPrune = lastFinished
          const task = tasks.pop()

          if (task?.type === "subtask") {
            // 场景：这一圈要派生一个子 agent 去干预定的子任务。派完就再转一圈（不在本圈调模型）。
            yield* handleSubtask({ task, model, lastUser, sessionID, session, msgs })
            continue
          }

          // 检测 compaction 边界：如果最后一条用户消息带有 compaction part，
          // 就路由到 compact.process()，而不是走正常的 LLM 流程。
          const lastUserMsgForCompaction = msgs.findLast((m) => m.info.role === "user")
          if (lastUserMsgForCompaction?.parts.some((p) => p.type === "compaction")) {
            const compactionPart = lastUserMsgForCompaction.parts.find(
              (p): p is MessageV2.CompactionPart => p.type === "compaction",
            )
            const allMsgs = yield* sessions.messages({ sessionID, agentID: lastUser.agentID ?? "main" })
            const result = yield* compaction.process({
              parentID: lastUser.id,
              messages: allMsgs,
              sessionID,
              auto: compactionPart?.auto ?? false,
              overflow: compactionPart?.overflow,
              agentID: lastUser.agentID,
            })
            // cron-sentinel 缓存通过 cron-bridge 内对 SessionCompaction.Event.Compacted
            // 的总线订阅来失效——见 `compaction.ts:468` 的 publish 与 `cron-bridge.ts`
            // 的 subscribe 这一对。既覆盖这里的用户 `/compact` 路径，也覆盖
            // compaction.create 里的溢出边界路径。
            // 场景：这一圈是压缩边界（用户 /compact 或自动溢出插入的标记），已执行历史摘要压缩。
            if (result === "stop") break // 压缩后判定该收工 → 收工
            continue // 否则用瘦身后的上下文再转一圈
          }

          // 高上下文压力下的内存刷写提示。
          //
          // 目的：在上下文填充较高时，会话可能很快就会做 checkpoint 并丢弃旧上下文，
          // 因此在那之前提醒模型把持久的学习成果外化到 memory。这是一条*保存你的工作*
          // 的提醒，*不是*收尾的信号。
          //
          // 它防范两种失败模式（都在生产环境中观察到过）：
          //   1. 读起来像"我们即将重置——收尾吧"的措辞，会让模型在任务中途过早结束
          //      其 turn 并把控制权交回给用户。下面的文本很明确：先持久化 memory，
          //      然后*继续*；不要结束 turn。
          //   2. 在压力持续偏高时于每一轮用户 turn 都重新注入提示，会把一次性的提醒
          //      变成逐 turn 的唠叨。我们现在在最近的对话窗口内去重，而不仅是当前
          //      这条用户消息。
          // 用户信息总结，上下文压缩
          if (lastFinished && lastFinished.summary !== true && model) {
            const cfg = yield* config.get()
            const pressure = pressureLevel({ cfg, tokens: lastFinished.tokens, model })
            if (pressure >= 2) {
              // 防抖：每个高压力片段（自上一个 checkpoint 边界以来的窗口）至多提示一次。
              // 关于为何以边界——而非固定消息数——作为正确的锚点，见 nudgedSinceBoundary。
              const NUDGE_MARKER = "Context is filling up"
              const boundaryID = yield* checkpoint
                .lastBoundary(sessionID)
                .pipe(Effect.catch(() => Effect.succeed(undefined)))
              const alreadyNudged = nudgedSinceBoundary(msgs, boundaryID, NUDGE_MARKER)
              const lastUserMsg = msgs.findLast((m) => m.info.role === "user")
              if (lastUserMsg && !alreadyNudged) {
                lastUserMsg.parts.push({
                  id: PartID.ascending(),
                  messageID: lastUserMsg.info.id,
                  sessionID,
                  type: "text",
                  synthetic: true,
                  text: [
                    // <系统提醒>
                    // 上下文容量即将占满（负载阈值：${pressure >= 3 时 >85%，否则 >70%}）。
                    // 本次会话中若存在尚未存入记忆的重要结论、关键决策或学习要点，请立刻记录下来（这些内容将在下一轮检查点统一摘要归档）。
                    // 本提示仅用于提醒保存工作进度。

                    // 重点要求：完成记忆写入后，**本轮继续执行原有任务**。
                    // 严禁因本提醒中断任务、收尾会话或将操作权限交还给用户；仅当全部实际工作完成后，方可结束流程。
                    // </系统提醒>
                    "<system-reminder>",
                    `Context is filling up (${pressure >= 3 ? ">85%" : ">70%"}).`,
                    "If you have important learnings or decisions from this session that are",
                    "not yet in memory, write them now (they may be summarized on the next",
                    "checkpoint). This is a save-your-work reminder only.",
                    "IMPORTANT: After writing to memory, CONTINUE with the current task in the",
                    "same turn. Do NOT stop, wrap up, or hand control back to the user because",
                    "of this reminder — only finish when the actual work is done.",
                    "</system-reminder>",
                  ].join("\n"),
                })
              }
            }
          }

          // 重复步骤提示：如果最近 REPEATED_STEP_THRESHOLD 个已完成的助手步骤发起了
          // 相同的工具调用，模型很可能陷入了循环。在最后一条用户消息上注入一条提醒，
          // 要求它改变思路。与上面的内存刷写提示相同（合成文本 part，按 build 去重）。
          // 防止死循环
          if (lastFinished) {
            const recentSignatures: string[] = []
            for (let i = msgs.length - 1; i >= 0 && recentSignatures.length < REPEATED_STEP_THRESHOLD; i--) {
              const m = msgs[i]
              if (m.info.role !== "assistant" || !m.info.finish) continue
              const sig = stepSignature(m.parts)
              if (sig === undefined) break
              recentSignatures.push(sig)
            }
            const repeating =
              recentSignatures.length === REPEATED_STEP_THRESHOLD &&
              recentSignatures.every((sig) => sig === recentSignatures[0])
            if (repeating) {
              const lastUserMsg = msgs.findLast((m) => m.info.role === "user")
              if (
                lastUserMsg &&
                !lastUserMsg.parts.some(
                  (p) => p.type === "text" && p.text?.includes("repeating the same action"),
                )
              ) {
                lastUserMsg.parts.push({
                  id: PartID.ascending(),
                  messageID: lastUserMsg.info.id,
                  sessionID,
                  type: "text",
                  synthetic: true,
                  text: [
                    // <系统提醒>
                    // 你最近连续 ${REPEATED_STEP_THRESHOLD} 步操作完全一致，一直在重复相同动作且没有任何进展。
                    // 请立刻停下重新思考：当前思路行不通。更换别的方案、调用其他工具；若陷入卡顿，向用户说明阻碍原因，不要再重复执行相同步骤。
                    // </系统提醒>
                    "<system-reminder>",
                    `Your last ${REPEATED_STEP_THRESHOLD} steps have been identical — you appear to be`,
                    "repeating the same action without making progress. Stop and reconsider:",
                    "the current approach is not working. Try a different strategy, use a",
                    "different tool, or if you are blocked, explain the blocker to the user",
                    "instead of repeating the same step again.",
                    "</system-reminder>",
                  ].join("\n"),
                })
              }
            }
          }

          // 为本次迭代解析一次 agent。下面的管理钩子（fireCheckpoints、溢出处理器）
          // 以及本次迭代后面已有的"agent 未找到"检查都复用这个绑定。
          // 有界计算 agent（native + hidden——目前是 title、summary、checkpoint-writer）
          // 免于上下文管理；见
          // docs/superpowers/specs/2026-04-28-bounded-computation-agents-design.md
          const agent = yield* agents.get(lastUser.agent)
          const isBoundedComputation =
            agent?.native === true && agent?.hidden === true

          // 基于最新已完成助手消息的 token 数，为任何新跨过的阈值触发后台 checkpoint
          // writer。必须在下面的溢出/maxThreshold 检查*之前*运行，这样 maxCrossed 标志
          // 才能及时置位，从而在同一次迭代里触发 rebuild。
          if (!skipOverflowCheck && !isBoundedComputation && lastFinished && lastFinished.tokens) {
            const fireOps = yield* ops()
            yield* prune
              .fireCheckpoints({
                sessionID,
                model,
                tokens: lastFinished.tokens,
                promptOps: fireOps,
                agentID: lastUser.agentID,
              })
              .pipe(Effect.ignore)
          }

          if (
            !skipOverflowCheck &&
            !isBoundedComputation &&
            lastFinished &&
            lastFinished.summary !== true &&
            (overflowCheck({ cfg: yield* config.get(), tokens: lastFinished.tokens, model }) ||
              (yield* prune.maxThresholdCrossed(sessionID)))
          ) {
            // 子 agent 溢出 → 按 actor 的 compaction（有损的 LLM 摘要，作用于该 actor 的
            //（sessionID, agent_id）切片）。子 agent 没有 checkpoint，所以 checkpoint+丢弃
            // 不适用。门控必须排除 agentID="main"——F49+F50 让 main 也携带
            // agentID="main"，所以裸的 `if (lastUser.agentID)` 会把 main 路由到这条
            // 子 agent 路径，从而跳过下面的 checkpoint rebuild。相应门控见 checkpoint.ts:715。
            if (lastUser.agentID && lastUser.agentID !== "main") {
              yield* compaction
                .create({
                  sessionID,
                  agent: lastUser.agent,
                  model: { providerID: model.providerID, modelID: model.id },
                  auto: true,
                  agentID: lastUser.agentID,
                })
                .pipe(Effect.ignore)
              // 插入边界后，该 actor 的 filterCompactedEffect 切片就从边界标记开始——
              // 上下文为下一次迭代的流被释放出来。跳过下一次溢出检查，让模型能在裁剪后的
              // 上下文上作答。
              skipOverflowCheck = true
              // 场景：子 agent 上下文溢出（§3.2），已做有损压缩 → 用瘦身后的上下文再转一圈。
              continue
            }

            // Main-agent overflow: insert a checkpoint boundary marker (never
            // deletes DB messages) so the next iteration rebuilds from the
            // freshest checkpoint. Shared with the manual `/rebuild` command via
            // rebuildFromCheckpoint so logic/boundary conditions can't drift.
            // Falls back to compaction only when no boundary can be produced.
            // 主 agent 溢出:插入一个 checkpoint 边界标记(绝不删 DB 消息),使下一次迭代从最新
            // checkpoint 重建。通过共享 helper rebuildFromCheckpoint 与手动 /rebuild 命令复用同一逻辑,
            // 避免漂移。只有在无法产生边界时才回退到 compaction。
            const inserted = yield* rebuildFromCheckpoint({
              sessionID,
              msgs,
              agentID: lastUser.agentID,
              agent: lastUser.agent,
              model: { providerID: model.providerID, id: model.id },
            })
            if (inserted) {
              skipOverflowCheck = true
              // 场景：主 agent 溢出（§3.2），已插入 checkpoint 边界 → 下一圈从 checkpoint 重建后再跑。
              continue
            }

            // F39：没有 checkpoint——回退到 compaction（LLM 驱动的有损摘要）。
            // 优于机械裁剪：通过摘要保留了语义内容。
            yield* compaction
              .create({
                sessionID,
                agent: lastUser.agent,
                model: { providerID: model.providerID, modelID: model.id },
                auto: true,
                agentID: lastUser.agentID,
              })
              .pipe(Effect.ignore)
            skipOverflowCheck = true
            // 场景：主 agent 溢出但无 checkpoint 可用，回退到有损压缩 → 再转一圈。
            continue
          }
          skipOverflowCheck = false

          // `agent` 已在迭代开始时解析；这里复用它来产生"agent 未找到"的用户可见错误。
          if (!agent) {
            const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
            const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
            const error = new NamedError.Unknown({ message: `Agent not found: "${lastUser.agent}".${hint}` })
            yield* bus.publish(Session.Event.Error, { sessionID, error: error.toObject() })
            throw error
          }
          const maxSteps = agent.steps ?? Infinity
          const isLastStep = step >= maxSteps
          msgs = yield* insertReminders({ messages: msgs, agent, session })

          const msg: MessageV2.Assistant = {
            id: MessageID.ascending(),
            parentID: lastUser.id,
            role: "assistant",
            agentID: lastUser.agentID,
            mode: agent.name,
            agent: agent.name,
            variant: lastUser.model.variant,
            path: { cwd: ctx.directory, root: ctx.worktree },
            cost: 0,
            tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
            modelID: model.id,
            providerID: model.providerID,
            time: { created: Date.now() },
            sessionID,
          }
          yield* sessions.updateMessage(msg)
          const handle = yield* processor.create({
            assistantMessage: msg,
            sessionID,
            model,
            agentMetrics,
          })

          const outcome: "break" | "continue" = yield* Effect.gen(function* () {
            const lastUserMsg = msgs.findLast((m) => m.info.role === "user")
            const bypassAgentCheck = lastUserMsg?.parts.some((p) => p.type === "agent") ?? false

            const tools = yield* resolveTools({
              agent,
              session,
              model,
              tools: lastUser.tools,
              processor: handle,
              bypassAgentCheck,
              messages: msgs,
              agentID: lastUser.agentID,
              task_id,
            })

            if (lastUser.format?.type === "json_schema") {
              tools["StructuredOutput"] = createStructuredOutputTool({
                schema: lastUser.format.schema,
                onSuccess(output) {
                  structured = output
                },
              })
            }

            if (step === 1)
              yield* summary.summarize({ sessionID, messageID: lastUser.id }).pipe(Effect.ignore, Effect.forkIn(scope))

            // 当前任务没完成时，用户发送了新的消息触发
            if (step > 1 && lastFinished) {
              for (const m of msgs) {
                if (m.info.role !== "user" || m.info.id <= lastFinished.id) continue
                for (const p of m.parts) {
                  if (p.type !== "text" || p.ignored || p.synthetic) continue
                  if (!p.text.trim()) continue
                  p.text = [
                    // <系统提醒>
                    // 用户发送了如下消息：
                    // ${p.text}

                    // 请回应这条消息，并继续完成当前任务。
                    // </系统提醒>
                    "<system-reminder>",
                    "The user sent the following message:",
                    p.text,
                    "",
                    "Please address this message and continue with your tasks.",
                    "</system-reminder>",
                  ].join("\n")
                }
              }
            }

            yield* plugin.trigger("experimental.chat.messages.transform", {}, { messages: msgs })

            const format = lastUser.format ?? { type: "text" as const }

            // 判断本次迭代是否针对一个 fork agent（contextMode === "full"）。
            // fork agent 使用在派生时捕获的冻结 ForkContext 快照（system +
            // inheritedMessages），而不是从自身的 agent 身份重新计算——那会与父级
            // 产生偏差并破坏前缀缓存。
            const actorRecord = lastUser.agentID
              ? yield* actorRegistry.get(sessionID, lastUser.agentID).pipe(
                  Effect.orElseSucceed(() => undefined),
                )
              : undefined
            // v9 把 main 注册为 `mode: "main"` 且 `contextMode: "full"`。
            // 只有派生出来的 actor（subagent/peer）才携带冻结的 ForkContext；
            // main 是捕获者，永远不是被捕获者。
            const isForkAgent =
              actorRecord?.contextMode === "full" &&
              (actorRecord.mode === "subagent" || actorRecord.mode === "peer")

            // fork 路径：从 Actor 服务读取冻结的 ForkContext（通过 spawnRef 晚绑定，
            // 以打破 Actor → SessionPrompt → Actor 的层循环）。如果 forkCtx 缺失
            //（竞态 / 清理 bug / 派生被跳过），就让该 actor 失败，这样下一次 prune turn
            // 可以派生一个新的 fork。
            if (isForkAgent) {
              const forkCtxEffect = spawnRef.current?.getForkContext(lastUser.agentID!)
              const forkCtx = forkCtxEffect ? yield* forkCtxEffect : undefined
              if (!forkCtx) {
                yield* slog.warn("fork agent runLoop: missing forkContext, failing actor", {
                  sessionID,
                  agentID: lastUser.agentID,
                })
                yield* actorRegistry
                  .updateStatus(sessionID, lastUser.agentID!, { status: "idle", lastOutcome: "failure", lastError: "missing fork context" })
                  .pipe(Effect.ignore)
                // 场景：fork 子 agent 的冻结快照丢了（§4）→ 标记 actor 失败、收工（下轮由 prune 重派生）。
                return "break" as const
              }
              const ownNew = msgs.filter(
                (m) => m.info.id > forkCtx.watermarkMsgID && m.info.agentID === lastUser.agentID,
              )
              const ownNewModelMsgs = yield* MessageV2.toModelMessagesEffect(ownNew, model)
              const prebuiltSystem = forkCtx.system
              const modelMsgs: ModelMessage[] = [...forkCtx.inheritedMessages, ...ownNewModelMsgs]
              // 对 fork agent 来说 additions 为空：system 逐字取自 forkCtx.system。
              // 作为 `system` 传给 handle.process 用于日志/回放。
              const additions: string[] = []
              // 注意：fork 使用来自 resolveTools 的 `tools`（而非 `forkCtx.tools`）——运行时
              // 的工具分发需要 execute 闭包，而 `forkCtx.tools` 并不携带它。目前与父级的
              // schema 一致是 checkpoint-writer 没有 toolAllowlist 的结果（Task 2.6 +
              // agent.test.ts 守卫）。完整契约见 packages/opencode/src/actor/spawn.ts 中
              // ForkContext.tools 的 JSDoc。
              const queryParts =
                msgs.findLast((m) => m.info.role === "user" && m.info.id === lastUser.id)?.parts ?? []
              const query = userQueryText(queryParts)
              const preQuery = {
                cancel: undefined as boolean | undefined,
                cancelReason: undefined as string | undefined,
              }
              yield* plugin.trigger(
                "session.userQuery.pre",
                { sessionID, agentID: resolvedAgentID, step, messageID: lastUser.id, query },
                preQuery,
              )
              if (preQuery.cancel) {
                cancelled = true
                cancelReason = preQuery.cancelReason
                handle.message.error = new MessageV2.AbortedError({
                  message: preQuery.cancelReason ?? "Step cancelled by plugin",
                }).toObject()
                handle.message.finish = "cancelled"
                yield* sessions.updateMessage(handle.message)
                yield* plugin.trigger(
                  "session.userQuery.post",
                  {
                    sessionID,
                    agentID: resolvedAgentID,
                    step,
                    messageID: lastUser.id,
                    query,
                    assistantMessageID: handle.message.id,
                    finish: handle.message.finish,
                    error: preQuery.cancelReason,
                    trajectory: trajectoryForStep(msgs, handle.message),
                  },
                  {},
                )
                // 场景：插件在 session.userQuery.pre 里取消了这一步（§4）→ 写取消错误、收工。
                return "break" as const
              }
              const result = yield* handle
                .process({
                  user: lastUser,
                  agent,
                  // fork 继承父 agent 的权限（在派生时捕获进 ForkContext）。这驱动
                  // llm.ts 的 resolveTools/disabled() 得到与父级*相同*的可见工具集
                  // → 在继承的前缀上保持 prompt 缓存一致。作用范围：这只影响工具的*可见性*；
                  // 每次调用的 ask ruleset（在 resolveTools 的 ask 闭包里单独构建）不变。
                  // 除非存在非默认的 `session.permission`，否则一致性是精确的：父级的可见性
                  // ruleset 是 merge(parent.permission, session.permission)，而 fork 的是
                  // merge(writer.permission, parentPermission)——所以会话级规则会固定父级，
                  // 但不会固定 fork。这仍然严格优于旧的定制 "*":"deny" 块（后者总是有偏差）。
                  // `?? session.permission` 仅作为纵深防御：parentPermission 是必填字段
                  //（捕获失败时为空 `[]`，而 `??` *不会*覆盖它），所以这个回退只有在未来的
                  // 重构把该字段改为可选时才会触发。
                  permission: forkCtx.parentPermission ?? session.permission,
                  sessionID,
                  parentSessionID: session.parentID,
                  system: additions,
                  prebuiltSystem,
                  messages: [...modelMsgs, ...(isLastStep ? [{ role: "user" as const, content: MAX_STEPS }] : [])],
                  tools,
                  model,
                  toolChoice: isLastStep ? "none" : format.type === "json_schema" ? "required" : undefined,
                  agentID: lastUser.agentID,
                })
                .pipe(
                  Effect.onExit((exit) =>
                    plugin
                      .trigger(
                        "session.userQuery.post",
                        {
                          sessionID,
                          agentID: resolvedAgentID,
                          step,
                          messageID: lastUser.id,
                          query,
                          assistantMessageID: handle.message.id,
                          finish: handle.message.finish,
                          error: Exit.isFailure(exit)
                            ? Cause.pretty(exit.cause)
                            : sessionErrorText(handle.message.error),
                          finalText: assistantFinalText(handle.message, MessageV2.parts(handle.message.id)),
                          trajectory: trajectoryForStep(msgs, handle.message),
                        },
                        {},
                      )
                      .pipe(Effect.ignore),
                  ),
                )

              // ==== fork 分支：把这一步模型返回归纳成 outcome（break/continue）====
              if (
                result === "continue" &&
                (yield* autoContinueOutputLength({ lastUser, assistant: handle.message }))
              ) {
                // 场景：输出被 token 截断（§2.1），已注入续写提醒 → continue。
                return "continue" as const
              }

              if (result === "text-repeat") {
                // 场景：流内复读（§2.5）。能救 continue，救不动 break。
                if (yield* handleTextRepeat({ lastUser })) return "continue" as const
                return "break" as const
              }

              if (structured !== undefined) {
                // 场景：拿到了要求的结构化输出（§4）→ 存下、收工。
                handle.message.structured = structured
                handle.message.finish = handle.message.finish ?? "stop"
                yield* sessions.updateMessage(handle.message)
                return "break" as const
              }

              // Empty/no-op tool-call loop guard (fork branch). Intercept before
              // classify would `continue` an empty tool-calls step: soft-nudge
              // within budget, hard-halt once exceeded. A non-empty step returns
              // "none" and falls through to normal classification.
              const forkEmptyStep = yield* handleEmptyStep({ lastUser, assistant: handle.message })
              if (forkEmptyStep === "halt") return "break" as const // 场景：空调用循环超限（§2.6）→ 硬熔断收工
              if (forkEmptyStep === "continue") return "continue" as const // 场景：空步骤可救，已提醒 → continue

              const forkClassification = classifyAssistantStep({
                phase: "after-process",
                lastUser,
                assistant: handle.message,
                parts: MessageV2.parts(handle.message.id),
                processResult: result,
              })
              if (forkClassification.type === "filtered") {
                // 场景：被内容安全过滤（§4）→ 写错误、收工。
                yield* writeContentFilterError({ assistant: handle.message })
                return "break" as const
              }
              if (forkClassification.type === "failed") {
                // 场景：模型报错（§4）→ 写错误、收工。
                yield* writeModelError({ assistant: handle.message, reason: forkClassification.reason })
                return "break" as const
              }
              if (forkClassification.type === "text-tool-call") {
                // 场景：工具调用写成了纯文本（§2.3）。能重试 continue，用尽 break。
                if (yield* autoRetryTextToolCall({ lastUser, assistant: handle.message })) return "continue" as const
                return "break" as const
              }
              if (forkClassification.type !== "continue" && !handle.message.error && format.type === "json_schema") {
                // 场景：要 json_schema 但没产出结构化输出（§2.4）。能重试 continue，用尽 break。
                if (yield* autoRetryStructuredOutput({ lastUser, assistant: handle.message }))
                  return "continue" as const
                return "break" as const
              }

              if (
                (forkClassification.type === "think-only" || forkClassification.type === "invalid") &&
                format.type !== "json_schema"
              ) {
                // 场景：只思考没答案 / 空输出（§2.2）。能续写 continue，用尽 break。
                const reason =
                  forkClassification.type === "invalid" ? forkClassification.reason : "think-only"
                if (yield* autoContinueInvalidOutput({ lastUser, assistant: handle.message, reason }))
                  return "continue" as const
                return "break" as const
              }

              if (forkClassification.type === "final" && forkClassification.degraded)
                yield* slog.warn("degraded final on abnormal finish", { finish: handle.message.finish })
              if (result === "stop") return "break" as const // 场景：模型正常说完（§4）→ 收工
              // 场景：子 agent 上下文溢出（§3.2）→ 按 actor 做有损压缩，随后 continue。
              // fork agent 始终是子 agent（lastUser.agentID 已设置）；溢出时使用
              // 按 actor 的 compaction（与非 fork 的子 agent 路径相同）。
              if (!isBoundedComputation && result === "overflow") {
                yield* compaction
                  .create({
                    sessionID,
                    agent: lastUser.agent,
                    model: { providerID: model.providerID, modelID: model.id },
                    auto: true,
                    overflow: true,
                    agentID: lastUser.agentID,
                  })
                  .pipe(Effect.ignore)
              }
              // 场景：模型调了工具（默认路径）→ 把工具结果喂回，再转一圈。
              return "continue" as const
            }

            const [skills, env, instructions] = yield* Effect.all([
              sys.skills(agent),
              sys.environment(model, session.time.created),
              instruction.system().pipe(Effect.orDie),
            ])
            // 展示加载了哪些指令文件（CLAUDE.md、AGENTS.md 等）。
            // 仅对主会话（子 agent 会很吵）且每个会话仅一次。
            if (!session.parentID && !instructionsNotified.has(sessionID)) {
              instructionsNotified.add(sessionID)
              const worktree = (yield* InstanceState.context).worktree
              const files = Array.from(instructions.paths, (p) => Instruction.display(p, worktree))
              if (files.length > 0) {
                yield* bus.publish(TuiEvent.InstructionsLoaded, { files }).pipe(Effect.ignore)
              }
            }
            const additions = [
              ...env,
              ...(skills ? [skills] : []),
              ...instructions.content,
              ...(format.type === "json_schema" ? [STRUCTURED_OUTPUT_SYSTEM_PROMPT] : []),
            ]
            // 注意：`buildLLMRequestPrefix` 也会返回一个 `tools` 字段，但我们这里有意不用它
            // ——来自 `resolveTools` 的 `tools` 变量（前面通过 `handle.process({tools: ...})`
            // 设置）携带了 AI SDK 运行时工具分发所需的 `execute` 闭包，而 `buildLLMRequestPrefix`
            // 产出的是仅含 schema 的工具。两条路径的 schema 字节一致（都以相同参数调用
            // registry.tools），所以前缀缓存的一致性成立。
            // 主 runLoop：无 watermark——LLM 必须看到完整的 msgs 列表，包括本 turn 中间的
            // 助手 turn（工具读取、task 创建等），这样每一步都不会从裸的用户 prompt 重放。
            // watermark 仅用于 fork 捕获（派生时对父级视图的冻结快照）。
            const { system: prebuiltSystem, inheritedMessages: modelMsgs } =
              yield* buildLLMRequestPrefix({
                sessionID,
                agent,
                model,
                msgs,
                additions,
              }).pipe(
                Effect.provideService(LLM.Service, llm),
                Effect.provideService(ToolRegistry.Service, registry),
              )
            const maxModeCfg = (yield* config.get()).experimental?.maxMode
            const useMaxMode =
              agent.name === MaxMode.MAX_MODE_AGENT && maxModeCfg !== undefined && format.type !== "json_schema"

            const processArgs = {
              user: lastUser,
              agent,
              permission: session.permission,
              sessionID,
              parentSessionID: session.parentID,
              // system: additions 为 StreamInput 的非 LLM 消费者保留（例如
              // MessageV2.User.system，用于日志/回放）；llm.stream 本身使用 prebuiltSystem。
              system: additions,
              prebuiltSystem,
              messages: [...modelMsgs, ...(isLastStep ? [{ role: "user" as const, content: MAX_STEPS }] : [])],
              tools,
              model,
              toolChoice: isLastStep ? ("none" as const) : format.type === "json_schema" ? ("required" as const) : undefined,
              agentID: lastUser.agentID,
            }

            const queryParts =
              msgs.findLast((m) => m.info.role === "user" && m.info.id === lastUser.id)?.parts ?? []
            const query = userQueryText(queryParts)
            const preQuery = {
              cancel: undefined as boolean | undefined,
              cancelReason: undefined as string | undefined,
            }
            yield* plugin.trigger(
              "session.userQuery.pre",
              { sessionID, agentID: resolvedAgentID, step, messageID: lastUser.id, query },
              preQuery,
            )
            if (preQuery.cancel) {
              cancelled = true
              cancelReason = preQuery.cancelReason
              handle.message.error = new MessageV2.AbortedError({
                message: preQuery.cancelReason ?? "Step cancelled by plugin",
              }).toObject()
              handle.message.finish = "cancelled"
              yield* sessions.updateMessage(handle.message)
              yield* plugin.trigger(
                "session.userQuery.post",
                {
                  sessionID,
                  agentID: resolvedAgentID,
                  step,
                  messageID: lastUser.id,
                  query,
                  assistantMessageID: handle.message.id,
                  finish: handle.message.finish,
                  error: preQuery.cancelReason,
                  trajectory: trajectoryForStep(msgs, handle.message),
                },
                {},
              )
              return "break" as const
            }

            const stepEffect = useMaxMode
              ? MaxMode.runMaxStep({
                  // runMaxStep 复用与 handle.process 完全相同的每步参数，
                  // 外加它所需的编排句柄。
                  ...processArgs,
                  handle,
                  llm,
                  candidates: maxModeCfg?.candidates,
                  setStatus: (message) =>
                    status.set(sessionID, message ? { type: "busy", message } : { type: "busy" }),
                })
              : handle.process(processArgs)

            const result = yield* stepEffect.pipe(
              Effect.onExit((exit) =>
                plugin
                  .trigger(
                    "session.userQuery.post",
                    {
                      sessionID,
                      agentID: resolvedAgentID,
                      step,
                      messageID: lastUser.id,
                      query,
                      assistantMessageID: handle.message.id,
                      finish: handle.message.finish,
                      error: Exit.isFailure(exit)
                        ? Cause.pretty(exit.cause)
                        : sessionErrorText(handle.message.error),
                      finalText: assistantFinalText(handle.message, MessageV2.parts(handle.message.id)),
                      trajectory: trajectoryForStep(msgs, handle.message),
                    },
                    {},
                  )
                  .pipe(Effect.ignore),
              ),
            )

            // ==== 主 agent 分支：把这一步模型返回归纳成 outcome（break/continue）====
            if (
              result === "continue" &&
              (yield* autoContinueOutputLength({ lastUser, assistant: handle.message }))
            ) {
              // 场景：输出被 token 截断（§2.1），已注入续写提醒 → continue。
              return "continue" as const
            }

            if (result === "text-repeat") {
              // 场景：流内复读（§2.5）。能救 continue，救不动 break。
              if (yield* handleTextRepeat({ lastUser })) return "continue" as const
              return "break" as const
            }

            if (structured !== undefined) {
              // 场景：拿到了要求的结构化输出（§4）→ 存下、收工。
              handle.message.structured = structured
              handle.message.finish = handle.message.finish ?? "stop"
              yield* sessions.updateMessage(handle.message)
              return "break" as const
            }

            // Empty/no-op tool-call loop guard (main branch). Intercept before
            // classify would `continue` an empty tool-calls step: soft-nudge
            // within budget, hard-halt once exceeded. A non-empty step returns
            // "none" and falls through to normal classification.
            const emptyStep = yield* handleEmptyStep({ lastUser, assistant: handle.message })
            if (emptyStep === "halt") return "break" as const // 场景：空调用循环超限（§2.6）→ 硬熔断收工
            if (emptyStep === "continue") return "continue" as const // 场景：空步骤可救，已提醒 → continue

            const classification = classifyAssistantStep({
              phase: "after-process",
              lastUser,
              assistant: handle.message,
              parts: MessageV2.parts(handle.message.id),
              processResult: result,
            })
            if (classification.type === "filtered") {
              // 场景：被内容安全过滤（§4）→ 写错误、收工。
              yield* writeContentFilterError({ assistant: handle.message })
              return "break" as const
            }
            if (classification.type === "failed") {
              // 场景：模型报错（§4）→ 写错误、收工。
              yield* writeModelError({ assistant: handle.message, reason: classification.reason })
              return "break" as const
            }
            if (classification.type === "text-tool-call") {
              // 场景：工具调用写成了纯文本（§2.3）。能重试 continue，用尽 break。
              if (yield* autoRetryTextToolCall({ lastUser, assistant: handle.message })) return "continue" as const
              return "break" as const
            }
            if (classification.type !== "continue" && !handle.message.error && format.type === "json_schema") {
              // 场景：要 json_schema 但没产出结构化输出（§2.4）。能重试 continue，用尽 break。
              if (yield* autoRetryStructuredOutput({ lastUser, assistant: handle.message })) return "continue" as const
              return "break" as const
            }

            if (
              (classification.type === "think-only" || classification.type === "invalid") &&
              format.type !== "json_schema"
            ) {
              // 场景：只思考没答案 / 空输出（§2.2）。能续写 continue，用尽 break。
              const reason = classification.type === "invalid" ? classification.reason : "think-only"
              if (yield* autoContinueInvalidOutput({ lastUser, assistant: handle.message, reason }))
                return "continue" as const
              return "break" as const
            }

            if (classification.type === "final" && classification.degraded)
              yield* slog.warn("degraded final on abnormal finish", { finish: handle.message.finish })
            if (result === "stop") return "break" as const // 场景：模型正常说完（§4）→ 收工
            if (!isBoundedComputation && result === "overflow") {
              // 子 agent 溢出 → 按 actor 的 compaction。插入一个用子 agent 的 agent_id
              // 标记的边界；下一次 runLoop 迭代将看到裁剪后的上下文（filterCompactedEffect
              // 在边界处停止）。
              // 门控必须排除 "main"——见本文件前面相应门控处的注释（约 1716 行）
              // 以及 checkpoint.ts:715。
              if (lastUser.agentID && lastUser.agentID !== "main") {
                yield* compaction
                  .create({
                    sessionID,
                    agent: lastUser.agent,
                    model: { providerID: model.providerID, modelID: model.id },
                    auto: true,
                    overflow: true,
                    agentID: lastUser.agentID,
                  })
                  .pipe(Effect.ignore)
                // 场景：子 agent provider 信号溢出（§3.2）→ 有损压缩后 continue。
                return "continue" as const
              }

              // Main-agent provider-signalled overflow: prefer rebuild over
              // compaction. Shared with the manual `/rebuild` command via
              // rebuildFromCheckpoint (does not block on the writer; uses the
              // on-disk checkpoint). Fall back to compaction only when no
              // boundary can be produced.
              // 主 agent 由 provider 发出信号的溢出:优先 rebuild 而非 compaction。通过共享 helper
              // rebuildFromCheckpoint 与手动 /rebuild 命令复用(不阻塞等待 writer;用磁盘上的 checkpoint)。
              // 只有在无法产生边界时才回退到 compaction。
              const inserted2 = yield* rebuildFromCheckpoint({
                sessionID,
                msgs,
                agentID: lastUser.agentID,
                agent: lastUser.agent,
                model: { providerID: model.providerID, id: model.id },
              })
              // 场景：主 agent 溢出（§3.2），已插入 checkpoint 边界 → 下一圈从 checkpoint 重建。
              if (inserted2) return "continue" as const

              // F39：没有 checkpoint——回退到 compaction（LLM 驱动的有损摘要）。
              yield* compaction
                .create({
                  sessionID,
                  agent: lastUser.agent,
                  model: { providerID: model.providerID, modelID: model.id },
                  auto: true,
                  overflow: true,
                  agentID: lastUser.agentID,
                })
                .pipe(Effect.ignore)
              // （主 agent 溢出但无 checkpoint 可用：已回退到有损压缩，落到下面的 continue）
            }
            // 场景：模型调了工具（默认路径），或溢出已处理完 → 把结果喂回，再转一圈。
            return "continue" as const
          }).pipe(Effect.ensuring(instruction.clear(handle.message.id)))

          // --- 文本循环检测（跨步骤）---
          const completedParts = MessageV2.parts(handle.message.id)
          const stepText = completedParts
            .filter((p): p is MessageV2.TextPart => p.type === "text" && !p.synthetic)
            .map((p) => p.text)
            .join(" ")
          if (stepText.trim()) {
            // 把工具调用签名也纳入 key，这样"相同文本 + 不同工具"不会被判为循环
            const toolSig = completedParts
              .filter((p): p is MessageV2.ToolPart => p.type === "tool")
              .map((p) => `${p.tool}:${JSON.stringify(p.state && "input" in p.state ? p.state.input : "")}`)
              .join("|")
            const normalized = normalizeForLoopDetection(stepText) + (toolSig ? `\0${toolSig}` : "")
            textLoopBuffer.push(normalized)
            if (textLoopBuffer.length > TEXT_LOOP_BUFFER_SIZE) textLoopBuffer.shift()

            if (textLoopBuffer.length >= TEXT_LOOP_TRIGGER_COUNT) {
              const isTextLoop = detectTextLoop(textLoopBuffer, TEXT_LOOP_TRIGGER_COUNT)

              if (isTextLoop) {
                if (textLoopRecoveryAttempts >= TEXT_LOOP_MAX_RECOVERY) {
                  yield* slog.info("text loop: max recovery exceeded, terminating")
                  yield* bus.publish(Session.Event.Error, {
                    sessionID,
                    error: new NamedError.Unknown({
                      message: `Text loop detected: model repeated the same output ${TEXT_LOOP_TRIGGER_COUNT} times after ${TEXT_LOOP_MAX_RECOVERY} recovery attempts. Session terminated.`,
                    }).toObject(),
                  })
                  // 场景：跨步文本循环（§2.7）恢复次数已用尽，模型仍在绕圈 → 发错误、收工。
                  break
                }
                const recoveryText =
                  textLoopRecoveryAttempts === 0 ? RECOVERY_PROMPT_MILD : RECOVERY_PROMPT_STRONG
                // 在对话末尾创建一条*新的*用户消息（而不是追加到原来的那条）
                const reentry = yield* sessions.updateMessage({
                  id: MessageID.ascending(),
                  role: "user" as const,
                  sessionID,
                  agentID: lastUser.agentID,
                  agent: lastUser.agent,
                  model: lastUser.model,
                  tools: lastUser.tools,
                  format: lastUser.format,
                  time: { created: Date.now() },
                })
                yield* sessions.updatePart({
                  id: PartID.ascending(),
                  messageID: reentry.id,
                  sessionID,
                  type: "text",
                  synthetic: true,
                  text: recoveryText,
                } satisfies MessageV2.TextPart)
                textLoopRecoveryAttempts++
                textLoopBuffer.length = 0
                yield* slog.info("text loop: recovery injected", { attempt: textLoopRecoveryAttempts })
                // 场景：跨步文本循环（§2.7），已注入换思路提醒 → 再转一圈让模型改变输出。
                continue
              }
            }
          }

          // ==== 本圈收尾：根据这一步归纳出的 outcome 决定收工还是再转一圈 ====
          if (outcome === "break") {
            // 场景：这一步想收工（模型说完 / 终态 / 各类重试用尽）。
            // A hard halt is terminal — skip the ReAct re-entry gates so a
            // degraded model can't be re-driven into the same empty loop.
            if (hardHalt) break // 硬熔断（§2.6）：直接收工，跳过下面两道否决闸门
            if (yield* taskGate(lastUser)) continue // 还有未完成 task（§5.1）→ 被否决，强制再转一圈
            if (yield* goalGate(lastUser)) continue // goal 未达成（§5.2）→ 被否决，强制再转一圈
            break // 两道闸门都放行 → 真正收工
          }
          // 场景：这一步是 continue（模型调了工具要喂回结果 / 触发了某个重试）→ 再转一圈。
          continue
        }

        const promptOps = yield* ops()
        if (lastModelForPrune && lastFinishedForPrune) {
          yield* prune
            .prune({
              sessionID,
              model: lastModelForPrune,
              tokens: lastFinishedForPrune.tokens,
              lastAssistantTime: lastFinishedForPrune.time.completed,
              promptOps,
            })
            .pipe(Effect.ignore, Effect.forkIn(scope))
        }
        const final = yield* lastAssistant(sessionID, agentID)
        const finalIsError = final.info.role === "assistant" && !!final.info.error
        const lastUserForMetrics = yield* sessions.findMessage(
          sessionID,
          (m) => m.info.role === "user",
          { agentID: "*" },
        )
        yield* publishAgentRequest(
          finalIsError ? "error" : "completed",
          Option.isSome(lastUserForMetrics) ? lastUserForMetrics.value.info.agent : final.info.agent,
        )
        // Woken-peer completion signal. forkWork.notify only wraps the FIRST
        // (spawn) turn; a persistent background peer that finishes a later,
        // inbox-driven turn would otherwise go idle silently and force the
        // orchestrator to poll. When this loop was woken via the inbox path
        // (notifyParentOnComplete), mirror forkWork's actor_notification to the
        // parent so the event-driven model holds. Gated to background peers and
        // excludes system subagents (checkpoint-writer/dream/distill). The flag
        // is never set on the spawn turn, so turn 1 is not double-notified.
        // 唤醒对等执行单元完成信号。forkWork.notify 仅封装首轮（创建实例）执行流程；
        // 若常驻后台对等单元完成后续由消息队列触发的执行流程，若无本段逻辑，它会直接静默进入空闲状态，
        // 迫使调度器持续轮询。当本次循环经由消息队列通路（notifyParentOnComplete）被唤醒时，
        // 向父节点同步发送与 forkWork 一致的 actor_notification 通知，以此保障事件驱动模型正常运行。
        // 该逻辑仅对后台对等单元生效，系统子代理（检查点写入器、推理单元、蒸馏单元）不触发此逻辑。
        // 创建实例首轮流程不会置位该标记，因此首轮执行不会重复发送通知。
        if (notifyParentOnComplete && agentID && session.parentID) {
          const actor = yield* actorRegistry.get(sessionID, agentID)
          if (
            actor &&
            actor.mode === "peer" &&
            actor.background &&
            !SYSTEM_SPAWNED_AGENT_TYPES.has(actor.agent)
          ) {
            const finalText =
              final.info.role === "assistant" ? assistantFinalText(final.info, final.parts) : undefined
            const parsed = parseReturnHeader(finalText)
            const status = finalIsError ? "failed" : "completed"
            yield* inbox
              .send({
                receiverSessionID: session.parentID,
                receiverActorID: actor.parentActorID ?? "main",
                senderSessionID: sessionID,
                senderActorID: agentID,
                type: "actor_notification",
                content: renderActorNotification({
                  actorID: agentID,
                  description: actor.description,
                  status,
                  ...(status === "completed"
                    ? {
                        result: finalText ?? "(no output)",
                        ...(parsed.status ? { reportedStatus: parsed.status } : {}),
                        ...(parsed.summary ? { reportedSummary: parsed.summary } : {}),
                      }
                    : { error: final.info.role === "assistant" ? sessionErrorText(final.info.error) : "unknown" }),
                }),
              })
              .pipe(Effect.ignore)
          }
        }
        return final
        }).pipe(Effect.onExit(firePostSession), Effect.orDie)
      },
    )

    const loop: (input: z.infer<typeof LoopInput>) => Effect.Effect<MessageV2.WithParts> = Effect.fn(
      "SessionPrompt.loop",
    )(function* (input: z.infer<typeof LoopInput>) {
      const agentID = input.agentID ?? "main"
      return yield* state.ensureRunning(
        input.sessionID,
        agentID,
        lastAssistant(input.sessionID, agentID),
        runLoop(input.sessionID, agentID, input.task_id, input.notifyParentOnComplete),
      )
    })

    const shell: (input: ShellInput) => Effect.Effect<MessageV2.WithParts> = Effect.fn("SessionPrompt.shell")(
      function* (input: ShellInput) {
        return yield* state.startShell(input.sessionID, lastAssistant(input.sessionID), shellImpl(input))
      },
    )

    const command = Effect.fn("SessionPrompt.command")(function* (input: CommandInput) {
      yield* elog.info("command", { sessionID: input.sessionID, command: input.command, agent: input.agent })
      const cmd = yield* commands.get(input.command)
      if (!cmd) {
        const available = (yield* commands.list()).map((c) => c.name)
        const hint = available.length ? ` Available commands: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Command not found: "${input.command}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }
      const agentName = cmd.agent ?? input.agent ?? (yield* agents.defaultAgent())

      // /goal —— 设置或清除一个会话级的停止条件 goal。条件文本本身成为本 turn 的
      // prompt（工作 agent 会立即开始追求它）；随后主 runLoop 会拒绝停止，直到裁判
      // 判定它已满足。见 session/goal.ts。
      if (input.command === Command.Default.GOAL) {
        const condition = input.arguments.trim()
        if (condition === "" || condition === "clear" || condition === "reset") {
          yield* goal.clear(input.sessionID)
          return yield* prompt({
            sessionID: input.sessionID,
            messageID: input.messageID,
            agent: agentName,
            parts: [{ type: "text", text: "Goal cleared.", synthetic: true }],
            noReply: true,
          })
        }
        yield* goal.set(input.sessionID, condition)
      }

      // /rebuild — manually rebuild the conversation context now, from the
      // latest checkpoint. Reuses the SAME rebuildFromCheckpoint step as the
      // automatic overflow path (identical logic + boundary conditions), so a
      // user-triggered rebuild behaves exactly like an auto one: it inserts a
      // checkpoint boundary at the watermark (recent messages after it are kept
      // verbatim; earlier ones collapse to the checkpoint summary on the next
      // turn). If no usable checkpoint exists yet, tell the user rather than
      // silently doing nothing — the first checkpoint has to be produced by
      // normal turns before there is anything to rebuild from.
      // /rebuild 命令——基于最新检查点，手动重建当前对话上下文。
      // 该逻辑复用与自动溢出流程完全一致的 rebuildFromCheckpoint 处理步骤（逻辑、边界条件均相同），
      // 因此用户手动触发重建的效果与自动重建完全一致：会在水位标记处插入检查点分界，
      // 分界之后的近期消息完整保留，分界之前的历史消息将在下一轮执行时压缩为检查点摘要。
      // 若当前不存在可用的检查点，会向用户返回提示，而非静默无响应；
      // 首个检查点必须经由常规对话轮次生成后，才具备重建的数据源。
      if (input.command === Command.Default.REBUILD) {
        const msgs = yield* sessions.messages({ sessionID: input.sessionID, agentID: "main" })
        const lastUser = msgs.findLast((m) => m.info.role === "user")
        const model = yield* lastModel(input.sessionID)
        const inserted = yield* rebuildFromCheckpoint({
          sessionID: input.sessionID,
          msgs,
          agentID: lastUser?.info.agentID ?? "main",
          agent: agentName,
          model: { providerID: model.providerID, id: model.modelID },
        }).pipe(Effect.catch(() => Effect.succeed(false)))
        return yield* prompt({
          sessionID: input.sessionID,
          messageID: input.messageID,
          agent: agentName,
          parts: [
            {
              type: "text",
              text: inserted
                ? "Context rebuilt from the latest checkpoint. Recent messages are preserved; earlier context is now summarized."
                : "No checkpoint is available to rebuild from yet — continue the conversation and a checkpoint will be written automatically.",
              synthetic: true,
            },
          ],
          noReply: true,
        })
      }

      const raw = input.arguments.match(argsRegex) ?? []
      const args = raw.map((arg) => arg.replace(quoteTrimRegex, ""))
      const templateCommand = yield* Effect.promise(async () => cmd.template)

      let template: string
      if (cmd.source === "skill") {
        template = input.arguments
      } else {
        const placeholders = templateCommand.match(placeholderRegex) ?? []
        let last = 0
        for (const item of placeholders) {
          const value = Number(item.slice(1))
          if (value > last) last = value
        }

        const withArgs = templateCommand.replaceAll(placeholderRegex, (_, index) => {
          const position = Number(index)
          const argIndex = position - 1
          if (argIndex >= args.length) return ""
          if (position === last) return args.slice(argIndex).join(" ")
          return args[argIndex]
        })
        const usesArgumentsPlaceholder = templateCommand.includes("$ARGUMENTS")
        template = withArgs.replaceAll("$ARGUMENTS", input.arguments)

        if (placeholders.length === 0 && !usesArgumentsPlaceholder && input.arguments.trim()) {
          template = template + "\n\n" + input.arguments
        }
      }

      const shellMatches = ConfigMarkdown.shell(template)
      if (shellMatches.length > 0) {
        const sh = Shell.preferred()
        const results = yield* Effect.promise(() =>
          Promise.all(
            shellMatches.map(async ([, cmd]) => (await Process.text([cmd], { shell: sh, nothrow: true })).text),
          ),
        )
        let index = 0
        template = template.replace(bashRegex, () => results[index++])
      }
      template = template.trim()

      const taskModel = yield* Effect.gen(function* () {
        if (cmd.model) return Provider.parseModel(cmd.model)
        if (cmd.agent) {
          const cmdAgent = yield* agents.get(cmd.agent)
          if (cmdAgent?.model) return cmdAgent.model
        }
        if (input.model) return Provider.parseModel(input.model)
        return yield* lastModel(input.sessionID)
      })

      yield* getModel(taskModel.providerID, taskModel.modelID, input.sessionID)

      const agent = yield* agents.get(agentName)
      if (!agent) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${agentName}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }

      const templateParts = yield* resolvePromptParts(template)
      const isSubtask = (agent.mode === "subagent" && cmd.subtask !== false) || cmd.subtask === true

      let parts: PromptInput["parts"]
      if (isSubtask) {
        const promptText = cmd.source === "skill"
          ? templateCommand + (input.arguments.trim() ? "\n\n" + input.arguments : "")
          : (templateParts.find((y): y is typeof y & { type: "text"; text: string } => y.type === "text"))?.text ?? ""
        parts = [
          {
            type: "subtask" as const,
            agent: agent.name,
            description: cmd.description ?? "",
            command: input.command,
            model: { providerID: taskModel.providerID, modelID: taskModel.modelID },
            prompt: promptText,
          },
        ]
      } else if (cmd.source === "skill") {
        const visibleText = input.arguments.trim()
          ? `/${input.command} ${input.arguments}`
          : `/${input.command}`
        const skillPart = {
          type: "text" as const,
          text: `<skill_content name="${input.command}">\n${templateCommand}\n</skill_content>`,
          synthetic: true,
        }
        const attachments = templateParts.filter((p): p is Exclude<typeof p, { type: "text" }> => p.type !== "text")
        parts = [{ type: "text" as const, text: visibleText }, skillPart, ...attachments, ...(input.parts ?? [])]
      } else {
        parts = [...templateParts, ...(input.parts ?? [])]
      }

      const userAgent = isSubtask ? (input.agent ?? (yield* agents.defaultAgent())) : agentName
      const userModel = isSubtask
        ? input.model
          ? Provider.parseModel(input.model)
          : yield* lastModel(input.sessionID)
        : taskModel

      yield* plugin.trigger(
        "command.execute.before",
        { command: input.command, sessionID: input.sessionID, arguments: input.arguments },
        { parts },
      )

      const result = yield* prompt({
        sessionID: input.sessionID,
        messageID: input.messageID,
        model: userModel,
        agent: userAgent,
        parts,
        variant: input.variant,
      })
      yield* bus.publish(Command.Event.Executed, {
        name: input.command,
        sessionID: input.sessionID,
        arguments: input.arguments,
        messageID: result.info.id,
      })
      return result
    })

    const impl = Service.of({
      cancel,
      prompt,
      loop,
      shell,
      command,
      resolvePromptParts,
      sweepOrphanAssistants,
      predict,
    })
    sessionPromptRef.current = { loop: impl.loop }
    // Expose the project default-model resolver to Inbox.drain's option-2
    // fallback (seed a synthetic message for a turnCount-0 standing peer whose
    // slice has no model-bearing message yet). Reads Provider, which is already
    // in scope here — Inbox.layer stays free of a Provider dependency.
    const defaultModelResolver = { defaultModel: () => provider.defaultModel() }
    defaultModelRef.current = defaultModelResolver
    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        if (sessionPromptRef.current?.loop === impl.loop) sessionPromptRef.current = undefined
        if (defaultModelRef.current === defaultModelResolver) defaultModelRef.current = undefined
      }),
    )
    return impl
  }),
)

export const defaultLayer = Layer.suspend(() =>
  layer.pipe(
    Layer.provide(SessionRunState.defaultLayer),
    Layer.provide(SessionStatus.defaultLayer),
    Layer.provide(SessionPrune.defaultLayer),
    Layer.provide(SessionCheckpoint.defaultLayer),
    Layer.provide(SessionCompaction.defaultLayer),
    Layer.provide(SessionProcessor.defaultLayer),
    Layer.provide(Command.defaultLayer),
    Layer.provide(Permission.defaultLayer),
    Layer.provide(MCP.defaultLayer),
    Layer.provide(LSP.defaultLayer),
    Layer.provide(ToolRegistry.defaultLayer),
    Layer.provide(Truncate.defaultLayer),
    Layer.provide(Provider.defaultLayer),
    Layer.provide(Instruction.defaultLayer),
    Layer.provide(AppFileSystem.defaultLayer),
    Layer.provide(Plugin.defaultLayer),
    Layer.provide(Session.defaultLayer),
    Layer.provide(SessionRevert.defaultLayer),
    Layer.provide(
      Layer.mergeAll(
        Config.defaultLayer,
        SessionSummary.defaultLayer,
        Team.defaultLayer,
        ActorRegistry.defaultLayer,
        Agent.defaultLayer,
        SystemPrompt.defaultLayer,
        LLM.defaultLayer,
        Bus.layer,
        CrossSpawnSpawner.defaultLayer,
        Inbox.defaultLayer,
        Goal.defaultLayer,
        TaskGateState.defaultLayer,
        TaskRegistry.defaultLayer,
      ),
    ),
  ),
)
export const PromptInput = z.object({
  sessionID: SessionID.zod,
  messageID: MessageID.zod.optional(),
  model: z
    .object({
      providerID: ProviderID.zod,
      modelID: ModelID.zod,
    })
    .optional(),
  modelRef: z
    .string()
    .optional()
    .describe(
      "Model group/tier name (e.g. ultra/standard/lite) or a literal provider/model. Resolved provider-aware. Takes precedence over `model` when both are set.",
    ),
  agent: z.string().optional(),
  agentID: z.string().optional(),
  task_id: z.string().optional()
    .describe("If the spawning caller bound this prompt to a specific user-task (T4 etc), pass its TID. Propagates to Tool.Context.taskId so memory-path-guard allows writes to tasks/<task_id>/*.md."),
  source: z.enum(["user", "spawn", "hook"]).optional(),
  provenance: MessageV2.Provenance.optional(),
  noReply: z.boolean().optional(),
  tools: z
    .record(z.string(), z.boolean())
    .optional()
    .describe("@deprecated tools and permissions have been merged, you can set permissions on the session itself now"),
  format: MessageV2.Format.optional(),
  system: z.string().optional(),
  variant: z.string().optional(),
  parts: z.array(
    z.discriminatedUnion("type", [
      MessageV2.TextPart.omit({
        messageID: true,
        sessionID: true,
      })
        .partial({
          id: true,
        })
        .meta({
          ref: "TextPartInput",
        }),
      MessageV2.FilePart.omit({
        messageID: true,
        sessionID: true,
      })
        .partial({
          id: true,
        })
        .meta({
          ref: "FilePartInput",
        }),
      MessageV2.AgentPart.omit({
        messageID: true,
        sessionID: true,
      })
        .partial({
          id: true,
        })
        .meta({
          ref: "AgentPartInput",
        }),
      MessageV2.SubtaskPart.omit({
        messageID: true,
        sessionID: true,
      })
        .partial({
          id: true,
        })
        .meta({
          ref: "SubtaskPartInput",
        }),
    ]),
  ),
})
export type PromptInput = z.infer<typeof PromptInput>

export const LoopInput = z.object({
  sessionID: SessionID.zod,
  agentID: z.string().optional(),
  task_id: z.string().optional(),
  // Set by the inbox wake path so a persistent background peer that finishes a
  // woken turn notifies its parent (mirroring forkWork.notify, which only wraps
  // the FIRST/spawn turn). Left false on spawn/user-driven loops to avoid
  // double-notifying the spawn turn that forkWork already covers.
  notifyParentOnComplete: z.boolean().optional(),
})

export const ShellInput = z.object({
  sessionID: SessionID.zod,
  messageID: MessageID.zod.optional(),
  agent: z.string(),
  model: z
    .object({
      providerID: ProviderID.zod,
      modelID: ModelID.zod,
    })
    .optional(),
  modelRef: z
    .string()
    .optional()
    .describe(
      "Model group/tier name (e.g. ultra/standard/lite) or a literal provider/model. Resolved provider-aware. Takes precedence over `model` when both are set.",
    ),
  command: z.string(),
})
export type ShellInput = z.infer<typeof ShellInput>

export const CommandInput = z.object({
  messageID: MessageID.zod.optional(),
  sessionID: SessionID.zod,
  agent: z.string().optional(),
  model: z.string().optional(),
  arguments: z.string(),
  command: z.string(),
  variant: z.string().optional(),
  parts: z
    .array(
      z.discriminatedUnion("type", [
        MessageV2.FilePart.omit({
          messageID: true,
          sessionID: true,
        }).partial({
          id: true,
        }),
      ]),
    )
    .optional(),
})
export type CommandInput = z.infer<typeof CommandInput>

/** @internal 导出仅供测试使用 */
export function createStructuredOutputTool(input: {
  schema: Record<string, any>
  onSuccess: (output: unknown) => void
}): AITool {
  // 如果存在 $schema 属性就移除它（工具输入不需要它）
  const { $schema: _, ...toolSchema } = input.schema

  return tool({
    description: STRUCTURED_OUTPUT_DESCRIPTION,
    inputSchema: jsonSchema(toolSchema as JSONSchema7),
    async execute(args) {
      // AI SDK 在调用 execute() 之前会校验 args 是否符合 inputSchema
      input.onSuccess(args)
      return {
        output: "Structured output captured successfully.",
        title: "Structured Output",
        metadata: { valid: true },
      }
    },
    toModelOutput({ output }) {
      return {
        type: "text",
        value: output.output,
      }
    },
  })
}
const bashRegex = /!`([^`]+)`/g
// 把 [Image N] 匹配为单个 token、带引号的字符串，或非空白字符序列
const argsRegex = /(?:\[Image\s+\d+\]|"[^"]*"|'[^']*'|[^\s"']+)/gi
const placeholderRegex = /\$(\d+)/g
const quoteTrimRegex = /^["']|["']$/g

/**
 * 定时 prompt 的触发接缝（T18，规范 [S5]）。
 *
 * 把一次 cron/loop 触发经由与打字输入的用户 prompt *相同*的入口漏斗式送入：
 * `SessionPrompt.Service.prompt`。合成 part 带有 `synthetic: true`（mimocode 中
 * `isMeta` 的约定），使对话预览界面可以隐藏它；并带有
 * `metadata.origin = { kind: "cron", taskId, kindOfTask }`，使 TUI 能渲染一个时钟图标。
 * 这里*有意*不做 sentinel 展开——T19 会在本调用之前包裹 `value`。
 */
export type ScheduledPromptOrigin = {
  kind: "cron"
  taskId: string
  kindOfTask: "cron" | "loop"
  /**
   * 调度器 tick 触发本任务时的 ISO-8601 时间戳。由 cron bridge 在 `onFire` 中设置；
   * 持久化在合成 part 的 metadata 上，使 TUI 及下游消费者无需解析前置的文本前缀
   * 即可恢复触发时间。
   */
  firedAt?: string
}

export type InjectScheduledPromptInput = {
  sessionID: SessionID
  value: string
  origin: ScheduledPromptOrigin
  priority?: "later" | "next" | "now"
  isMeta?: boolean
}

export const injectScheduledPrompt = (input: InjectScheduledPromptInput) =>
  Effect.gen(function* () {
    const sp = yield* Service
    yield* Effect.asVoid(
      sp.prompt({
        sessionID: input.sessionID,
        source: "hook",
        parts: [
          {
            type: "text",
            text: input.value,
            synthetic: input.isMeta ?? true,
            metadata: {
              origin: input.origin,
              priority: input.priority ?? "later",
            },
          },
        ],
      }),
    )
  })

export * as SessionPrompt from "./prompt"
