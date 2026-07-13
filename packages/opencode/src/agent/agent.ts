// ============================================================================
// 【文件导读】agent.ts —— Agent（智能体）注册中心
// ----------------------------------------------------------------------------
// 这个文件不是"一个 agent"，而是管理"一整册 agent"的服务。核心分四块：
//   1. Info      —— 单个 agent 的数据结构（名字/模式/权限/提示词…）
//   2. Interface —— 这个服务对外暴露的能力（get / list / defaultAgent / generate）
//   3. Service   —— Effect 框架的依赖注入"令牌"，别的模块靠它拿到本服务
//   4. layer     —— 真正的实现：内部构建出所有 agent，并实现上面 4 个方法
// 阅读顺序建议：Info → Interface → Service → layer 内部的 agents 对象 → 4 个方法。
// ============================================================================

import { Config } from "../config"
import { Flag } from "@/flag/flag"
import z from "zod"
import { Provider } from "../provider"
import { ModelID, ProviderID } from "../provider/schema"
import { generateObject, streamObject, type ModelMessage } from "ai"
import { Instance } from "../project/instance"
import { Truncate } from "../tool"
import { Auth } from "../auth"
import { ProviderTransform } from "../provider"

import PROMPT_GENERATE from "./generate.txt"
import PROMPT_EXPLORE from "./prompt/explore.txt"
import PROMPT_DREAM from "./prompt/dream.txt"
import PROMPT_DISTILL from "./prompt/distill.txt"
import PROMPT_SUMMARY from "./prompt/summary.txt"
import PROMPT_COMPACTION from "./prompt/compaction.txt"
import PROMPT_TITLE from "./prompt/title.txt"
import PROMPT_ORCHESTRATOR from "../session/prompt/orchestrator.txt"
import { Permission } from "@/permission"
import { mergeDeep, pipe, sortBy, values } from "remeda"
import { Global } from "@/global"
import path from "path"
import { Plugin } from "@/plugin"
import { Skill } from "../skill"
import { Effect, Context, Layer } from "effect"
import { InstanceState } from "@/effect"
import * as Option from "effect/Option"
import * as OtelTracer from "@effect/opentelemetry/Tracer"

// 【Info】单个 agent 的"档案"。用 zod 定义 = 既是 TS 类型，又能做运行时校验。
// 下面每一个字段就是描述一个 agent 的一个维度。
export const Info = z
  .object({
    name: z.string(), // agent 的唯一标识，如 "build" / "plan" / "explore"
    description: z.string().optional(), // 描述：给用户看，也给"上级 agent"判断何时调用
    // mode 决定这个 agent 的用途：
    //   primary  = 主 agent，用户可直接选/切换（build、plan、compose…）
    //   subagent = 子 agent，只能被主 agent 派生调用（explore、general…）
    //   all      = 两种场景都可用
    mode: z.enum(["subagent", "primary", "all"]),
    native: z.boolean().optional(), // true=代码内置；false=用户在配置里自定义的
    hidden: z.boolean().optional(), // true=不在 UI 里显示（如 title/summary 这种内部 agent）
    topP: z.number().optional(), // 采样参数，透传给模型
    temperature: z.number().optional(), // 温度，透传给模型
    color: z.string().optional(), // UI 里显示用的主题色
    permission: Permission.Ruleset.zod, // 权限规则集：这个 agent 能用哪些工具/能碰哪些目录
    // 不可被覆盖的规则。运行时求值时，它会在"用户/会话权限"之后再追加一次
    //（见 runtimePermission），用于表达"配置绝不能放宽"的 agent 硬性约束，
    // 例如 plan 模式对 edit/write 的禁止。
    hardPermission: Permission.Ruleset.zod.optional(),
    // 指定这个 agent 用哪个模型（完整写死 provider + model）
    model: z
      .object({
        modelID: ModelID.zod,
        providerID: ProviderID.zod,
      })
      .optional(),
    modelRef: z.string().optional(), // 模型的"别名引用"（配置里用简写指向某个模型）
    variant: z.string().optional(), // 模型变体
    prompt: z.string().optional(), // 这个 agent 的系统提示词（system prompt）
    options: z.record(z.string(), z.any()), // 其它自定义选项，透传
    steps: z.number().int().positive().optional(), // 单次运行允许的最大步数
    toolAllowlist: z.array(z.string()).optional(), // 工具白名单：只允许用列表里的工具
  })
  .meta({
    ref: "Agent",
  })
// 从 zod schema 反推出 TS 类型，这样 Info 既是"校验器"又是"类型"，两者不会脱节
export type Info = z.infer<typeof Info>

// 【Interface】本服务对外暴露的能力清单（"能做什么"，不含实现）。
// 返回值都是 Effect.Effect<...>，是 Effect 框架里"待执行的副作用描述"，
// 可以先理解成"异步操作"（类似 Promise，但更强，可组合、可注入依赖）。
export interface Interface {
  readonly get: (agent: string) => Effect.Effect<Info> // 按名字取一个 agent
  readonly list: () => Effect.Effect<Info[]> // 列出全部 agent（已排序）
  readonly defaultAgent: () => Effect.Effect<string> // 返回默认主 agent 的名字
  // 用 LLM 根据一句自然语言描述，"生成"一份新 agent 的配置草稿
  readonly generate: (input: {
    description: string
    model?: { providerID: ProviderID; modelID: ModelID }
  }) => Effect.Effect<{
    identifier: string
    whenToUse: string
    systemPrompt: string
  }>
}

// State = Interface 去掉 generate 的部分。因为 get/list/defaultAgent 都依赖
// "每个实例(工作区)独立的一份 agents 数据"，被放进 InstanceState 里按实例缓存；
// 而 generate 是纯粹调 LLM、不依赖那份缓存，所以留在外层单独实现。
type State = Omit<Interface, "generate">

// 【Service】Effect 的依赖注入令牌。字符串 "@opencode/Agent" 是它的唯一 key。
// 别的模块写 `yield* Agent.Service` 就能拿到一个实现了 Interface 的实例。
export class Service extends Context.Service<Service, Interface>()("@opencode/Agent") {}

// 把 agent 的 permission 与"用户/会话规则集"合并，然后再把 agent 的 hardPermission
// 追加到最后，从而让这些硬性约束胜过用户或会话审批可能引入的任何 allow 规则。
// 所有做权限求值的地方都走这个函数——不存在按 agent 名字做特判的逻辑。
export function runtimePermission(agent: Info, permission?: Permission.Ruleset) {
  return Permission.merge(agent.permission, permission ?? [], agent.hardPermission ?? [])
}

// 【layer】Service 的具体实现（"配方"）。Layer.effect(Service, ...) 的意思是：
// "我要构建 Service，构建过程写在下面的 Effect.gen 里"。
export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    // 先从容器里取出本服务依赖的 5 个上游服务（yield* = 等它们就绪并拿到实例）。
    // 这 5 个依赖具体由谁提供，见文件最底部的 defaultLayer。
    const config = yield* Config.Service // 读用户配置
    const auth = yield* Auth.Service // 读鉴权信息（generate 里判断 openai oauth 用）
    const plugin = yield* Plugin.Service // 插件系统（可改写系统提示词）
    const skill = yield* Skill.Service // 技能系统（提供 skill 目录）
    const provider = yield* Provider.Service // 模型提供方（解析/选择模型）

    // InstanceState.make：为"每个工作区实例"惰性构建并缓存一份 State。
    // 也就是说 agents 这份数据是按实例隔离的，下面这个 function* 就是"如何构建这份数据"。
    const state = yield* InstanceState.make<State>(
      Effect.fn("Agent.state")(function* (_ctx) {
        const cfg = yield* config.get() // 当前实例的配置
        const skillDirs = yield* skill.dirs() // 技能所在目录
        // 白名单目录：这些目录即便在"工作区外"也默认允许访问（截断输出目录 + 各技能目录）
        const whitelistedDirs = [Truncate.GLOB, ...skillDirs.map((dir) => path.join(dir, "*"))]

        // 【defaults】所有 agent 的"默认权限底座"。后面每个 agent 都在它之上叠加自己的规则。
        //   "*": "allow"      —— 默认所有工具都放行
        //   doom_loop: "ask"  —— 疑似死循环时要询问
        //   external_directory —— 工作区外的目录默认要问，白名单目录直接放行
        //   question: "deny"  —— 默认不允许向用户提问（主 agent 会各自改成 allow）
        //   read: {...}       —— .env 类文件要询问，避免泄露密钥
        const defaults = Permission.fromConfig({
          "*": "allow",
          doom_loop: "ask",
          external_directory: {
            "*": "ask",
            ...Object.fromEntries(whitelistedDirs.map((dir) => [dir, "allow"])),
          },
          question: "deny",
          // 对齐 github.com/github/gitignore 里 Node.gitignore 针对 .env 文件的模式
          read: {
            "*": "allow",
            "*.env": "ask",
            "*.env.*": "ask",
            "*.env.example": "allow",
          },
        })

        // 用户在配置里写的权限。会叠加到每个 agent 上，让用户能收紧/放宽默认权限。
        const user = Permission.fromConfig(cfg.permission ?? {})

        // 【agents】核心注册表：key=agent 名字，value=该 agent 的 Info。
        // 这里就是"这个项目到底有哪些 agent"的答案。分主 agent 和子 agent 两类。
        // 每个 agent 的 permission 都是 Permission.merge(defaults, 自己的规则, user)：
        // 越靠后优先级越高，所以用户配置能覆盖内置默认。
        const agents: Record<string, Info> = {
          // ── 主 agent：build（默认执行者，按配置权限跑工具）
          build: {
            name: "build",
            color: "#fb8147",
            description: "Executes tools based on configured permissions.",
            options: {},
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                question: "allow",
              }),
              user,
            ),
            mode: "primary",
            native: true,
          },
          // Max 模式是实验性的、需显式开启：仅当配置了 `experimental.maxMode` 时才注册。
          // 这样在该功能关闭时，默认 agent 集合仍保持为 {build, plan, compose}。
          ...(cfg.experimental?.maxMode
            ? {
                max: {
                  name: "max",
                  color: "#e85d75",
                  description:
                    "Max mode (experimental). Runs N parallel reasoning candidates each step and executes the best one. Same permissions as build.",
                  options: {},
                  permission: Permission.merge(
                    defaults,
                    Permission.fromConfig({
                      question: "allow",
                    }),
                    user,
                  ),
                  mode: "primary" as const,
                  native: true,
                },
              }
            : {}),
          // ── 主 agent：plan（只读"计划模式"，禁止一切编辑，靠 hardPermission 强约束）
          plan: {
            name: "plan",
            color: "#c7e2a8",
            description: "Plan mode. Disallows all edit tools.",
            options: {},
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                question: "allow",
                external_directory: {
                  [path.join(Global.Path.data, "plans", "*")]: "allow",
                },
              }),
              user,
            ),
            // plan 模式唯一的硬性不变式：禁止写入"非计划文件"，且用户/会话配置绝不能放宽它。
            // 通过 runtimePermission 在用户合并之后再追加，因此它永远胜出。
            //（每个写入工具——write/edit/multiedit/apply_patch/notebook_edit——都汇聚到
            // ctx.ask({ permission: "edit" })，所以这一条规则就能管住所有文件写入。）
            // 刻意只作用于 edit：bash/change_directory/workflow 交给模型自身的只读自律
            // 加上 plan 提示词来约束，符合本项目"信任模型、权限层只作兜底"的立场。
            // "*":"deny" 里带有一个非 "*" 的 allow 例外，因此 edit 工具仍留在 schema 中
            //（切换模式时不会改动工具列表，见 PR #1207）。
            hardPermission: Permission.fromConfig({
              edit: {
                "*": "deny",
                [path.join(".mimocode", "plans", "*.md")]: "allow",
                [path.relative(Instance.worktree, path.join(Global.Path.data, path.join("plans", "*.md")))]: "allow",
              },
            }),
            mode: "primary",
            native: true,
          },
          // ── 主 agent：compose（编排模式，带内置 compose 技能来驱动工作流）
          compose: {
            name: "compose",
            color: "#a7a3d8",
            description: "Compose mode. Orchestrates workflows with built-in compose skills.",
            options: {},
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                question: "allow",
              }),
              user,
            ),
            mode: "primary",
            native: true,
          },
          // Orchestrator（编排者）模式是实验性的、需显式开启（默认关闭）：仅当设置了
          // MIMOCODE_EXPERIMENTAL_ORCHESTRATOR 时才注册。在这里做注册门禁，会让它从 TUI
          // 的模式循环、agent 选择弹窗、defaultAgent 中消失，并阻止任何 `session` 工具的
          // 同级派生——从而在关闭时，orchestrator 功能的其余部分都成为不会执行的死代码。
          ...(Flag.MIMOCODE_EXPERIMENTAL_ORCHESTRATOR
            ? {
                orchestrator: {
                  name: "orchestrator",
                  color: "#7fb3d5",
                  description:
                    "Orchestrator mode. A general-purpose coordinator that accomplishes goals by delegating work to child sessions; use the `session` tool to create/switch/list/cancel children running in their own mode and model.",
                  prompt: PROMPT_ORCHESTRATOR,
                  options: {},
                  permission: Permission.merge(
                    defaults,
                    Permission.fromConfig({
                      question: "allow",
                    }),
                    user,
                  ),
                  mode: "primary" as const,
                  native: true,
                },
              }
            : {}),
          // ── 子 agent：general（通用型，被主 agent 派生，用来并行执行多步任务）
          general: {
            name: "general",
            color: "#aac4e1",
            description: `General-purpose agent for researching complex questions and executing multi-step tasks. Use this agent to execute multiple units of work in parallel.`,
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                change_directory: "deny",
              }),
              user,
            ),
            options: {},
            mode: "subagent",
            native: true,
          },
          // ── 子 agent：explore（只读，专门快速探索代码库：grep/glob/read 等，禁止改动）
          explore: {
            name: "explore",
            color: "#f5c9b0",
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                "*": "deny",
                grep: "allow",
                glob: "allow",
                list: "allow",
                bash: "allow",
                webfetch: "allow",
                websearch: "allow",
                codesearch: "allow",
                read: "allow",
                external_directory: {
                  "*": "ask",
                  ...Object.fromEntries(whitelistedDirs.map((dir) => [dir, "allow"])),
                },
              }),
              user,
            ),
            description: `Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. "src/components/**/*.tsx"), search code for keywords (eg. "API endpoints"), or answer questions about the codebase (eg. "how do API endpoints work?"). When calling this agent, specify the desired thoroughness level: "quick" for basic searches, "medium" for moderate exploration, or "very thorough" for comprehensive analysis across multiple locations and naming conventions.`,
            prompt: PROMPT_EXPLORE,
            options: {},
            mode: "subagent",
            native: true,
          },
          // ── 以下都是"隐藏的内部子 agent"(hidden:true)，用户看不到，系统内部自动调用：
          //    title=生成会话标题，summary=生成摘要，compaction=压缩上下文，
          //    checkpoint-writer=复用父级 prefix 缓存的 fork agent，
          //    dream/distill=记忆整理。它们大多 toolAllowlist:[] 或严格限制工具。
          title: {
            name: "title",
            mode: "subagent",
            options: {},
            native: true,
            hidden: true,
            temperature: 0.5,
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                "*": "deny",
              }),
              user,
            ),
            prompt: PROMPT_TITLE,
            toolAllowlist: [],
          },
          summary: {
            name: "summary",
            mode: "subagent",
            options: {},
            native: true,
            hidden: true,
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                "*": "deny",
              }),
              user,
            ),
            prompt: PROMPT_SUMMARY,
            toolAllowlist: [],
          },
          compaction: {
            name: "compaction",
            mode: "subagent",
            options: {},
            native: true,
            hidden: true,
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                "*": "deny",
              }),
              user,
            ),
            prompt: PROMPT_COMPACTION,
            toolAllowlist: [],
          },
          "checkpoint-writer": {
            name: "checkpoint-writer",
            mode: "subagent" as const,
            options: {},
            native: true,
            hidden: true,
            // 没有 `prompt` 字段 —— fork（分叉）agent 的约定：在派生时，
            // tryStartCheckpointWriter 会把父级完整的 LLM 请求前缀
            //（system + tools + 到 watermark 为止的消息）捕获进一个冻结的 ForkContext，
            // 存放在 Actor 服务的内存映射里。fork 的 runLoop 从这份快照读取，
            // 而不是根据本 agent 的身份重新计算。
            // 详见 docs/superpowers/specs/2026-05-26-fork-agent-prefix-cache-design.md
            //
            // 没有 `toolAllowlist` 字段 —— fork agent 必须与父级的工具 schema 保持一致，
            // 以对齐 prefix 缓存。运行时的工具限制通过 actor.tools 白名单来强制执行
            //（在 tryStartCheckpointWriter 中设置）。
            // 权限只继承 `defaults`(+ user) —— 没有专门定制的规则块。
            // 运行时，fork 对 LLM 可见的工具 schema 会按"父 agent"的权限过滤
            //（ForkContext.parentPermission，在 prompt.ts 的 fork 分支喂给 handle.process），
            // 因此与父级一致（保证 prompt 缓存对齐）。注意：每次调用的 ctx.ask 仍会求值
            // 本 agent 自己的权限，但它已被 actor.tools 白名单（在 tryStartCheckpointWriter
            // 中设置）和 memory 路径守卫——真正的写入权限来源——所限制，所以继承 `defaults`
            // 在实际中并不会多授予任何权限。
            // memory 写入会跳过 edit 询问（askEditUnlessMemory），任何无法回答的询问都会
            // 干净地失败（SYSTEM_SPAWNED_AGENT_TYPES → interactive:false）。详见
            // docs/superpowers/specs/2026-06-05-checkpoint-writer-permission-deadlock-design.md
            permission: Permission.merge(defaults, user),
          },
          dream: {
            name: "dream",
            mode: "subagent" as const,
            options: {},
            native: true,
            hidden: true,
            prompt: PROMPT_DREAM,
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                "*": "deny",
                read: "allow",
                write: "allow",
                edit: "allow",
                glob: "allow",
                grep: "allow",
                memory: "allow",
                bash: "allow",
                external_directory: {
                  [path.join(Global.Path.data, "memory")]: "allow",
                  [path.join(Global.Path.data, "memory", "*")]: "allow",
                },
              }),
              user,
            ),
            toolAllowlist: ["read", "write", "edit", "glob", "grep", "memory", "bash"],
          },
          distill: {
            name: "distill",
            mode: "subagent" as const,
            options: {},
            native: true,
            hidden: true,
            prompt: PROMPT_DISTILL,
            permission: Permission.merge(
              defaults,
              Permission.fromConfig({
                "*": "deny",
                read: "allow",
                write: "allow",
                edit: "allow",
                glob: "allow",
                grep: "allow",
                memory: "allow",
                bash: "allow",
                external_directory: {
                  [path.join(Global.Path.data, "memory")]: "allow",
                  [path.join(Global.Path.data, "memory", "*")]: "allow",
                },
              }),
              user,
            ),
            toolAllowlist: ["read", "write", "edit", "glob", "grep", "memory", "bash"],
          },
        }

        // 【合并用户配置】遍历 cfg.agent，让用户可以：
        //   - disable: 删掉某个内置 agent
        //   - 覆盖已有 agent 的字段（model/prompt/权限…）
        //   - 新增一个自定义 agent（此时 native:false）
        // 下面逐字段用 `value.x ?? item.x`：用户没配就保留内置默认值。
        for (const [key, value] of Object.entries(cfg.agent ?? {})) {
          if (value.disable) {
            delete agents[key]
            continue
          }
          let item = agents[key]
          // 配置里的 key 在内置表里不存在 → 视为用户新增的自定义 agent
          if (!item)
            item = agents[key] = {
              name: key,
              mode: "all",
              permission: Permission.merge(defaults, user),
              options: {},
              native: false,
            }
          if (value.model) {
            if (value.model.includes("/")) item.model = Provider.parseModel(value.model)
            else item.modelRef = value.model
          }
          item.variant = value.variant ?? item.variant
          item.prompt = value.prompt ?? item.prompt
          item.description = value.description ?? item.description
          item.temperature = value.temperature ?? item.temperature
          item.topP = value.top_p ?? item.topP
          item.mode = value.mode ?? item.mode
          item.color = value.color ?? item.color
          item.hidden = value.hidden ?? item.hidden
          item.name = value.name ?? item.name
          item.steps = value.steps ?? item.steps
          item.toolAllowlist = value.tool_allowlist ?? item.toolAllowlist
          item.options = mergeDeep(item.options, value.options ?? {})
          item.permission = Permission.merge(item.permission, Permission.fromConfig(value.permission ?? {}))
        }

        // 确保 Truncate.GLOB 和各技能目录默认被允许（除非已被显式配置为 deny）
        for (const name in agents) {
          const agent = agents[name]
          const globs = whitelistedDirs.filter(
            (glob) =>
              !agent.permission.some((r) => r.permission === "external_directory" && r.action === "deny" && r.pattern === glob),
          )
          if (globs.length === 0) continue

          agents[name].permission = Permission.merge(
            agents[name].permission,
            Permission.fromConfig({ external_directory: Object.fromEntries(globs.map((g) => [g, "allow" as const])) }),
          )
        }

        // 下面是 State 的三个方法实现（都是在上面构建好的 agents 这份数据上做查询）：

        // get：按名字直接取一个 agent
        const get = Effect.fnUntraced(function* (agent: string) {
          return agents[agent]
        })

        // list：返回全部 agent，并按固定优先级排序（默认 agent 和 build/plan/... 排前面）
        const list = Effect.fnUntraced(function* () {
          const cfg = yield* config.get()
          return pipe(
            agents,
            values(),
            sortBy(
              [(x) => cfg.default_agent !== undefined && x.name === cfg.default_agent, "desc"],
              [(x) => x.name === "build", "desc"],
              [(x) => x.name === "plan", "desc"],
              [(x) => x.name === "compose", "desc"],
              [(x) => x.name === "orchestrator", "desc"],
              [(x) => x.name === "max", "desc"],
              [(x) => x.name, "asc"],
            ),
          )
        })

        // defaultAgent：决定默认用哪个主 agent。
        //   - 若用户配了 default_agent，就用它（但会校验：必须存在、不是子 agent、不是隐藏的）
        //   - 否则取第一个"可见的主 agent"（通常就是 build）
        const defaultAgent = Effect.fnUntraced(function* () {
          const c = yield* config.get()
          if (c.default_agent) {
            const agent = agents[c.default_agent]
            if (!agent) throw new Error(`default agent "${c.default_agent}" not found`)
            if (agent.mode === "subagent") throw new Error(`default agent "${c.default_agent}" is a subagent`)
            if (agent.hidden === true) throw new Error(`default agent "${c.default_agent}" is hidden`)
            return agent.name
          }
          const visible = Object.values(agents).find((a) => a.mode !== "subagent" && a.hidden !== true)
          if (!visible) throw new Error("no primary visible agent found")
          return visible.name
        })

        return {
          get,
          list,
          defaultAgent,
        } satisfies State
      }),
    )

    // 【组装 Service 实例】Service.of({...}) = 把 Interface 的 4 个方法真正实现出来。
    // get/list/defaultAgent 都是"取出当前实例缓存的 state，再调它上面的同名方法"。
    // generate 不依赖 state，直接调 LLM。
    return Service.of({
      get: Effect.fn("Agent.get")(function* (agent: string) {
        return yield* InstanceState.useEffect(state, (s) => s.get(agent))
      }),
      list: Effect.fn("Agent.list")(function* () {
        return yield* InstanceState.useEffect(state, (s) => s.list())
      }),
      defaultAgent: Effect.fn("Agent.defaultAgent")(function* () {
        return yield* InstanceState.useEffect(state, (s) => s.defaultAgent())
      }),
      // generate：给一句需求描述，让 LLM 产出一份新 agent 的配置草稿
      // （identifier / whenToUse / systemPrompt）。用 schema 约束输出为结构化 JSON。
      // 会把已存在的 agent 名字告诉模型，避免重名。openai oauth 走 streamObject 分支，
      // 其它情况走 generateObject。
      generate: Effect.fn("Agent.generate")(function* (input: {
        description: string
        model?: { providerID: ProviderID; modelID: ModelID }
      }) {
        const cfg = yield* config.get()
        const model = input.model ?? (yield* provider.defaultModel())
        const resolved = yield* provider.getModel(model.providerID, model.modelID)
        const language = yield* provider.getLanguage(resolved)
        const tracer = cfg.experimental?.openTelemetry
          ? Option.getOrUndefined(yield* Effect.serviceOption(OtelTracer.OtelTracer))
          : undefined

        const system = [PROMPT_GENERATE]
        yield* plugin.trigger("experimental.chat.system.transform", { model: resolved }, { system })
        const existing = yield* InstanceState.useEffect(state, (s) => s.list())

        // TODO: 清理一下这里，避免特定 provider 的逻辑渗透进来
        const authInfo = yield* auth.get(model.providerID).pipe(Effect.orDie)
        const isOpenaiOauth = model.providerID === "openai" && authInfo?.type === "oauth"

        const params = {
          experimental_telemetry: {
            isEnabled: cfg.experimental?.openTelemetry,
            tracer,
            metadata: {
              userId: cfg.username ?? "unknown",
            },
          },
          temperature: 0.3,
          messages: [
            ...(isOpenaiOauth
              ? []
              : system.map(
                  (item): ModelMessage => ({
                    role: "system",
                    content: item,
                  }),
                )),
            {
              role: "user",
              content: `Create an agent configuration based on this request: "${input.description}".\n\nIMPORTANT: The following identifiers already exist and must NOT be used: ${existing.map((i) => i.name).join(", ")}\n  Return ONLY the JSON object, no other text, do not wrap in backticks`,
            },
          ],
          model: language,
          schema: z.object({
            identifier: z.string(),
            whenToUse: z.string(),
            systemPrompt: z.string(),
          }),
        } satisfies Parameters<typeof generateObject>[0]

        if (isOpenaiOauth) {
          return yield* Effect.promise(async () => {
            const result = streamObject({
              ...params,
              providerOptions: ProviderTransform.providerOptions(resolved, {
                instructions: system.join("\n"),
                store: false,
              }),
              onError: () => {},
            })
            for await (const part of result.fullStream) {
              if (part.type === "error") throw part.error
            }
            return result.object
          })
        }

        return yield* Effect.promise(() => generateObject(params).then((r) => r.object))
      }),
    })
  }),
)

// 【defaultLayer】把 layer 声明的 5 个上游依赖逐个"喂"进去（Layer.provide），
// 得到一个不再有悬空依赖、可以直接使用的完整服务层。
// 注意：这里注入的是 5 个"依赖服务"，跟"有几个 agent"无关。
export const defaultLayer = layer.pipe(
  Layer.provide(Plugin.defaultLayer),
  Layer.provide(Provider.defaultLayer),
  Layer.provide(Auth.defaultLayer),
  Layer.provide(Config.defaultLayer),
  Layer.provide(Skill.defaultLayer),
)

// 把整个模块以命名空间 Agent 导出，别处即可写 `Agent.Service` / `Agent.Info` 等。
export * as Agent from "./agent"
