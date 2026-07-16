/**
 * Tool framework contracts and the base decorator for defining tools.
 *
 * Declares the shape of a tool (`Def`, `Context`, `ExecuteResult`) and provides the
 * `define` factory plus the internal `wrap` decorator that gives every tool argument
 * validation and output truncation. This module frames and decorates a single tool at
 * a time; it does not pick, route, or run tools — dispatch happens in the model/AI SDK
 * layer and the concrete work lives in each tool module (read, bash, edit, ...).
 *
 * 工具框架的契约定义与"基础装饰器"。声明单个工具的形状（`Def`、`Context`、`ExecuteResult`），
 * 并提供 `define` 工厂以及内部的 `wrap` 装饰器——后者为每个工具统一套上参数校验与输出截断。
 * 本模块一次只描述并装饰一个工具，不负责选择、路由或执行工具：分发发生在模型 / AI SDK 层，
 * 具体逻辑位于各工具模块（read、bash、edit 等）中。
 */
import z from "zod"
import { Effect } from "effect"
import type { MessageV2 } from "../session/message-v2"
import type { Permission } from "../permission"
import type { SessionID, MessageID } from "../session/schema"
import * as Truncate from "./truncate"
import { RecoverableError } from "./recoverable"
import { Agent } from "@/agent/agent"

/**
 * Arbitrary key-value metadata a tool attaches to its result for the UI and telemetry.
 *
 * 工具附加到执行结果上的任意键值元数据，供 UI 展示与埋点使用。
 */
export interface Metadata {
  [key: string]: any
}

/**
 * Produces a tool's description dynamically from the calling agent, letting one tool
 * present different guidance per agent.
 *
 * 根据调用方 agent 动态生成工具描述，使同一个工具能对不同 agent 呈现不同说明。
 *
 * @param agent - The agent the description is generated for / 用于生成描述的 agent
 * @returns An effect resolving to the description text / 解析为描述文本的 effect
 */
// TODO: remove this hack
export type DynamicDescription = (agent: Agent.Info) => Effect.Effect<string>

/**
 * Per-execution context injected into a tool's `execute`. Carries session identity, the
 * cancellation signal, and the callbacks a tool uses to report progress (`metadata`)
 * and request permission (`ask`), keeping the tool decoupled from how the UI updates or
 * how permissions are resolved.
 *
 * 注入到工具 `execute` 的单次执行上下文。携带会话身份、取消信号，以及工具用来上报进度
 * （`metadata`）和申请权限（`ask`）的回调，使工具与"如何更新 UI、如何判定权限"解耦。
 *
 * @template M - The metadata shape this tool reports / 该工具上报的元数据结构
 */
export type Context<M extends Metadata = Metadata> = {
  /** Id of the session this call belongs to / 本次调用所属会话的 id */
  sessionID: SessionID
  /** Id of the assistant message the tool call is attached to / 工具调用挂靠的助手消息 id */
  messageID: MessageID
  /** Name of the agent invoking the tool / 调用该工具的 agent 名字 */
  agent: string
  /** Id of the actor (subagent) on whose behalf the call runs, if any / 代表哪个 actor（子 agent）执行，若有 */
  actorID?: string
  /** Id of the task this call is part of, if any / 本次调用所属任务的 id，若有 */
  taskId?: string
  /** Signal fired when the run is cancelled; long-running operations should honor it / 运行被取消时触发的信号，长操作应遵守它 */
  abort: AbortSignal
  /** Id of this specific tool call / 本次工具调用的唯一 id */
  callID?: string
  /** Extra ambient values passed through to the tool, such as model and promptOps / 透传给工具的额外上下文，如 model、promptOps */
  extra?: { [key: string]: unknown }
  /** Full message history visible to the tool / 工具可见的完整消息历史 */
  messages: MessageV2.WithParts[]
  /** Reports a progress title and metadata back to the UI mid-execution / 执行过程中向 UI 回传进度标题与元数据 */
  metadata(input: { title?: string; metadata?: M }): Effect.Effect<void>
  /** Requests permission before a sensitive action; the resolution logic is injected / 执行敏感操作前申请权限，判定逻辑被注入进来 */
  ask(input: Omit<Permission.Request, "id" | "sessionID" | "tool">): Effect.Effect<void>
}

/**
 * The result a tool returns from `execute`: a display title, the model-facing output
 * text, structured metadata, and optional file attachments.
 *
 * 工具从 `execute` 返回的结果：展示用标题、面向模型的输出文本、结构化元数据，以及可选的文件附件。
 *
 * @template M - The metadata shape / 元数据结构
 */
export interface ExecuteResult<M extends Metadata = Metadata> {
  /** Short title shown in the UI for this tool call / 该工具调用在 UI 上显示的简短标题 */
  title: string
  /** Structured metadata for the UI and telemetry / 供 UI 与埋点使用的结构化元数据 */
  metadata: M
  /** Output text fed back to the model / 回传给模型的输出文本 */
  output: string
  /** Optional file parts produced by the tool / 工具产出的可选文件附件 */
  attachments?: Omit<MessageV2.FilePart, "id" | "sessionID" | "messageID">[]
}

/**
 * A tool definition: its stable id, model-facing description, argument schema, and the
 * `execute` implementation. Optionally supports a `shell` mode so the tool can be
 * invoked from a shell-style command instead of a structured tool call.
 *
 * 一个工具定义：稳定的 id、面向模型的描述、参数 schema，以及 `execute` 实现。可选支持
 * `shell` 模式，使该工具能以 shell 命令形式调用，而非结构化工具调用。
 *
 * @template Parameters - Zod schema for the tool's arguments / 工具参数的 Zod schema
 * @template M - The metadata shape the tool reports / 工具上报的元数据结构
 */
export interface Def<Parameters extends z.ZodType = z.ZodType, M extends Metadata = Metadata> {
  /** Stable identifier the model uses to call this tool / 模型据此调用该工具的稳定标识 */
  id: string
  /** Description of what the tool does, shown to the model / 工具功能说明，展示给模型 */
  description: string
  /** Zod schema that validates the tool's arguments / 校验工具参数的 Zod schema */
  parameters: Parameters
  /** Runs the tool's actual work with validated args and the execution context / 用校验后的参数和执行上下文运行工具的实际逻辑 */
  execute(args: z.infer<Parameters>, ctx: Context): Effect.Effect<ExecuteResult<M>>
  /** Optionally formats a Zod validation error into a model-friendly message / 可选：把 Zod 校验错误格式化为对模型友好的消息 */
  formatValidationError?(error: z.ZodError): string
  /** Optional shell-mode support, letting the tool be driven by a shell-style command / 可选的 shell 模式支持，使工具能被 shell 命令驱动 */
  shell?: {
    /** Description of the shell form of this tool / 该工具 shell 形态的描述 */
    description: string
    /** Parses a shell script into one or more sets of tool arguments / 把 shell 脚本解析成一组或多组工具参数 */
    parse(script: string): Effect.Effect<z.infer<Parameters>[], unknown>
    // Optional recovery for shell-mode calls that arrive shaped like the tool's
    // JSON args (no usable `script`). Returns the tool's parsed JSON shape to be
    // routed to execute, or undefined if rawArgs can't be lifted. Lets shell mode
    // transparently accept a JSON-shape call instead of erroring.
    recover?(rawArgs: unknown): z.infer<Parameters> | undefined
  }
}
/**
 * A tool definition before its id is assigned. Tool modules return this shape, and
 * `define` attaches the id afterward.
 *
 * 尚未分配 id 的工具定义。工具模块返回这个结构，随后由 `define` 补上 id。
 */
export type DefWithoutID<Parameters extends z.ZodType = z.ZodType, M extends Metadata = Metadata> = Omit<
  Def<Parameters, M>,
  "id"
>

/**
 * A registered tool entry: its id plus a lazy `init` that builds the wrapped definition
 * on demand. This is what the registry stores and hands to `resolveTools`.
 *
 * 已注册的工具条目：其 id 加上一个惰性 `init`，按需构建包装后的定义。这是注册表存储、
 * 并交给 `resolveTools` 的结构。
 *
 * @template Parameters - Zod schema for the tool's arguments / 工具参数的 Zod schema
 * @template M - The metadata shape the tool reports / 工具上报的元数据结构
 */
export interface Info<Parameters extends z.ZodType = z.ZodType, M extends Metadata = Metadata> {
  /** Stable tool identifier / 稳定的工具标识 */
  id: string
  /** Lazily builds the wrapped tool definition / 惰性构建包装后的工具定义 */
  init: () => Effect.Effect<DefWithoutID<Parameters, M>>
}

/**
 * The raw definition source accepted by `define`: either a definition object directly,
 * or a factory effect that produces one for tools needing async setup.
 *
 * `define` 接受的原始定义来源：可以是定义对象本身，也可以是产出定义对象的工厂 effect
 *（用于需要异步初始化的工具）。
 */
type Init<Parameters extends z.ZodType, M extends Metadata> =
  | DefWithoutID<Parameters, M>
  | (() => Effect.Effect<DefWithoutID<Parameters, M>>)

/**
 * Extracts the parsed argument type from a tool `Info`, or from an effect producing one.
 *
 * 从工具 `Info`（或产出它的 effect）中提取解析后的参数类型。
 *
 * @template T - A tool `Info` or an effect resolving to one / 工具 `Info` 或解析为它的 effect
 */
export type InferParameters<T> =
  T extends Info<infer P, any> ? z.infer<P> : T extends Effect.Effect<Info<infer P, any>, any, any> ? z.infer<P> : never
/**
 * Extracts the metadata type from a tool `Info`, or from an effect producing one.
 *
 * 从工具 `Info`（或产出它的 effect）中提取元数据类型。
 *
 * @template T - A tool `Info` or an effect resolving to one / 工具 `Info` 或解析为它的 effect
 */
export type InferMetadata<T> =
  T extends Info<any, infer M> ? M : T extends Effect.Effect<Info<any, infer M>, any, any> ? M : never

/**
 * Extracts the full `Def` type from a tool `Info`, or from an effect producing one.
 *
 * 从工具 `Info`（或产出它的 effect）中提取完整的 `Def` 类型。
 *
 * @template T - A tool `Info` or an effect resolving to one / 工具 `Info` 或解析为它的 effect
 */
export type InferDef<T> =
  T extends Info<infer P, infer M>
    ? Def<P, M>
    : T extends Effect.Effect<Info<infer P, infer M>, any, any>
      ? Def<P, M>
      : never

/**
 * Builds the agent-facing message for an argument-validation failure. For a `ZodError`,
 * zod v4's `prettifyError` gives a precise, path-annotated breakdown of which field was
 * wrong and what was expected, which is far more actionable for the model than raw
 * issue JSON; any other error falls back to a generic rewrite instruction.
 *
 * 为参数校验失败构造面向 agent 的提示消息。对 `ZodError`，zod v4 的 `prettifyError` 会给出
 * 精确、带路径标注的说明（哪个字段错、期望是什么），比原始 issue JSON 更利于模型改正；
 * 其他错误则回退到一条通用的"请按 schema 重写"提示。
 *
 * @param id - The tool's id, used in the message / 工具 id，用于消息中
 * @param error - The validation error to describe. Can be:
 *   - `z.ZodError`: rendered with `prettifyError` / 用 `prettifyError` 渲染
 *   - any other value: rendered with a generic fallback / 用通用回退文案渲染
 * @returns A model-readable error message / 一条模型可读的错误消息
 * @example
 * const message = validationErrorMessage("read", new z.ZodError([]));
 */
export function validationErrorMessage(id: string, error: unknown): string {
  if (error instanceof z.ZodError) {
    return `Invalid arguments for the ${id} tool:\n${z.prettifyError(error)}`
  }
  return `The ${id} tool was called with invalid arguments: ${error}.\nPlease rewrite the input so it satisfies the expected schema.`
}

/**
 * Wraps a raw tool definition with the base decorator every tool receives: it validates
 * arguments against the tool's schema before running (raising a `RecoverableError` so the
 * model can rewrite and retry), runs the original `execute`, then truncates oversized
 * output. The original `execute` is captured in a local before the property is replaced,
 * so the wrapper calls the original rather than recursing into itself.
 *
 * 用每个工具都会获得的"基础装饰器"包装原始工具定义：运行前先按工具 schema 校验参数
 *（校验失败抛 `RecoverableError`，以便模型改写后重试），再调用原始 `execute`，最后截断过长输出。
 * 替换属性前会先把原始 `execute` 存入局部变量，因此包装层调用的是原始实现，而非递归自身。
 *
 * @param id - The tool's id, used for validation messages and tracing / 工具 id，用于校验消息与链路追踪
 * @param init - The raw definition or a factory effect producing one / 原始定义或产出它的工厂 effect
 * @param truncate - Service used to truncate oversized output / 用于截断过长输出的服务
 * @param agents - Registry used to resolve the calling agent's truncation limits / 用于解析调用方 agent 截断上限的注册表
 * @returns A lazy factory that yields the wrapped definition / 一个惰性工厂，产出包装后的定义
 */
function wrap<Parameters extends z.ZodType, Result extends Metadata>(
  id: string,
  init: Init<Parameters, Result>,
  truncate: Truncate.Interface,
  agents: Agent.Interface,
) {
  return () =>
    Effect.gen(function* () {
      // Resolve the raw definition (invoking the factory if needed) into a shallow copy,
      // so reassigning `execute` below does not mutate the caller's original object.
      // 把原始定义解析出来（必要时调用工厂）并做浅拷贝，这样下面重写 `execute` 时
      // 不会改动调用方的原始对象。
      const toolInfo = typeof init === "function" ? { ...(yield* init()) } : { ...init }
      // Capture the original `execute` in a local BEFORE overwriting the property.
      // The wrapper below calls this local, so it decorates rather than recurses.
      // 在覆盖属性之前，先把原始 `execute` 存入局部变量。下面的包装层调用的是这个局部变量，
      // 因此是"装饰"而非递归自身。
      const execute = toolInfo.execute
      // Replace `execute` with the decorated version: validate args, run the original,
      // then truncate output.
      // 把 `execute` 替换为装饰后的版本：先校验参数，再调原始实现，最后截断输出。
      toolInfo.execute = (args, ctx) => {
        // Tracing attributes attached to this tool call's span.
        // 附加到本次工具调用 span 上的链路追踪属性。
        const attrs = {
          "tool.name": id,
          "session.id": ctx.sessionID,
          "message.id": ctx.messageID,
          ...(ctx.callID ? { "tool.call_id": ctx.callID } : {}),
        }
        return Effect.gen(function* () {
          // Step 1: validate the model-supplied args against the tool's schema.
          // 步骤 1：用工具 schema 校验模型传入的参数。
          yield* Effect.try({
            try: () => toolInfo.parameters.parse(args),
            catch: (error) => {
              // Bad arguments are always agent-recoverable: the model sees the message
              // and rewrites the call next turn. Wrap in RecoverableError so the TUI
              // renders it muted instead of alarming the user with a red block.
              // 参数错误对 agent 总是可恢复的：模型会看到这条消息并在下一轮改写调用。
              // 包成 RecoverableError，让 TUI 以灰色低调渲染，而不是用红色报警块吓到用户。
              if (error instanceof z.ZodError && toolInfo.formatValidationError) {
                return new RecoverableError(toolInfo.formatValidationError(error), { cause: error })
              }
              return new RecoverableError(validationErrorMessage(id, error), { cause: error })
            },
          })
          // Step 2: run the tool's original implementation.
          // 步骤 2：运行工具的原始实现。
          const result = yield* execute(args, ctx)
          // Step 3a: if the tool already decided its own truncation, return as-is.
          // 步骤 3a：若工具已自行决定了截断状态，直接原样返回。
          if (result.metadata.truncated !== undefined) {
            return result
          }
          // Step 3b: otherwise truncate oversized output per the calling agent's limits.
          // 步骤 3b：否则按调用方 agent 的上限截断过长输出。
          const agent = yield* agents.get(ctx.agent)
          const truncated = yield* truncate.output(result.output, {}, agent)
          // Merge the truncated output back, recording whether truncation happened and
          // where the full output was spilled to (outputPath) when it did.
          // 把截断后的输出合并回结果，记录是否发生了截断，以及截断时完整输出落盘的路径（outputPath）。
          return {
            ...result,
            output: truncated.content,
            metadata: {
              ...result.metadata,
              truncated: truncated.truncated,
              ...(truncated.truncated && { outputPath: truncated.outputPath }),
            },
          }
          // orDie: any non-recoverable failure becomes a defect (crash), since tool bugs
          // are not part of the normal error channel. withSpan: wrap the run in a trace span.
          // orDie：任何不可恢复的失败都升级为 defect（崩溃），因为工具自身的 bug 不属于正常错误通道。
          // withSpan：把整次执行包进一个追踪 span。
        }).pipe(Effect.orDie, Effect.withSpan("Tool.execute", { attributes: attrs }))
      }
      // Return the definition with its `execute` now decorated.
      // 返回 `execute` 已被装饰过的定义。
      return toolInfo
    })
}

/**
 * Defines a tool: resolves its raw definition, pulls in the `Truncate` and `Agent`
 * services, applies the base decorator via `wrap`, and returns a tool `Info` tagged with
 * its id. This is the public entry point each tool module uses to declare itself.
 *
 * 定义一个工具：解析原始定义，注入 `Truncate` 与 `Agent` 服务，通过 `wrap` 套上基础装饰器，
 * 返回带 id 标记的工具 `Info`。这是每个工具模块用来声明自身的公共入口。
 *
 * @param id - The tool's stable identifier / 工具的稳定标识
 * @param init - An effect resolving to the tool's raw definition / 解析为工具原始定义的 effect
 * @returns An effect resolving to the tool `Info`, also carrying `id` as a property / 解析为工具 `Info` 的 effect，并附带 `id` 属性
 * @example
 * export const ReadTool = define(
 *   "read",
 *   Effect.succeed({
 *     description: "Read a file",
 *     parameters: z.object({ path: z.string() }),
 *     execute: (args, ctx) => Effect.succeed({ title: args.path, output: "...", metadata: {} }),
 *   }),
 * );
 */
export function define<Parameters extends z.ZodType, Result extends Metadata, R, ID extends string = string>(
  id: ID,
  init: Effect.Effect<Init<Parameters, Result>, never, R>,
): Effect.Effect<Info<Parameters, Result>, never, R | Truncate.Service | Agent.Service> & { id: ID } {
  return Object.assign(
    Effect.gen(function* () {
      const resolved = yield* init
      const truncate = yield* Truncate.Service
      const agents = yield* Agent.Service
      return { id, init: wrap(id, resolved, truncate, agents) }
    }),
    { id },
  )
}

/**
 * Materializes a tool `Info` into a ready-to-run `Def` by invoking its lazy `init` and
 * re-attaching the id.
 *
 * 通过调用工具 `Info` 的惰性 `init` 并重新附上 id，将其实例化为可直接运行的 `Def`。
 *
 * @param info - The registered tool entry to materialize / 要实例化的已注册工具条目
 * @returns An effect resolving to the runnable tool definition / 解析为可运行工具定义的 effect
 * @example
 * const def = yield* init(ReadTool);
 * const result = yield* def.execute({ path: "README.md" }, ctx);
 */
export function init<P extends z.ZodType, M extends Metadata>(info: Info<P, M>): Effect.Effect<Def<P, M>> {
  return Effect.gen(function* () {
    const init = yield* info.init()
    return {
      ...init,
      id: info.id,
    }
  })
}
