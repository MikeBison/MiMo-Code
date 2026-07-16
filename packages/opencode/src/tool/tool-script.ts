import z from "zod"
import os from "os"
import path from "path"
import { Effect } from "effect"
import { EffectBridge, InstanceState } from "@/effect"
import { Log, Filesystem } from "@/util"
import { evalScript, type HostFn } from "../workflow/sandbox"
import { toolScriptRegistry, TOOL_SCRIPT_EXCLUDED } from "./tool-script-ref"
import DESCRIPTION from "./tool-script.txt"
import * as Tool from "./tool"

const log = Log.create({ service: "tool.tool_script" })

const MAX_TOOL_CALLS = 50
const MAX_CONCURRENT = 8
const ACTIVE_DEADLINE_MS = 60_000
const WALL_DEADLINE_MS = 30 * 60 * 1000
const MAX_RESULT_BYTES = 256 * 1024
const MAX_LOG_BYTES = 64 * 1024
const MAX_CODE_BYTES = 128 * 1024
const MAX_FILE_BYTES = 10 * 1024 * 1024

/** JSON Schema (zod v4 toJSONSchema output) → compact TS type text. Best-effort:
 * anything unrecognized renders as `unknown`, which is safe for declarations. */
function schemaToTs(schema: any): string {
  if (!schema || typeof schema !== "object") return "unknown"
  if (schema.const !== undefined) return JSON.stringify(schema.const)
  if (schema.enum) return schema.enum.map((v: unknown) => JSON.stringify(v)).join(" | ")
  const variants = schema.anyOf ?? schema.oneOf
  if (variants) return variants.map(schemaToTs).join(" | ")
  switch (schema.type) {
    case "string":
      return "string"
    case "number":
    case "integer":
      return "number"
    case "boolean":
      return "boolean"
    case "null":
      return "null"
    case "array":
      return `Array<${schemaToTs(schema.items)}>`
    case "object": {
      if (!schema.properties) {
        if (schema.additionalProperties && typeof schema.additionalProperties === "object")
          return `Record<string, ${schemaToTs(schema.additionalProperties)}>`
        return "Record<string, unknown>"
      }
      const required = new Set<string>(schema.required ?? [])
      const fields = Object.entries(schema.properties).map(
        ([key, value]) => `${key}${required.has(key) ? "" : "?"}: ${schemaToTs(value)}`,
      )
      return `{ ${fields.join("; ")} }`
    }
    default:
      return "unknown"
  }
}

/** Render the `tools` API declaration block appended to the tool description. */
export function renderToolScriptDeclarations(defs: Tool.Def[]): string {
  const lines = defs
    .filter((def) => !TOOL_SCRIPT_EXCLUDED.has(def.id))
    .map((def) => {
      const summary = def.description.split("\n").find((l) => l.trim()) ?? ""
      const input = schemaToTs(z.toJSONSchema(def.parameters))
      return `  /** ${summary.trim().slice(0, 200)} */\n  ${def.id}(input: ${input}): Promise<ToolResult>`
    })
  return [
    "```ts",
    "type ToolResult = { title: string; output: string; metadata: Record<string, unknown> }",
    "declare const tools: {",
    ...lines,
    "}",
    "// Raw file IO for machine-to-machine data (pipelines across executions).",
    "declare const files: {",
    "  /** Raw file contents — no line numbers, no truncation. null if missing. Paths: worktree or OS tmp. */",
    "  readText(path: string): Promise<string | null>",
    "  /** Write raw text; parent dirs auto-created. OS tmp dir ONLY — project writes go through tools.write/edit. */",
    "  writeText(path: string, content: string): Promise<void>",
    "}",
    "```",
  ].join("\n")
}

/** Guest-side prelude: `tools` proxy → __callTool RPC, console → __log capture.
 * Prepended AFTER transpilation so it stays plain JS. The catch-rethrow exists
 * because the sandbox promise bridge rejects with a plain STRING (not Error) —
 * wrapping restores `e.message` / `e instanceof Error` for guest catch blocks. */
const GUEST_PRELUDE = `
const tools = new Proxy({}, {
  get: (_t, name) => (args) =>
    __callTool(String(name), args === undefined ? {} : args).catch((e) => {
      throw e instanceof Error ? e : new Error(String(e));
    }),
});
const __fmt = (x) => {
  if (typeof x === "string") return x;
  try { return JSON.stringify(x); } catch { return String(x); }
};
const console = {
  log: (...a) => __log(a.map(__fmt).join(" ")),
  error: (...a) => __log("[error] " + a.map(__fmt).join(" ")),
  warn: (...a) => __log("[warn] " + a.map(__fmt).join(" ")),
};
const __wrapErr = (e) => {
  throw e instanceof Error ? e : new Error(String(e));
};
// marshalIn maps host null to guest undefined; normalize back so the declared
// "string | null" contract holds for === null checks.
const files = {
  readText: (p) => __readText(p).then((v) => (v === undefined ? null : v), __wrapErr),
  writeText: (p, c) => __writeText(p, c).catch(__wrapErr),
};
`

/** Jail for the `files` raw-IO primitives. Read: worktree + OS tmp. Write: OS
 * tmp ONLY — project writes must go through tools.write/edit so Permission.ask
 * applies (enforced here, not just advised in the prompt). Lexical containment
 * (same posture as workflow's resolveInWorkspace) — blocks `..` traversal and
 * out-of-jail absolutes; symlink resolution deferred. */
function resolveJailed(roots: string[], p: string, kind: "read" | "write"): string {
  const abs = path.resolve(roots[0], p)
  if (roots.some((root) => abs === root || Filesystem.contains(root, abs))) return abs
  throw new Error(
    kind === "write"
      ? `files.writeText is limited to the OS temp dir — write project files via tools.write/tools.edit: ${JSON.stringify(p)}`
      : `path outside allowed roots (worktree, tmp): ${JSON.stringify(p)}`,
  )
}

type TraceEntry = {
  name: string
  status: "success" | "error"
  durationMs: number
  error?: string
}

function makeSemaphore(max: number) {
  let active = 0
  const queue: Array<() => void> = []
  return async <T>(fn: () => Promise<T>): Promise<T> => {
    if (active >= max) await new Promise<void>((resolve) => queue.push(resolve))
    active++
    try {
      return await fn()
    } finally {
      active--
      queue.shift()?.()
    }
  }
}

export const ToolScriptTool = Tool.define(
  "tool_script",
  Effect.gen(function* () {
    return {
      description: DESCRIPTION,
      parameters: z.object({
        code: z
          .string()
          .describe(
            "TypeScript (or JavaScript) source for the body of an async function. Call tools via the global `tools` object; `return` the final aggregated value.",
          ),
      }),
      execute: (params: { code: string }, ctx: Tool.Context) =>
        Effect.gen(function* () {
          if (Buffer.byteLength(params.code, "utf8") > MAX_CODE_BYTES) {
            return {
              title: "code too large",
              metadata: { status: "code_error", toolCalls: 0 },
              output: `status: code_error\ncode exceeds ${MAX_CODE_BYTES} bytes`,
            }
          }

          const getDefs = toolScriptRegistry.current
          if (!getDefs) throw new Error("tool_script registry unavailable")
          const defs = (yield* getDefs()).filter((def) => !TOOL_SCRIPT_EXCLUDED.has(def.id))
          const byId = new Map(defs.map((def) => [def.id, def]))
          // Non-git projects report worktree === "/" (see Instance.containsPath) —
          // "/" as a jail root would allow EVERYTHING. Fall back to the project
          // directory in that case. Relative guest paths resolve against roots[0].
          const ins = yield* InstanceState.context
          const jailRoots = [ins.worktree === "/" ? ins.directory : ins.worktree, os.tmpdir()]

          // Snapshot the Effect context BEFORE crossing into Promise-land: the
          // quickjs hook boundary loses Instance/Workspace context otherwise.
          const bridge = yield* EffectBridge.make()

          // Wrap before transpiling: the code is the BODY of an async function
          // (top-level `return`/`await`), which is invalid at module top level —
          // Bun.Transpiler would reject it. The wrapped form transpiles to a plain
          // JS async-arrow expression the guest body can invoke.
          const transpiled = yield* Effect.try({
            try: () => new Bun.Transpiler({ loader: "ts" }).transformSync(`globalThis.__main = async () => {\n${params.code}\n}`),
            catch: (err) => err,
          }).pipe(
            Effect.catch((err) =>
              Effect.succeed({
                error: `TypeScript transpile failed: ${err instanceof Error ? err.message : String(err)}`,
              }),
            ),
          )
          if (typeof transpiled === "object") {
            return {
              title: "transpile error",
              metadata: { status: "code_error", toolCalls: 0 },
              output: `status: code_error\n${transpiled.error}`,
            }
          }

          const trace: TraceEntry[] = []
          const logs: string[] = []
          let logBytes = 0
          let calls = 0
          const withSlot = makeSemaphore(MAX_CONCURRENT)

          // Live progress for the TUI: after each settled call, publish the
          // aggregated per-tool counts through the OUTER part's metadata (each
          // ctx.metadata fires a part delta the ToolScript view renders
          // reactively). Fire-and-forget — progress must never fail a call.
          const publishProgress = () => {
            const counts: Record<string, { n: number; errors: number }> = {}
            for (const t of trace) {
              const c = (counts[t.name] ??= { n: 0, errors: 0 })
              c.n++
              if (t.status === "error") c.errors++
            }
            bridge.promise(ctx.metadata({ metadata: { running: true, toolCalls: trace.length, counts } })).catch(() => {})
          }

          const callTool: HostFn = (name: unknown, args: unknown) => {
            const id = String(name)
            const def = byId.get(id)
            if (!def) return Promise.reject(new Error(`unknown tool: ${id}`))
            calls++
            if (calls > MAX_TOOL_CALLS)
              return Promise.reject(new Error(`tool call budget exceeded (${MAX_TOOL_CALLS} per execution)`))
            const seq = calls
            const start = Date.now()
            return withSlot(() =>
              bridge
                .promise(
                  def.execute(args, {
                    ...ctx,
                    callID: `${ctx.callID ?? "tool_script"}:${seq}`,
                    // Sub-call metadata would clobber the outer tool_script call's
                    // title in the UI — swallow it; the trace covers observability.
                    metadata: () => Effect.void,
                  }),
                )
                .then(
                  (result) => {
                    trace.push({ name: id, status: "success", durationMs: Date.now() - start })
                    publishProgress()
                    return { title: result.title, output: result.output, metadata: result.metadata }
                  },
                  (err) => {
                    const message = err instanceof Error ? err.message : String(err)
                    trace.push({ name: id, status: "error", durationMs: Date.now() - start, error: message })
                    publishProgress()
                    throw new Error(`${id}: ${message}`)
                  },
                ),
            )
          }

          const logHook: HostFn = (message: unknown) => {
            const text = String(message)
            if (logBytes >= MAX_LOG_BYTES) return undefined
            logBytes += Buffer.byteLength(text, "utf8")
            logs.push(logBytes >= MAX_LOG_BYTES ? text.slice(0, 200) + " …(log budget exhausted)" : text)
            return undefined
          }

          // Raw file IO (`files.*`): machine-to-machine data channel, bypassing the
          // agent-facing read/write formatting (line numbers, truncation). Reads are
          // jailed to worktree + OS tmp; writes to OS tmp ONLY (project writes must
          // carry permissions → tools.write/edit). Read side also caps size so a
          // giant file can't blow the guest memory limit.
          const readText: HostFn = async (p: unknown) => {
            const abs = resolveJailed(jailRoots, String(p), "read")
            const file = Bun.file(abs)
            if (!(await file.exists())) return null
            if (file.size > MAX_FILE_BYTES) throw new Error(`file exceeds ${MAX_FILE_BYTES} bytes: ${String(p)}`)
            return file.text()
          }
          const writeText: HostFn = async (p: unknown, content: unknown) => {
            const abs = resolveJailed([os.tmpdir()], String(p), "write")
            const text = String(content)
            if (Buffer.byteLength(text, "utf8") > MAX_FILE_BYTES)
              throw new Error(`content exceeds ${MAX_FILE_BYTES} bytes`)
            await Filesystem.write(abs, text)
            return undefined
          }

          const outcome = yield* Effect.tryPromise({
            try: () =>
              evalScript(GUEST_PRELUDE + "\n" + transpiled + "\nreturn await globalThis.__main()", {
                __callTool: callTool,
                __log: logHook,
                __readText: readText,
                __writeText: writeText,
              }, {
                deterministic: false,
                deadlineMs: WALL_DEADLINE_MS,
                activeDeadlineMs: ACTIVE_DEADLINE_MS,
                interrupt: () => ctx.abort.aborted,
              }),
            catch: (err) => (err instanceof Error ? err : new Error(String(err))),
          }).pipe(Effect.result)

          const traceLines = trace.map(
            (t) => `- ${t.name} → ${t.status}${t.error ? ` (${t.error.slice(0, 200)})` : ""} [${t.durationMs}ms]`,
          )
          const logBlock = logs.length ? `\n\nLogs:\n${logs.join("\n")}` : ""
          const traceBlock = trace.length ? `\n\nTool calls (${trace.length}):\n${traceLines.join("\n")}` : ""

          if (outcome._tag === "Failure") {
            const message = outcome.failure instanceof Error ? outcome.failure.message : String(outcome.failure)
            const status = ctx.abort.aborted
              ? "cancelled"
              : message.includes("deadline exceeded") || message.includes("interrupted")
                ? "timeout"
                : message.includes("budget exceeded")
                  ? "budget_exceeded"
                  : "code_error"
            log.warn("tool_script failed", { status, message: message.slice(0, 500) })
            return {
              title: status,
              metadata: { status, toolCalls: trace.length },
              output: `status: ${status}\n${message}${logBlock}${traceBlock}`,
            }
          }

          const json = JSON.stringify(outcome.success, null, 2) ?? "undefined"
          if (Buffer.byteLength(json, "utf8") > MAX_RESULT_BYTES) {
            return {
              title: "result too large",
              metadata: { status: "budget_exceeded", toolCalls: trace.length },
              output: `status: budget_exceeded\nreturned value is ${json.length} bytes (max ${MAX_RESULT_BYTES}). Aggregate or slice the data before returning.${logBlock}${traceBlock}`,
            }
          }

          return {
            title: `${trace.length} tool calls`,
            metadata: { status: "completed", toolCalls: trace.length },
            output: `status: completed\n\nResult:\n${json}${logBlock}${traceBlock}`,
          }
        }).pipe(Effect.orDie),
    }
  }),
)
