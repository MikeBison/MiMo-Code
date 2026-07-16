// Late-bound reference to the tool set executable from inside tool_script.
//
// tool_script needs the full ToolRegistry def list to dispatch guest RPC calls,
// but the registry itself constructs tool_script (registry → tool_script →
// registry would be a module cycle). Mirroring workflowRef (workflow/runtime-ref.ts):
// the registry layer populates this module-local reference on initialisation and
// the tool reads it at call time.
import type { Effect } from "effect"
import type * as Tool from "./tool"

export const toolScriptRegistry: {
  current: (() => Effect.Effect<Tool.Def[]>) | undefined
} = { current: undefined }

// Agent control-flow tools make no sense inside a script (they steer the
// conversation, not data) — excluded from both the declared API and dispatch.
export const TOOL_SCRIPT_EXCLUDED = new Set([
  "tool_script",
  "invalid",
  "question",
  "task",
  "actor",
  "skill",
  "plan_enter",
  "plan_exit",
  "cron",
  "session",
  "workflow",
  "change_directory",
])
