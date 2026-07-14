import { Flag } from "@/flag/flag"
import type { MessageV2 } from "../message-v2"

/**
 * Empty / no-op tool-call loop guard.
 *
 * Sibling to text-ngram-detection: the text-ngram ladder only inspects TEXT
 * parts, so a model that spins by re-emitting content-free tool calls (or by
 * "calling a tool" that produces no structured tool part at all) slips past it
 * with no text to match. stepSignature (prompt.ts) also misses this: it signs
 * only `part.type === "tool"` steps and returns undefined for a step with zero
 * tool parts, so an empty terminal step is dropped from repeat counting instead
 * of counted. This module fills that gap with a pure classifier + a soft→hard
 * recovery ladder mirroring TEXT_NGRAM_MAX_RECOVERY.
 *
 * This is a HARNESS backstop for a MODEL bug: a degraded model that keeps
 * emitting empty/no-op steps will never self-correct from a reminder, so after
 * a bounded number of soft nudges the harness must hard-halt the turn rather
 * than spin forever.
 */

export const EMPTY_STEP_MAX_RECOVERY = Flag.MIMOCODE_EMPTY_STEP_MAX_RECOVERY

/**
 * Is this assistant step an empty / no-op tool call?
 *
 * Two shapes count as empty (mirrors the task's definition):
 *
 *  (a) The step emitted one or more client (non-providerExecuted) tool parts,
 *      but EVERY such tool part has an empty/invalid input — no keys, or only
 *      keys whose values are null/undefined/empty-string/whitespace. The model
 *      "called a tool" but passed nothing actionable, so the call cannot make
 *      progress and re-looping just repeats it.
 *
 *  (b) The step produced NO client tool part at all AND no substantive text and
 *      no substantive reasoning — a fully empty terminal. (This overlaps with
 *      classify's `invalid`/"empty output", but we count it here too so the
 *      hard-halt ladder can escalate on a run of them rather than only softly
 *      nudging via autoContinueInvalidOutput.)
 *
 * A step that emits at least one tool part with real input, or any substantive
 * text/reasoning, is NOT empty — the model is making some kind of progress.
 *
 * Provider-executed tool parts (e.g. server-side web search) are ignored for
 * the "has a tool part" test: they are not client actions and their presence
 * does not mean the model issued an actionable call.
 */
export function isEmptyStep(parts: readonly MessageV2.Part[]): boolean {
  const clientToolParts = parts.filter(
    (part): part is Extract<MessageV2.Part, { type: "tool" }> =>
      part.type === "tool" && !part.metadata?.providerExecuted,
  )

  if (clientToolParts.length > 0) {
    // (a) Every client tool part has an empty/invalid input.
    return clientToolParts.every((part) => isEmptyInput(part.state.input))
  }

  // (b) No client tool part — empty only if there is also no substantive
  // text and no substantive reasoning (a pure-empty terminal). A step with a
  // real text answer or real reasoning is a legitimate (non-loop) outcome.
  const hasSubstantiveText = parts.some(
    (part) => part.type === "text" && !part.synthetic && !part.ignored && part.text.trim().length > 0,
  )
  if (hasSubstantiveText) return false
  const hasSubstantiveReasoning = parts.some(
    (part) => part.type === "reasoning" && part.text.trim().length > 0,
  )
  if (hasSubstantiveReasoning) return false
  return true
}

/**
 * An input object counts as empty when it has no keys, or every value is
 * null/undefined/empty-string/whitespace-only. Nested objects/arrays with any
 * content count as non-empty (the model passed *something*).
 */
function isEmptyInput(input: Record<string, unknown> | undefined | null): boolean {
  if (input === undefined || input === null) return true
  const keys = Object.keys(input)
  if (keys.length === 0) return true
  return keys.every((k) => isEmptyValue(input[k]))
}

function isEmptyValue(value: unknown): boolean {
  if (value === undefined || value === null) return true
  if (typeof value === "string") return value.trim().length === 0
  if (Array.isArray(value)) return value.length === 0
  if (typeof value === "object") return Object.keys(value as Record<string, unknown>).length === 0
  // number / boolean → the model passed a real value.
  return false
}

export const EMPTY_STEP_RECOVERY_REMIND = [
  "<system-reminder>",
  "NO PROGRESS: your previous step made no valid tool call and produced no answer",
  "(the tool call had empty/invalid arguments, or there was no tool call and no text).",
  "Stop repeating an empty step. Do exactly ONE of these now:",
  "- Issue a valid tool call with COMPLETE, non-empty arguments, or",
  "- Reply to the user directly with plain text.",
  "Do NOT emit another empty or argument-less tool call.",
  "</system-reminder>",
].join("\n")

export const EMPTY_STEP_RECOVERY_REPLAN = [
  "<system-reminder>",
  "STILL NO PROGRESS: you are repeating empty/no-op tool calls after a reminder.",
  "This is your final chance before the turn is halted. You MUST either:",
  "1. Send a single valid tool call whose arguments are fully populated, or",
  "2. Give the user a plain-text response explaining the result or the blocker.",
  "Any further empty/argument-less tool call will terminate this turn.",
  "</system-reminder>",
].join("\n")
