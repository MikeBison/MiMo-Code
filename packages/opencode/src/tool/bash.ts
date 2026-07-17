/**
 * The `bash` tool: runs a shell command in a child process on behalf of the model.
 *
 * Beyond just spawning, this module does three notable things: (1) parses the command
 * with a tree-sitter AST (bash and PowerShell) to statically scan which filesystem paths
 * and destructive operations it touches, so it can request the right permissions before
 * running; (2) streams the child's combined output back to the UI live while enforcing
 * size and line truncation; (3) races process exit against the user's abort signal and a
 * timeout, killing the child when either fires.
 *
 * `bash` 工具：代表模型在子进程中执行 shell 命令。
 *
 * 除了单纯地启动进程，本模块还做三件值得注意的事：(1) 用 tree-sitter AST（bash 与
 * PowerShell）解析命令，静态扫描它会触碰哪些文件系统路径和破坏性操作，从而在运行前申请
 * 正确的权限；(2) 一边把子进程的合并输出实时流式回传给 UI，一边执行体积与行数截断；
 * (3) 让"进程退出"与"用户取消信号"和"超时"三者竞速，任一触发就杀掉子进程。
 */
import z from "zod"
import os from "os"
import { createWriteStream, readFileSync } from "node:fs"
import * as Tool from "./tool"
import path from "path"
import DESCRIPTION from "./bash.txt"
import { Log } from "../util"
import { Instance } from "../project/instance"
import { lazy } from "@/util/lazy"
import { Language, type Node } from "web-tree-sitter"

import { AppFileSystem } from "@mimo-ai/shared/filesystem"
import { fileURLToPath } from "url"
import { Flag } from "@/flag/flag"
import { Shell } from "@/shell/shell"

import { SessionCwd } from "./session-cwd"
import { BashArity } from "@/permission/arity"
import * as Truncate from "./truncate"
import { Plugin } from "@/plugin"
import { Effect, Stream } from "effect"
import { ChildProcess } from "effect/unstable/process"
import { ChildProcessSpawner } from "effect/unstable/process/ChildProcessSpawner"
import * as BashInteractive from "./bash-interactive"
import * as BashTokenEfficient from "./bash_token_efficient_pipeline"
import * as BashTokenEfficientHeuristic from "./bash_token_efficient_heuristic"

/** Max characters of live output kept in the UI metadata preview / UI 元数据预览中保留的输出最大字符数 */
const MAX_METADATA_LENGTH = 30_000
/** Default command timeout in milliseconds when the caller omits one / 调用方未指定时的默认命令超时（毫秒） */
const DEFAULT_TIMEOUT = Flag.MIMOCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS || 2 * 60 * 1000
/** Command names treated as PowerShell / 被视为 PowerShell 的命令名 */
const PS = new Set(["powershell", "pwsh"])
/** Directory-changing commands (bash + PowerShell); excluded from path-permission scans / 切换目录的命令（bash + PowerShell），路径权限扫描时排除 */
const CWD = new Set(["cd", "push-location", "set-location"])
/** Commands whose arguments are filesystem paths, used to scan for out-of-workspace access / 参数为文件系统路径的命令，用于扫描工作区外访问 */
const FILES = new Set([
  ...CWD,
  "rm",
  "cp",
  "mv",
  "mkdir",
  "touch",
  "chmod",
  "chown",
  "cat",
  // Leave PowerShell aliases out for now. Common ones like cat/cp/mv/rm/mkdir
  // already hit the entries above, and alias normalization should happen in one
  // place later so we do not risk double-prompting.
  "get-content",
  "set-content",
  "add-content",
  "copy-item",
  "move-item",
  "remove-item",
  "new-item",
  "rename-item",
])
/** PowerShell parameters whose next token is a path value (for example `-Path <file>`) / 下一个 token 是路径值的 PowerShell 参数（如 `-Path <file>`） */
const FLAGS = new Set(["-destination", "-literalpath", "-path"])
/** PowerShell boolean switches that take no path value, skipped during path scanning / 不带路径值的 PowerShell 开关参数，路径扫描时跳过 */
const SWITCHES = new Set(["-confirm", "-debug", "-force", "-nonewline", "-recurse", "-verbose", "-whatif"])

// Irreversible file/directory removal commands. Names are matched
// case-insensitively for PowerShell; bash is case-sensitive.
const DELETE_COMMANDS = new Set([
  "rm",
  "rmdir",
  "unlink",
  "shred",
  // Windows / PowerShell removal verbs and their common aliases. `remove-item`
  // is the canonical verb; `ri`, `rd`, `del`, `erase` are aliases.
  "del",
  "erase",
  "rd",
  "remove-item",
  "ri",
])

// git subcommands that destroy history, working tree state, or remote branches.
// Value is the set of tokens (flag or subcommand keyword) that must appear
// anywhere in the argv for the invocation to count as destructive. An empty
// set means the subcommand is destructive on its own.
const GIT_DESTRUCTIVE = new Map<string, Set<string>>([
  ["reset", new Set(["--hard"])],
  ["clean", new Set(["-f", "-ff", "-fd", "-fdx", "-df", "-dfx", "-fx", "--force"])],
  ["branch", new Set(["-D", "--delete"])],
  ["tag", new Set(["-d", "--delete"])],
  ["worktree", new Set(["remove"])],
  ["push", new Set(["--force", "-f"])],
  ["stash", new Set(["drop", "clear"])],
])

/**
 * Argument schema for the `bash` tool: the command plus optional timeout, working
 * directory, interactive flag, and a short human-readable description.
 *
 * `bash` 工具的参数 schema：命令，加上可选的超时、工作目录、交互标志，以及一句人类可读的简短描述。
 */
const Parameters = z.object({
  command: z.string().describe("The command to execute"),
  timeout: z.number().describe("Optional timeout in milliseconds").optional(),
  workdir: z
    .string()
    .describe(
      `The working directory to run the command in. Defaults to the current directory. Use this instead of 'cd' commands.`,
    )
    .optional(),
  interactive: z
    .boolean()
    .describe(
      "Set to true when the command requires user interaction (password input, y/N confirmation, SSH key passphrase, etc). The terminal will be handed to the user for direct interaction.",
    )
    .optional(),
  description: z
    .string()
    .describe(
      "Clear, concise description of what this command does in 5-10 words. Examples:\nInput: ls\nOutput: Lists files in current directory\n\nInput: git status\nOutput: Shows working tree status\n\nInput: npm install\nOutput: Installs package dependencies\n\nInput: mkdir foo\nOutput: Creates directory 'foo'",
    ),
})

/**
 * A single token extracted from a parsed command node: its AST node type and raw text.
 *
 * 从解析后的命令节点中提取的单个 token：其 AST 节点类型与原始文本。
 */
type Part = {
  type: string
  text: string
}

/**
 * The result of statically scanning a command for permission purposes.
 *
 * 为权限判定而对命令做静态扫描的结果。
 */
type Scan = {
  /** Out-of-workspace directories the command touches / 命令触碰的工作区外目录 */
  dirs: Set<string>
  /** Full command strings to ask about (the exact command text) / 需询问的完整命令字符串（精确命令文本） */
  patterns: Set<string>
  /** Prefix-based "always allow" patterns (command prefix + " *") / 基于前缀的"总是允许"模式（命令前缀 + " *"） */
  always: Set<string>
  /** Commands performing irreversible deletion, needing forced confirmation / 执行不可逆删除、需强制确认的命令 */
  deletes: Set<string>
}

/**
 * A streamed output chunk with its precomputed UTF-8 byte size, used for the rolling
 * in-memory buffer that keeps only the most recent output.
 *
 * 一段流式输出块，附带预先算好的 UTF-8 字节大小，用于只保留最近输出的滚动内存缓冲。
 */
type Chunk = {
  text: string
  size: number
}

export const log = Log.create({ service: "bash-tool" })

/**
 * Resolves a tree-sitter wasm asset reference (file URL, absolute path, or bundler
 * specifier) into a concrete filesystem path the parser can load.
 *
 * 把 tree-sitter 的 wasm 资源引用（file URL、绝对路径或打包器 specifier）解析为解析器
 * 可加载的具体文件系统路径。
 *
 * @param asset - The asset reference to resolve / 待解析的资源引用
 * @returns An absolute filesystem path / 一个绝对文件系统路径
 */
const resolveWasm = (asset: string) => {
  if (asset.startsWith("file://")) return fileURLToPath(asset)
  if (asset.startsWith("/") || /^[a-z]:/i.test(asset)) return asset
  const url = new URL(asset, import.meta.url)
  return fileURLToPath(url)
}

/**
 * Flattens a command AST node into its meaningful tokens (command name and arguments),
 * dropping separators and redirections. This is the argv the permission scanner reasons about.
 *
 * 把一个命令 AST 节点拍平为有意义的 token（命令名与参数），丢弃分隔符和重定向。
 * 这就是权限扫描器所依据的 argv。
 *
 * @param node - The command AST node to flatten / 待拍平的命令 AST 节点
 * @returns The ordered list of token parts / 有序的 token 列表
 */
function parts(node: Node) {
  const out: Part[] = []
  for (let i = 0; i < node.childCount; i++) {
    const child = node.child(i)
    if (!child) continue
    if (child.type === "command_elements") {
      for (let j = 0; j < child.childCount; j++) {
        const item = child.child(j)
        if (!item || item.type === "command_argument_sep" || item.type === "redirection") continue
        out.push({ type: item.type, text: item.text })
      }
      continue
    }
    if (
      child.type !== "command_name" &&
      child.type !== "command_name_expr" &&
      child.type !== "word" &&
      child.type !== "string" &&
      child.type !== "raw_string" &&
      child.type !== "concatenation"
    ) {
      continue
    }
    out.push({ type: child.type, text: child.text })
  }
  return out
}

/**
 * Returns the source text of a command node, widening to the parent when the command is
 * part of a redirection so the permission prompt shows the full redirected statement.
 *
 * 返回命令节点的源文本；当命令属于一个重定向时，向上取父节点，使权限提示能展示完整的
 * 重定向语句。
 *
 * @param node - The command AST node / 命令 AST 节点
 * @returns The trimmed source text / 去除首尾空白的源文本
 */
function source(node: Node) {
  return (node.parent?.type === "redirected_statement" ? node.parent.text : node.text).trim()
}

/**
 * Collects every `command` node inside a parsed tree, so a compound line (pipes, `&&`,
 * subshells) is scanned command-by-command.
 *
 * 收集解析树中所有的 `command` 节点，使复合命令行（管道、`&&`、子 shell）能逐条命令扫描。
 *
 * @param node - The root (or any) AST node to search under / 要在其下搜索的根（或任意）AST 节点
 * @returns All descendant command nodes / 所有后代命令节点
 */
function commands(node: Node) {
  return node.descendantsOfType("command").filter((child): child is Node => Boolean(child))
}

// Returns true when `tokens` (the flat argv of a single command node) invokes
// an irreversible deletion — either a direct removal command (rm, remove-item,
// …) or a destructive git subcommand (git reset --hard, git clean -f, …).
// `ps` toggles PowerShell case-insensitive matching.
function isDelete(tokens: string[], ps: boolean) {
  if (tokens.length === 0) return false
  const head = ps ? tokens[0].toLowerCase() : tokens[0]
  if (DELETE_COMMANDS.has(head)) return true
  if (head === "git" && tokens.length >= 2) {
    const sub = tokens[1]
    const flags = GIT_DESTRUCTIVE.get(sub)
    if (!flags) return false
    if (flags.size === 0) return true
    return tokens.slice(2).some((tok) => flags.has(tok))
  }
  return false
}

/**
 * Strips a single matching pair of surrounding single or double quotes from a token.
 *
 * 去掉 token 外层一对匹配的单引号或双引号。
 *
 * @param text - The token text / token 文本
 * @returns The unquoted text, or the original if not quoted / 去引号后的文本，未加引号则原样返回
 */
function unquote(text: string) {
  if (text.length < 2) return text
  const first = text[0]
  const last = text[text.length - 1]
  if ((first === '"' || first === "'") && first === last) return text.slice(1, -1)
  return text
}

/**
 * Expands a leading `~` to the user's home directory, matching shell tilde expansion.
 *
 * 把开头的 `~` 展开为用户主目录，等价于 shell 的波浪号展开。
 *
 * @param text - A path token possibly starting with `~` / 可能以 `~` 开头的路径 token
 * @returns The path with `~` expanded / 展开 `~` 后的路径
 */
function home(text: string) {
  if (text === "~") return os.homedir()
  if (text.startsWith("~/") || text.startsWith("~\\")) return path.join(os.homedir(), text.slice(2))
  return text
}

/**
 * Reads an environment variable, matching case-insensitively on Windows where env var
 * names are not case-sensitive.
 *
 * 读取环境变量；在 Windows 上按大小写不敏感匹配，因为其环境变量名不区分大小写。
 *
 * @param key - The environment variable name / 环境变量名
 * @returns The value, or undefined if unset / 变量值，未设置则为 undefined
 */
function envValue(key: string) {
  if (process.platform !== "win32") return process.env[key]
  const name = Object.keys(process.env).find((item) => item.toLowerCase() === key.toLowerCase())
  return name ? process.env[name] : undefined
}

/**
 * Resolves shell-automatic variables (`HOME`, `PWD`, `PSHOME`) that are not present in
 * `process.env` but are meaningful during path expansion.
 *
 * 解析 shell 自动变量（`HOME`、`PWD`、`PSHOME`）——它们不在 `process.env` 中，但在路径展开时有意义。
 *
 * @param key - The variable name / 变量名
 * @param cwd - The current working directory, used for `PWD` / 当前工作目录，用于 `PWD`
 * @param shell - The shell executable path, used for `PSHOME` / shell 可执行文件路径，用于 `PSHOME`
 * @returns The resolved value, or undefined if not one of these / 解析出的值，非上述变量则为 undefined
 */
function auto(key: string, cwd: string, shell: string) {
  const name = key.toUpperCase()
  if (name === "HOME") return os.homedir()
  if (name === "PWD") return cwd
  if (name === "PSHOME") return path.dirname(shell)
}

/**
 * Expands a PowerShell path token: unquotes it, substitutes `${env:X}` / `$env:X` and
 * automatic variables, then applies tilde expansion. Used to resolve the real path a
 * command targets before a permission decision.
 *
 * 展开一个 PowerShell 路径 token：去引号，替换 `${env:X}` / `$env:X` 及自动变量，再做波浪号展开。
 * 用于在权限判定前解析命令实际指向的真实路径。
 *
 * @param text - The raw path token / 原始路径 token
 * @param cwd - Current working directory for `$PWD` / 用于 `$PWD` 的当前工作目录
 * @param shell - Shell executable path for `$PSHOME` / 用于 `$PSHOME` 的 shell 可执行文件路径
 * @returns The expanded path / 展开后的路径
 */
function expand(text: string, cwd: string, shell: string) {
  const out = unquote(text)
    .replace(/\$\{env:([^}]+)\}/gi, (_, key: string) => envValue(key) || "")
    .replace(/\$env:([A-Za-z_][A-Za-z0-9_]*)/gi, (_, key: string) => envValue(key) || "")
    .replace(/\$(HOME|PWD|PSHOME)(?=$|[\\/])/gi, (_, key: string) => auto(key, cwd, shell) || "")
  return home(out)
}

/**
 * Interprets a PowerShell provider-qualified path. Returns the bare path for the
 * `FileSystem::` provider (or unqualified paths), and undefined for non-filesystem
 * providers (for example `Registry::`, `Env:`) or drive-qualified specs that should not
 * be treated as filesystem paths.
 *
 * 解释 PowerShell 的 provider 限定路径。对 `FileSystem::` provider（或未限定路径）返回裸路径；
 * 对非文件系统 provider（如 `Registry::`、`Env:`）或不应视为文件系统路径的驱动器限定写法，返回 undefined。
 *
 * @param text - The possibly provider-qualified path / 可能带 provider 限定的路径
 * @returns The filesystem path, or undefined if not a filesystem path / 文件系统路径，非文件系统路径则为 undefined
 */
function provider(text: string) {
  const match = text.match(/^([A-Za-z]+)::(.*)$/)
  if (match) {
    if (match[1].toLowerCase() !== "filesystem") return
    return match[2]
  }
  const prefix = text.match(/^([A-Za-z]+):(.*)$/)
  if (!prefix) return text
  if (prefix[1].length === 1) return text
  return
}

/**
 * Detects whether a token contains dynamic content (command substitution, variable
 * expansion, subexpressions) whose value cannot be known statically. Such tokens are
 * skipped by the path scanner since their real target is unknowable before running.
 *
 * 检测 token 是否含有动态内容（命令替换、变量展开、子表达式），其值无法静态得知。
 * 此类 token 会被路径扫描器跳过，因为运行前无法确定其真实目标。
 *
 * @param text - The token text / token 文本
 * @param ps - Whether to use PowerShell rules / 是否使用 PowerShell 规则
 * @returns True if the token is dynamic / token 为动态内容时返回 true
 */
function dynamic(text: string, ps: boolean) {
  if (text.startsWith("(") || text.startsWith("@(")) return true
  if (text.includes("$(") || text.includes("${") || text.includes("`")) return true
  if (ps) return /\$(?!env:)/i.test(text)
  return text.includes("$")
}

/**
 * Returns the literal prefix of a token before any glob metacharacter (`?`, `*`, `[`).
 * A token that begins with a glob has no literal prefix and yields undefined.
 *
 * 返回 token 中出现任何 glob 元字符（`?`、`*`、`[`）之前的字面前缀。
 * 以 glob 开头的 token 没有字面前缀，返回 undefined。
 *
 * @param text - The token text / token 文本
 * @returns The literal prefix, the whole token if no glob, or undefined if it starts with a glob / 字面前缀；无 glob 则为整个 token；以 glob 开头则为 undefined
 */
function prefix(text: string) {
  const match = /[?*[]/.exec(text)
  if (!match) return text
  if (match.index === 0) return
  return text.slice(0, match.index)
}

/**
 * Extracts the path-valued arguments from a command's tokens, skipping the command name
 * and option flags. For bash, drops leading-dash flags (and `chmod`'s `+` modes); for
 * PowerShell, skips switches and only takes the value after a path-taking parameter.
 *
 * 从命令 token 中提取"值为路径"的参数，跳过命令名与选项标志。bash 会去掉以短横线开头的标志
 *（以及 `chmod` 的 `+` 模式）；PowerShell 会跳过开关参数，并只取"接受路径的参数"后面的值。
 *
 * @param list - The command's token parts / 命令的 token 列表
 * @param ps - Whether to use PowerShell rules / 是否使用 PowerShell 规则
 * @returns The path argument strings / 路径参数字符串列表
 */
function pathArgs(list: Part[], ps: boolean) {
  if (!ps) {
    return list
      .slice(1)
      .filter((item) => !item.text.startsWith("-") && !(list[0]?.text === "chmod" && item.text.startsWith("+")))
      .map((item) => item.text)
  }

  const out: string[] = []
  let want = false
  for (const item of list.slice(1)) {
    if (want) {
      out.push(item.text)
      want = false
      continue
    }
    if (item.type === "command_parameter") {
      const flag = item.text.toLowerCase()
      if (SWITCHES.has(flag)) continue
      want = FLAGS.has(flag)
      continue
    }
    out.push(item.text)
  }
  return out
}

/**
 * Clamps live-preview text to `MAX_METADATA_LENGTH`, keeping the most recent tail so the
 * UI preview stays bounded while still showing the latest output.
 *
 * 把实时预览文本限制在 `MAX_METADATA_LENGTH` 以内，保留最新的尾部，使 UI 预览体积有界，
 * 同时仍展示最新输出。
 *
 * @param text - The accumulated preview text / 累积的预览文本
 * @returns The clamped preview text / 裁剪后的预览文本
 */
function preview(text: string) {
  if (text.length <= MAX_METADATA_LENGTH) return text
  return "...\n\n" + text.slice(-MAX_METADATA_LENGTH)
}

/** Detects error-like keywords in truncated output, triggering a head+tail preview / 在被截断输出中检测类错误关键词，触发"头+尾"预览 */
const ERROR_PATTERN = /error|exception|failed|fatal|traceback|panic|exit code/i
/** Byte budget for the head slice when errors warrant showing both head and tail / 出错需同时展示头尾时，头部切片的字节预算 */
const HEAD_BYTES = Math.floor(Truncate.MAX_BYTES * 0.7)
/** Line budget for the head slice when errors warrant showing both head and tail / 出错需同时展示头尾时，头部切片的行数预算 */
const HEAD_LINES = Math.floor(Truncate.MAX_LINES * 0.7)

/**
 * Takes the leading portion of text bounded by both a line count and a byte count,
 * stopping at whichever limit is reached first.
 *
 * 取文本开头的部分，同时受行数与字节数两个上限约束，任一先达到即停止。
 *
 * @param text - The full text / 完整文本
 * @param maxLines - Maximum number of lines to keep / 保留的最大行数
 * @param maxBytes - Maximum number of UTF-8 bytes to keep / 保留的最大 UTF-8 字节数
 * @returns The head slice as a string / 头部切片字符串
 */
function head(text: string, maxLines: number, maxBytes: number): string {
  const lines = text.split("\n")
  const out: string[] = []
  let bytes = 0
  for (let i = 0; i < lines.length && out.length < maxLines; i++) {
    const size = Buffer.byteLength(lines[i], "utf-8") + (i > 0 ? 1 : 0)
    if (bytes + size > maxBytes) break
    out.push(lines[i])
    bytes += size
  }
  return out.join("\n")
}

/**
 * Takes the trailing portion of text bounded by both a line count and a byte count. When
 * even a single line exceeds the byte budget, it keeps a UTF-8-safe suffix of that line.
 * Reports whether any content was cut.
 *
 * 取文本末尾的部分，同时受行数与字节数两个上限约束。当单独一行都超出字节预算时，会保留该行
 * 一个 UTF-8 安全的后缀。并报告是否发生了裁剪。
 *
 * @param text - The full text / 完整文本
 * @param maxLines - Maximum number of lines to keep / 保留的最大行数
 * @param maxBytes - Maximum number of UTF-8 bytes to keep / 保留的最大 UTF-8 字节数
 * @returns An object with the tail `text` and a `cut` flag / 含尾部 `text` 与 `cut` 标志的对象
 */
function tail(text: string, maxLines: number, maxBytes: number) {
  const lines = text.split("\n")
  if (lines.length <= maxLines && Buffer.byteLength(text, "utf-8") <= maxBytes) {
    return {
      text,
      cut: false,
    }
  }

  const out: string[] = []
  let bytes = 0
  for (let i = lines.length - 1; i >= 0 && out.length < maxLines; i--) {
    const size = Buffer.byteLength(lines[i], "utf-8") + (out.length > 0 ? 1 : 0)
    if (bytes + size > maxBytes) {
      if (out.length === 0) {
        const buf = Buffer.from(lines[i], "utf-8")
        let start = buf.length - maxBytes
        if (start < 0) start = 0
        while (start < buf.length && (buf[start] & 0xc0) === 0x80) start++
        out.unshift(buf.subarray(start).toString("utf-8"))
      }
      break
    }
    out.unshift(lines[i])
    bytes += size
  }
  return {
    text: out.join("\n"),
    cut: true,
  }
}

/**
 * Parses a command string into a tree-sitter AST root node, choosing the PowerShell or
 * bash grammar based on `ps`.
 *
 * 把命令字符串解析为 tree-sitter AST 根节点，按 `ps` 选择 PowerShell 或 bash 语法。
 *
 * @param command - The raw command text / 原始命令文本
 * @param ps - Whether to parse as PowerShell / 是否按 PowerShell 解析
 * @returns An effect resolving to the AST root node / 解析为 AST 根节点的 effect
 * @throws {Error} When the parser fails to produce a tree / 解析器无法生成语法树时
 */
const parse = Effect.fn("BashTool.parse")(function* (command: string, ps: boolean) {
  const tree = yield* Effect.promise(() => parser().then((p) => (ps ? p.ps : p.bash).parse(command)))
  if (!tree) throw new Error("Failed to parse command")
  return tree.rootNode
})

/**
 * Requests the permissions implied by a command scan: an `external_directory` prompt for
 * any out-of-workspace directories it touches, and a `bash` prompt for the command itself.
 *
 * 根据命令扫描结果申请对应权限：对触碰到的工作区外目录发起 `external_directory` 询问，
 * 对命令本身发起 `bash` 询问。
 *
 * @param ctx - The tool execution context providing `ask` / 提供 `ask` 的工具执行上下文
 * @param scan - The static scan result / 静态扫描结果
 * @returns An effect that resolves once permission is granted / 权限通过后解析的 effect
 */
const ask = Effect.fn("BashTool.ask")(function* (ctx: Tool.Context, scan: Scan) {
  if (scan.dirs.size > 0) {
    const globs = Array.from(scan.dirs).map((dir) => {
      if (process.platform === "win32") return AppFileSystem.normalizePathPattern(path.join(dir, "*"))
      return path.join(dir, "*")
    })
    yield* ctx.ask({
      permission: "external_directory",
      patterns: globs,
      always: globs,
      metadata: {},
    })
  }

  if (scan.patterns.size === 0) return
  yield* ctx.ask({
    permission: "bash",
    patterns: Array.from(scan.patterns),
    always: Array.from(scan.always),
    metadata: {},
  })
})

// Secondary confirmation for irreversible deletion commands. Uses its own
// permission type ("bash_delete"), which the Permission layer flags as
// forced-ask: no `allow` rule (not even a broad `"*": allow`) can silently
// pre-approve it — only an explicit `deny` blocks. `always` is empty because
// a persisted "allow all deletes" rule is exactly what forced-ask exists to
// prevent. The delete UI shows the full command, so this ask FULLY replaces
// the regular bash/external_directory prompts when it fires (see the caller
// below) — deletion is authorized in a single, unambiguous confirmation.
const askDelete = Effect.fn("BashTool.askDelete")(function* (ctx: Tool.Context, scan: Scan, command: string) {
  const patterns = Array.from(scan.deletes)
  yield* ctx.ask({
    permission: "bash_delete",
    patterns,
    always: [],
    metadata: { command, deletes: patterns },
  })
})

/**
 * Builds the `ChildProcess` spec for running a command, adapting to the target shell:
 * PowerShell on Windows runs via `-Command` with a UTF-8 prefix; `cmd` gets a UTF-8
 * prefix; POSIX shells run the command through the shell with `detached` set so the whole
 * process group can be killed on abort.
 *
 * 构建运行命令的 `ChildProcess` 规格，并适配目标 shell：Windows 上的 PowerShell 通过
 * `-Command` 加 UTF-8 前缀运行；`cmd` 加 UTF-8 前缀；POSIX shell 则经 shell 运行命令并设置
 * `detached`，以便取消时能杀掉整个进程组。
 *
 * @param shell - The shell executable / shell 可执行文件
 * @param name - The shell's short name / shell 的短名
 * @param command - The command text to run / 要运行的命令文本
 * @param cwd - Working directory / 工作目录
 * @param env - Environment variables / 环境变量
 * @returns A ChildProcess spec ready to spawn / 可用于 spawn 的 ChildProcess 规格
 */
function cmd(shell: string, name: string, command: string, cwd: string, env: NodeJS.ProcessEnv) {
  if (process.platform === "win32" && PS.has(name)) {
    const prefixed = `${Shell.POWERSHELL_UTF8_PREFIX}${command}`
    return ChildProcess.make(shell, ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", prefixed], {
      cwd,
      env,
      stdin: "ignore",
      detached: false,
    })
  }

  const finalCommand =
    process.platform === "win32" && name === "cmd" ? `${Shell.CMD_UTF8_PREFIX}${command}` : command

  return ChildProcess.make(finalCommand, [], {
    shell,
    cwd,
    env,
    stdin: "ignore",
    detached: process.platform !== "win32",
  })
}

/**
 * Lazily initializes the tree-sitter parsers for bash and PowerShell, loading their wasm
 * grammars once and caching the result. Deferred so the wasm cost is paid only when a
 * command is actually parsed.
 *
 * 惰性初始化 bash 与 PowerShell 的 tree-sitter 解析器，只加载一次它们的 wasm 语法并缓存结果。
 * 延迟加载，使 wasm 的开销仅在真正解析命令时才付出。
 *
 * @returns An object with the initialized `bash` and `ps` parsers / 含已初始化 `bash` 与 `ps` 解析器的对象
 */
const parser = lazy(async () => {
  const { Parser } = await import("web-tree-sitter")
  const { default: treeWasm } = await import("web-tree-sitter/tree-sitter.wasm" as string, {
    with: { type: "wasm" },
  })
  const treePath = resolveWasm(treeWasm)
  await Parser.init({
    locateFile() {
      return treePath
    },
  })
  const { default: bashWasm } = await import("tree-sitter-bash/tree-sitter-bash.wasm" as string, {
    with: { type: "wasm" },
  })
  const { default: psWasm } = await import("tree-sitter-powershell/tree-sitter-powershell.wasm" as string, {
    with: { type: "wasm" },
  })
  const bashPath = resolveWasm(bashWasm)
  const psPath = resolveWasm(psWasm)
  const [bashLanguage, psLanguage] = await Promise.all([Language.load(bashPath), Language.load(psPath)])
  const bash = new Parser()
  bash.setLanguage(bashLanguage)
  const ps = new Parser()
  ps.setLanguage(psLanguage)
  return { bash, ps }
})

/**
 * The `bash` tool definition. Wires up the child-process spawner, filesystem, truncation,
 * and plugin services, then returns a factory that produces the tool's shell-aware
 * description and `execute` implementation.
 *
 * `bash` 工具定义。接入子进程 spawner、文件系统、截断与插件服务，然后返回一个工厂，
 * 产出该工具"感知 shell"的描述与 `execute` 实现。
 */
// TODO: we may wanna rename this tool so it works better on other shells
export const BashTool = Tool.define(
  "bash",
  Effect.gen(function* () {
    const spawner = yield* ChildProcessSpawner
    const fs = yield* AppFileSystem.Service
    const trunc = yield* Truncate.Service
    const plugin = yield* Plugin.Service

    /**
     * Converts a POSIX-style path to a Windows path by shelling out to `cygpath`, used
     * when a Cygwin/MSYS-style shell is active on Windows. Fails soft, returning undefined.
     *
     * 通过调用 `cygpath` 把 POSIX 风格路径转换为 Windows 路径，用于 Windows 上使用
     * Cygwin/MSYS 风格 shell 的场景。失败时返回 undefined（软失败）。
     *
     * @param shell - The shell executable to run `cygpath` in / 运行 `cygpath` 的 shell 可执行文件
     * @param text - The POSIX path to convert / 待转换的 POSIX 路径
     * @returns An effect resolving to the Windows path, or undefined / 解析为 Windows 路径或 undefined 的 effect
     */
    const cygpath = Effect.fn("BashTool.cygpath")(function* (shell: string, text: string) {
      const lines = yield* spawner
        .lines(ChildProcess.make(shell, ["-lc", 'cygpath -w -- "$1"', "_", text]))
        .pipe(Effect.catch(() => Effect.succeed([] as string[])))
      const file = lines[0]?.trim()
      if (!file) return
      return AppFileSystem.normalizePath(file)
    })

    /**
     * Resolves a path token to an absolute filesystem path against a root directory,
     * handling Windows quirks (POSIX-to-Windows conversion via cygpath, path normalization).
     *
     * 相对某个根目录，把路径 token 解析为绝对文件系统路径，并处理 Windows 的特殊情况
     *（经 cygpath 做 POSIX 到 Windows 的转换、路径归一化）。
     *
     * @param text - The path token to resolve / 待解析的路径 token
     * @param root - The base directory to resolve against / 用于解析的基准目录
     * @param shell - The active shell, for platform-specific handling / 当前 shell，用于平台相关处理
     * @returns An effect resolving to the absolute path / 解析为绝对路径的 effect
     */
    const resolvePath = Effect.fn("BashTool.resolvePath")(function* (text: string, root: string, shell: string) {
      if (process.platform === "win32") {
        if (Shell.posix(shell) && text.startsWith("/") && AppFileSystem.windowsPath(text) === text) {
          const file = yield* cygpath(shell, text)
          if (file) return file
        }
        return AppFileSystem.normalizePath(path.resolve(root, AppFileSystem.windowsPath(text)))
      }
      return path.resolve(root, text)
    })

    /**
     * Resolves a single command argument to the absolute path it targets, or undefined
     * when the argument has no static literal path (a glob-only, dynamic, or non-filesystem
     * provider token). Combines expansion, glob-prefix extraction, and path resolution.
     *
     * 把单个命令参数解析为它指向的绝对路径；当参数没有静态字面路径（纯 glob、动态内容、或
     * 非文件系统 provider 的 token）时返回 undefined。综合了变量展开、glob 前缀提取与路径解析。
     *
     * @param arg - The raw argument token / 原始参数 token
     * @param cwd - Current working directory / 当前工作目录
     * @param ps - Whether PowerShell rules apply / 是否适用 PowerShell 规则
     * @param shell - The active shell / 当前 shell
     * @returns An effect resolving to the absolute path, or undefined / 解析为绝对路径或 undefined 的 effect
     */
    const argPath = Effect.fn("BashTool.argPath")(function* (arg: string, cwd: string, ps: boolean, shell: string) {
      const text = ps ? expand(arg, cwd, shell) : home(unquote(arg))
      const file = text && prefix(text)
      if (!file || dynamic(file, ps)) return
      const next = ps ? provider(file) : file
      if (!next) return
      return yield* resolvePath(next, cwd, shell)
    })

    /**
     * Statically scans every command in a parsed line to build the permission `Scan`:
     * out-of-workspace directories touched by file commands, the command patterns to ask
     * about, prefix-based "always allow" patterns, and any irreversible deletions.
     *
     * 静态扫描一行命令中的每条子命令，构建权限 `Scan`：文件类命令触碰的工作区外目录、
     * 需询问的命令模式、基于前缀的"总是允许"模式，以及任何不可逆删除。
     *
     * @param root - The parsed AST root node / 解析后的 AST 根节点
     * @param cwd - Current working directory / 当前工作目录
     * @param ps - Whether PowerShell rules apply / 是否适用 PowerShell 规则
     * @param shell - The active shell / 当前 shell
     * @returns An effect resolving to the completed scan / 解析为完成后 Scan 的 effect
     */
    const collect = Effect.fn("BashTool.collect")(function* (root: Node, cwd: string, ps: boolean, shell: string) {
      const scan: Scan = {
        dirs: new Set<string>(),
        patterns: new Set<string>(),
        always: new Set<string>(),
        deletes: new Set<string>(),
      }

      // Scan each command in the line separately (pipes, &&, subshells produce multiple).
      // 逐条扫描命令行中的每个命令（管道、&&、子 shell 会产生多条）。
      for (const node of commands(root)) {
        const command = parts(node)
        const tokens = command.map((item) => item.text)
        const cmd = ps ? tokens[0]?.toLowerCase() : tokens[0]

        // File-touching commands: resolve their path args and record any that fall
        // outside the workspace, so we can ask for external_directory permission.
        // 触碰文件的命令：解析其路径参数，记录落在工作区之外的路径，以便申请 external_directory 权限。
        if (cmd && FILES.has(cmd)) {
          for (const arg of pathArgs(command, ps)) {
            const resolved = yield* argPath(arg, cwd, ps, shell)
            log.info("resolved path", { arg, resolved })
            if (!resolved || Instance.containsPath(resolved)) continue
            const dir = (yield* fs.isDir(resolved)) ? resolved : path.dirname(resolved)
            scan.dirs.add(dir)
          }
        }

        // Record the command itself for a bash prompt (except bare cd-style commands),
        // plus a prefix-based "always allow this kind of command" pattern.
        // 记录命令本身以发起 bash 询问（纯 cd 类命令除外），并附加一个基于前缀的
        // "总是允许这类命令"模式。
        if (tokens.length && (!cmd || !CWD.has(cmd))) {
          scan.patterns.add(source(node))
          scan.always.add(BashArity.prefix(tokens).join(" ") + " *")
        }

        // Flag irreversible deletions for the stronger askDelete confirmation.
        // 标记不可逆删除，交给更强的 askDelete 确认。
        if (isDelete(tokens, ps)) scan.deletes.add(source(node))
      }

      return scan
    })

    /**
     * Builds the environment for the child process: the current process env, a Windows
     * UTF-8 fix for Python child output, and any variables contributed by the `shell.env`
     * plugin hook.
     *
     * 构建子进程的环境变量：当前进程环境、针对 Python 子进程输出的 Windows UTF-8 修正，
     * 以及 `shell.env` 插件钩子贡献的任何变量。
     *
     * @param ctx - The tool execution context / 工具执行上下文
     * @param cwd - Working directory passed to the plugin hook / 传给插件钩子的工作目录
     * @returns An effect resolving to the merged environment / 解析为合并后环境的 effect
     */
    const shellEnv = Effect.fn("BashTool.shellEnv")(function* (ctx: Tool.Context, cwd: string) {
      const extra = yield* plugin.trigger(
        "shell.env",
        { cwd, sessionID: ctx.sessionID, callID: ctx.callID },
        { env: {} },
      )
      return {
        ...process.env,
        // Python ignores the console code page when stdout is a pipe and falls
        // back to the ANSI code page (GBK on zh-CN), producing mojibake. Force
        // UTF-8 for child Python processes on Windows.
        ...(process.platform === "win32" ? { PYTHONIOENCODING: "utf-8" } : {}),
        ...extra.env,
      }
    })

    /**
     * Spawns the child process and drives it to completion: streams combined output to the
     * UI live, spills oversized output to a truncation file, and races process exit against
     * the abort signal and the timeout, killing the child if either fires. Finally assembles
     * the model-facing output (with truncation notices and a head+tail view when errors are
     * present) and the result metadata.
     *
     * 启动子进程并驱动其运行到结束：把合并输出实时流式传给 UI，超大输出溢写到截断文件，
     * 并让"进程退出"与"取消信号"和"超时"竞速，任一触发就杀掉子进程。最后组装面向模型的输出
     *（含截断提示，出错时给出"头+尾"视图）与结果元数据。
     *
     * @param input - Shell, command, cwd, env, timeout, and description / shell、命令、cwd、env、超时与描述
     * @param ctx - The tool execution context (abort signal, metadata callback) / 工具执行上下文（取消信号、metadata 回调）
     * @returns An effect resolving to the tool's `ExecuteResult` / 解析为工具 `ExecuteResult` 的 effect
     */
    const run = Effect.fn("BashTool.run")(function* (
      input: {
        shell: string
        name: string
        command: string
        cwd: string
        env: NodeJS.ProcessEnv
        timeout: number
        description: string
      },
      ctx: Tool.Context,
    ) {
      const bytes = Truncate.MAX_BYTES // Inline output byte budget before spilling to a file / 溢写到文件前的内联输出字节预算
      const lines = Truncate.MAX_LINES // Inline output line budget / 内联输出行数预算
      const keep = bytes * 2 // Rolling buffer size: keep roughly the last 2x budget of raw output / 滚动缓冲大小：约保留最近 2 倍预算的原始输出
      let full = "" // Accumulated inline output while still under the byte budget / 仍在字节预算内时累积的内联输出
      let last = "" // Latest clamped preview text shown in the UI / UI 中展示的最新裁剪预览文本
      const list: Chunk[] = [] // Rolling list of recent chunks (older ones dropped) / 最近输出块的滚动列表（旧的被丢弃）
      let used = 0 // Current byte size held in `list` / `list` 当前占用的字节数
      let file = "" // Truncation file path once output spills to disk / 输出溢写到磁盘后的截断文件路径
      let sink: ReturnType<typeof createWriteStream> | undefined // Write stream to the truncation file / 指向截断文件的写入流
      let cut = false // Whether any output was truncated / 是否发生了输出截断
      let expired = false // Whether the command was killed by timeout / 命令是否因超时被杀
      let aborted = false // Whether the command was killed by user abort / 命令是否因用户取消被杀

      // Push an empty preview immediately so the UI renders the command as "running"
      // before any output arrives.
      // 先推一个空预览，让 UI 在任何输出到来之前就把命令渲染成"运行中"。
      yield* ctx.metadata({
        metadata: {
          output: "",
          description: input.description,
        },
      })

      // Run the child process inside a scope and resolve to its exit code (or null when it
      // was killed by abort/timeout). Everything spawned here is torn down when the scope ends.
      // 在一个 scope 内运行子进程，解析出退出码（被取消/超时杀掉时为 null）。这里派生的一切
      // 都会在 scope 结束时被清理。
      const code: number | null = yield* Effect.scoped(
        Effect.gen(function* () {
          // Spawn the child process; the enclosing scope guarantees it is cleaned up.
          // 启动子进程；外层 scope 保证它会被清理。
          const handle = yield* spawner.spawn(cmd(input.shell, input.name, input.command, input.cwd, input.env))

          // Consume the combined stdout+stderr stream in a forked fiber, updating the UI
          // preview on every chunk. Runs concurrently with the exit/abort/timeout race below.
          // 在一个 fork 出的 fiber 中消费合并的 stdout+stderr 流，每来一块就更新 UI 预览。
          // 与下面的 退出/取消/超时 竞速并发运行。
          yield* Effect.forkScoped(
            Stream.runForEach(Stream.decodeText(handle.all), (chunk) => {
              const size = Buffer.byteLength(chunk, "utf-8")
              list.push({ text: chunk, size })
              used += size
              // Drop oldest chunks once the rolling buffer exceeds `keep`; mark truncated.
              // 一旦滚动缓冲超过 `keep`，丢弃最旧的块并标记为已截断。
              while (used > keep && list.length > 1) {
                const item = list.shift()
                if (!item) break
                used -= item.size
                cut = true
              }

              last = preview(last + chunk)

              // Once spilling to a file, append there; otherwise keep accumulating inline
              // until the byte budget is exceeded, then open the truncation file and switch.
              // 一旦已在溢写文件，就追加到文件；否则先内联累积，直到超出字节预算，再打开截断
              // 文件并切换到文件写入。
              if (file) {
                sink?.write(chunk)
              } else {
                full += chunk
                // Inline buffer exceeded the byte budget: spill everything accumulated so
                // far to a truncation file, switch subsequent writes to that file's stream,
                // mark truncated, then refresh the UI preview.
                // 内联缓冲超过字节预算：把目前累积的内容溢写到截断文件，后续写入切到该文件的流，
                // 标记为已截断，然后刷新 UI 预览。
                if (Buffer.byteLength(full, "utf-8") > bytes) {
                  return trunc.write(full).pipe(
                    Effect.andThen((next) =>
                      Effect.sync(() => {
                        file = next
                        cut = true
                        sink = createWriteStream(next, { flags: "a" })
                        full = ""
                      }),
                    ),
                    Effect.andThen(
                      ctx.metadata({
                        metadata: {
                          output: last,
                          description: input.description,
                        },
                      }),
                    ),
                  )
                }
              }

              return ctx.metadata({
                metadata: {
                  output: last,
                  description: input.description,
                },
              })
            }),
          )

          // Bridge the DOM-style abort signal (from ctx.abort, wired to the run's Fiber
          // interruption) into an effect that resolves when the user cancels. This is how
          // a loop cancellation reaches into and stops the in-flight child process.
          // 把 DOM 风格的取消信号（来自 ctx.abort，它连着本次运行的 Fiber 中断）桥接成一个
          // "用户取消时才解析"的 effect。这正是 loop 取消如何深入并停止在途子进程的方式。
          const abort = Effect.callback<void>((resume) => {
            if (ctx.abort.aborted) return resume(Effect.void)
            const handler = () => resume(Effect.void)
            ctx.abort.addEventListener("abort", handler, { once: true })
            return Effect.sync(() => ctx.abort.removeEventListener("abort", handler))
          })

          const timeout = Effect.sleep(`${input.timeout + 100} millis`)

          // Race process exit against user abort and the timeout; whichever wins decides
          // how the command ends.
          // 让"进程退出"与"用户取消"和"超时"竞速；谁先胜出就决定命令如何结束。
          const exit = yield* Effect.raceAll([
            handle.exitCode.pipe(Effect.map((code) => ({ kind: "exit" as const, code }))),
            abort.pipe(Effect.map(() => ({ kind: "abort" as const, code: null }))),
            timeout.pipe(Effect.map(() => ({ kind: "timeout" as const, code: null }))),
          ])

          // On abort or timeout, actively kill the child (SIGTERM, then SIGKILL after 3s).
          // This is the "cooperative kill" for a signal-aware operation discussed earlier.
          // 取消或超时时，主动杀掉子进程（先 SIGTERM，3 秒后 SIGKILL）。这正是之前讨论的
          // "认信号的操作"所对应的协作式 kill。
          if (exit.kind === "abort") {
            aborted = true
            yield* handle.kill({ forceKillAfter: "3 seconds" }).pipe(Effect.orDie)
          }
          if (exit.kind === "timeout") {
            expired = true
            yield* handle.kill({ forceKillAfter: "3 seconds" }).pipe(Effect.orDie)
          }

          return exit.kind === "exit" ? exit.code : null
        }),
      ).pipe(Effect.orDie)

      // Build trailing metadata notices (timeout / user abort) appended to the output.
      // 构建追加到输出末尾的元信息提示（超时 / 用户取消）。
      const meta: string[] = []
      if (expired) {
        meta.push(
          `bash tool terminated command after exceeding timeout ${input.timeout} ms. If this command is expected to take longer and is not waiting for interactive input, retry with a larger timeout value in milliseconds.`,
        )
      }
      if (aborted) meta.push("User aborted the command")
      // Rebuild the retained output from the rolling buffer, then take a bounded tail slice.
      // If that tail itself had to cut content and nothing was spilled yet, write the full
      // raw output to a truncation file so the model can still reach it.
      // 从滚动缓冲重建保留下来的输出，再取一段有界的尾部切片。若该尾部本身发生了裁剪、且此前
      // 还没溢写过，就把完整原始输出写入截断文件，让模型仍能取到。
      const raw = list.map((item) => item.text).join("")
      const end = tail(raw, lines, bytes)
      if (end.cut) cut = true
      if (!file && end.cut) {
        file = yield* trunc.write(raw)
      }

      // Token-efficient post-cleanse: RTK-style ANSI strip / progress fold /
      // secret redact / long-line elide. Only applied when no tool storage is
      // involved — once the output spills to a truncation file, the on-disk
      // archive stays raw and cleaning is skipped to keep the inline preview
      // consistent with the archive.
      // 省 token 的后处理清洗：去 ANSI 转义 / 折叠进度条 / 脱敏密钥 / 省略超长行。仅在没有落盘
      // 时应用——一旦输出溢写到截断文件，磁盘存档保持原始、跳过清洗，以保证内联预览与存档一致。
      const cleaned =
        !file && Flag.MIMOCODE_EXPERIMENTAL_TOKEN_EFFICIENCY
          ? BashTokenEfficient.clean(end.text, { command: input.command })
          : null
      if (cleaned && cleaned.bytesOut < cleaned.bytesIn) {
        log.info("bash output cleaned", {
          bytesIn: cleaned.bytesIn,
          bytesOut: cleaned.bytesOut,
          saved: cleaned.bytesIn - cleaned.bytesOut,
        })
      }

      // Heuristic (shape-based) pipeline runs AFTER the common pipeline and
      // only when both flags are on. Same never-worse contract — a shape that
      // doesn't shrink the bytes is discarded.
      // 基于"形状"的启发式管线在通用清洗之后运行，且仅当两个开关都打开时。同样遵循"绝不更差"
      // 约定——如果某种形状没能减小字节数，就丢弃它。
      const heuristic =
        !file &&
        Flag.MIMOCODE_EXPERIMENTAL_TOKEN_EFFICIENCY &&
        Flag.MIMOCODE_EXPERIMENTAL_TOKEN_EFFICIENCY_HEURISTIC
          ? BashTokenEfficientHeuristic.cleanHeuristic(cleaned?.text ?? end.text, { command: input.command })
          : null
      if (heuristic && heuristic.bytesOut < heuristic.bytesIn) {
        log.info("bash output heuristic cleaned", {
          shape: heuristic.shape,
          bytesIn: heuristic.bytesIn,
          bytesOut: heuristic.bytesOut,
          saved: heuristic.bytesIn - heuristic.bytesOut,
        })
      }

      let output = heuristic?.text ?? cleaned?.text ?? end.text
      if (!output) output = "(no output)"

      // When output was truncated to a file, decide how to present the inline slice:
      // if the tail shows errors, prepend a head slice too (errors often reference earlier
      // context); otherwise just show the tail with a pointer to the full file.
      // 当输出被截断到文件时，决定内联切片如何呈现：若尾部含错误，则同时前置一段头部切片
      //（错误常引用更早的上下文）；否则只展示尾部，并附上完整文件的指引。
      if (cut && file) {
        // Check if tail contains error patterns — if so, prepend head for context
        const tailScan = end.text.length > 2048 ? end.text.slice(-2048) : end.text
        const hasErrors = ERROR_PATTERN.test(tailScan)
        if (hasErrors) {
          let fileContent: string | undefined
          try {
            fileContent = readFileSync(file, "utf-8")
          } catch {
            fileContent = undefined
          }
          if (fileContent) {
            const headText = head(fileContent, HEAD_LINES, HEAD_BYTES)
            output = `...output truncated (head+tail shown due to errors)...\n\nFull output saved to: ${file}\n\n${headText}\n\n...middle omitted...\n\n${end.text}`
          } else {
            output = `...output truncated...\n\nFull output saved to: ${file}\n\n` + output
          }
        } else {
          output = `...output truncated...\n\nFull output saved to: ${file}\n\n` + output
        }
      }

      if (meta.length > 0) {
        output += "\n\n<bash_metadata>\n" + meta.join("\n") + "\n</bash_metadata>"
      }
      if (sink) {
        const stream = sink
        yield* Effect.promise(
          () =>
            new Promise<void>((resolve) => {
              stream.end(() => resolve())
              stream.on("error", () => resolve())
            }),
        )
      }

      // Assemble the final result: `metadata` feeds the UI (live preview + exit code +
      // truncation info), while `output` is the full text handed back to the model.
      // 组装最终结果：`metadata` 供 UI 使用（实时预览 + 退出码 + 截断信息），`output` 则是交回
      // 给模型的完整文本。
      return {
        title: input.description,
        metadata: {
          output: last || preview(output),
          exit: code,
          description: input.description,
          truncated: cut,
          ...(cut && file ? { outputPath: file } : {}),
        },
        output,
      }
    })

    // Factory returning the tool definition. Picks the platform's shell, renders the
    // shell-specific description, and defines `execute`: resolve cwd, parse + scan the
    // command, request permissions (delete confirmation or the regular prompts), then run
    // it interactively or via the streaming `run` above.
    // 返回工具定义的工厂。选择平台 shell，渲染与 shell 相关的描述，并定义 `execute`：
    // 解析 cwd、解析并扫描命令、申请权限（删除确认或常规询问），然后以交互模式或经上面的
    // 流式 `run` 执行。
    return () =>
      Effect.sync(() => {
        const shell = Shell.acceptable()
        const name = Shell.name(shell)
        const chain =
          name === "powershell"
            ? "If the commands depend on each other and must run sequentially, avoid '&&' in this shell because Windows PowerShell 5.1 does not support it. Use PowerShell conditionals such as `cmd1; if ($?) { cmd2 }` when later commands must depend on earlier success."
            : "If the commands depend on each other and must run sequentially, use a single Bash call with '&&' to chain them together (e.g., `git add . && git commit -m \"message\" && git push`). For instance, if one operation must complete before another starts (like mkdir before cp, Write before Bash for git operations, or git add before git commit), run these operations sequentially instead."
        log.info("bash tool using shell", { shell })

        return {
          description: DESCRIPTION.replaceAll("${directory}", Instance.directory)
            .replaceAll("${os}", process.platform)
            .replaceAll("${shell}", name)
            .replaceAll("${chaining}", chain)
            .replaceAll("${maxLines}", String(Truncate.MAX_LINES))
            .replaceAll("${maxBytes}", String(Truncate.MAX_BYTES)),
          parameters: Parameters,
          execute: (params: z.infer<typeof Parameters>, ctx: Tool.Context) =>
            Effect.gen(function* () {
              // Resolve the working directory: an explicit workdir, else the session's cwd.
              // 解析工作目录：显式的 workdir，否则用会话的 cwd。
              const effectiveCwd = SessionCwd.get(ctx.sessionID)
              const cwd = params.workdir
                ? yield* resolvePath(params.workdir, effectiveCwd, shell)
                : effectiveCwd
              if (params.timeout !== undefined && params.timeout < 0) {
                throw new Error(`Invalid timeout value: ${params.timeout}. Timeout must be a positive number.`)
              }
              const timeout = params.timeout ?? DEFAULT_TIMEOUT
              const ps = PS.has(name)
              // Parse the command into an AST and statically scan it for permission needs.
              // 把命令解析为 AST，并静态扫描出它的权限需求。
              const root = yield* parse(params.command, ps)
              const scan = yield* collect(root, cwd, ps, shell)
              // A cwd outside the workspace is itself an external directory to ask about.
              // 位于工作区之外的 cwd 本身就是一个需要询问的外部目录。
              if (!Instance.containsPath(cwd)) scan.dirs.add(cwd)
              // Delete-containing commands are authorized by askDelete alone —
              // the delete UI shows the full command (including any external
              // paths it touches), so a separate bash/external_directory
              // prompt would just be a second confirmation of the same thing.
              // MIMOCODE_AUTO_APPROVE_DELETE trusts deletes and falls back to
              // the regular ask (where a `bash: deny` rule still blocks).
              if (scan.deletes.size > 0 && !Flag.MIMOCODE_AUTO_APPROVE_DELETE) {
                yield* askDelete(ctx, scan, params.command)
              } else {
                yield* ask(ctx, scan)
              }

              // Interactive mode: hand terminal to user for direct interaction
              // 交互模式：把终端交给用户直接交互（如输入密码、y/N 确认）。
              if (params.interactive) {
                const env = yield* shellEnv(ctx, cwd)
                yield* ctx.metadata({
                  metadata: {
                    output: "(waiting for user interaction...)",
                    description: params.description,
                  },
                })
                const interactiveResult = yield* Effect.tryPromise(() =>
                  BashInteractive.request({
                    command: params.command,
                    cwd,
                    env: env as Record<string, string>,
                    description: params.description,
                  }),
                ).pipe(Effect.orDie)
                return {
                  title: params.description,
                  metadata: {
                    output: interactiveResult.output || "(interactive command completed)",
                    exit: interactiveResult.exitCode,
                    description: params.description,
                    truncated: false,
                  },
                  output:
                    interactiveResult.output ||
                    `(interactive command completed with exit code ${interactiveResult.exitCode})`,
                }
              }

              // Non-interactive path: execute via the streaming `run` above and return its
              // result (spawn + live output + abort/timeout race + truncation).
              // 非交互路径：经上面的流式 `run` 执行并返回其结果（spawn + 实时输出 + 取消/超时竞速 + 截断）。
              return yield* run(
                {
                  shell,
                  name,
                  command: params.command,
                  cwd,
                  env: yield* shellEnv(ctx, cwd),
                  timeout,
                  description: params.description,
                },
                ctx,
              )
            }),
        }
      })
  }),
)
