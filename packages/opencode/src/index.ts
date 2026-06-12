// ============================================================================
// MiMoCode CLI 入口文件
// `bun run dev` 最终执行的就是这个文件:用 yargs 搭建命令行,分发到各子命令。
// 不带参数时默认启动 TUI(终端界面)。
// ============================================================================

import yargs from "yargs" // 命令行参数解析框架
import { hideBin } from "yargs/helpers" // 去掉 argv 前两项(node 路径、脚本路径)
// ---- 下面是各个子命令(每个文件导出一个 yargs command 定义)----
import { RunCommand } from "./cli/cmd/run"
import { GenerateCommand } from "./cli/cmd/generate"
import { Log } from "./util"
import { ConsoleCommand } from "./cli/cmd/account"
import { ProvidersCommand } from "./cli/cmd/providers"
import { AgentCommand } from "./cli/cmd/agent"
import { UpgradeCommand } from "./cli/cmd/upgrade"
import { UninstallCommand } from "./cli/cmd/uninstall"
import { ModelsCommand } from "./cli/cmd/models"
import { UI } from "./cli/ui"
import { Installation } from "./installation"
import { InstallationVersion } from "./installation/version"
import { NamedError } from "@mimo-ai/shared/util/error"
import { FormatError } from "./cli/error"
import { ServeCommand } from "./cli/cmd/serve"
import { Filesystem } from "./util"
import { DebugCommand } from "./cli/cmd/debug"
import { StatsCommand } from "./cli/cmd/stats"
import { McpCommand } from "./cli/cmd/mcp"
import { GithubCommand } from "./cli/cmd/github"
import { ExportCommand } from "./cli/cmd/export"
import { ImportCommand } from "./cli/cmd/import"
import { AttachCommand } from "./cli/cmd/tui/attach"
import { TuiThreadCommand } from "./cli/cmd/tui/thread"
import { AcpCommand } from "./cli/cmd/acp"
import { EOL } from "os"
import { WebCommand } from "./cli/cmd/web"
import { PrCommand } from "./cli/cmd/pr"
import { SessionCommand } from "./cli/cmd/session"
import { DbCommand } from "./cli/cmd/db"
import path from "path"
import { Global } from "./global"
import { JsonMigration } from "./storage"
import { Database } from "./storage"
import { ClaudeImport } from "./session/claude-import"
import { errorMessage } from "./util/error"
import { PluginCommand } from "./cli/cmd/plug"
import { Heap } from "./cli/heap"
import { drizzle } from "drizzle-orm/bun-sqlite"
import { ensureProcessMetadata } from "./util/mimo-process"

// 标记当前进程角色为 "main"(主进程)。配合 Worker 线程时用于区分进程身份。
const processMetadata = ensureProcessMetadata("main")

// 全局兜底:捕获未处理的 Promise rejection,只记日志、不让进程崩溃。
process.on("unhandledRejection", (e) => {
  Log.Default.error("rejection", {
    e: errorMessage(e),
  })
})

// 全局兜底:捕获未捕获的同步异常,同样只记日志。
process.on("uncaughtException", (e) => {
  Log.Default.error("exception", {
    e: errorMessage(e),
  })
})

// 去掉 process.argv 的前两项,只保留用户真正输入的参数。
const args = hideBin(process.argv)

// 打印帮助/输出文本的辅助函数:非 "mimo " 开头的内容前面带上 logo。
function show(out: string) {
  const text = out.trimStart()
  if (!text.startsWith("mimo ")) {
    process.stderr.write(UI.logo() + EOL + EOL)
    process.stderr.write(text)
    return
  }
  process.stderr.write(out)
}

// ---- 构建 CLI 实例:设置全局选项 + 注册子命令 ----
const cli = yargs(args)
  .parserConfiguration({ "populate--": true }) // 允许把 `--` 之后的参数收集到 argv["--"]
  .scriptName("mimo") // 命令名显示为 mimo
  .wrap(100) // 帮助信息换行宽度
  .help("help", "show help")
  .alias("help", "h")
  .version("version", "show version number", InstallationVersion) // -v 显示版本
  .alias("version", "v")
  // ---- 全局选项(所有子命令都能用)----
  .option("print-logs", {
    describe: "print logs to stderr",
    type: "boolean",
  })
  .option("log-level", {
    describe: "log level",
    type: "string",
    choices: ["DEBUG", "INFO", "WARN", "ERROR"],
  })
  .option("pure", {
    describe: "run without external plugins", // 不加载外部插件的"纯净"模式
    type: "boolean",
  })
  // ---- middleware:在执行任何子命令之前先跑的初始化逻辑 ----
  // opts 是 yargs 解析命令行后得到的参数对象(argv):
  //   - 包含上面定义的全局选项,如 opts.pure / opts.logLevel(--log-level 自动转驼峰)
  //   - 也含 yargs 内置字段:opts._(位置参数)、opts.$0(脚本名)、opts["--"]
  // middleware 靠读 opts 来决定初始化行为(纯净模式、日志级别等)。
  .middleware(async (opts) => {
    // 读 --pure:开启后进入"纯净模式",运行时不加载外部插件。
    if (opts.pure) {
      process.env.MIMOCODE_PURE = "1"
    }

    // 初始化日志系统:本地开发默认 DEBUG,否则 INFO。
    await Log.init({
      print: process.argv.includes("--print-logs"),
      dev: Installation.isLocal(),
      level: (() => {
        if (opts.logLevel) return opts.logLevel as Log.Level
        if (Installation.isLocal()) return "DEBUG"
        return "INFO"
      })(),
    })

    Heap.start() // 启动堆内存监控(用于调试内存)

    // 设置一批环境变量,供子进程/工具识别"自己跑在 MiMoCode 里"。
    process.env.AGENT = "1"
    process.env.MIMOCODE = "1"
    process.env.MIMOCODE_PID = String(process.pid)

    Log.Default.info("mimocode", {
      version: InstallationVersion,
      args: process.argv.slice(2),
      process_role: processMetadata.processRole,
      run_id: processMetadata.runID,
    })

    // ---- 首次运行:执行一次性的 SQLite 数据库迁移 ----
    // 通过 mimocode.db 是否存在来判断是不是第一次跑。
    const marker = path.join(Global.Path.data, "mimocode.db")
    if (!(await Filesystem.exists(marker))) {
      const tty = process.stderr.isTTY // 是否在真实终端(决定是否画进度条)
      process.stderr.write("Performing one time database migration, may take a few minutes..." + EOL)

      // terminal 样式设置相关变量
      const width = 36
      const orange = "\x1b[38;5;214m"
      const muted = "\x1b[0;2m"
      const reset = "\x1b[0m"

      let last = -1
      if (tty) process.stderr.write("\x1b[?25l") // 隐藏光标
      try {
        // TTY 是否为真实终端还是内部调用
        // 执行迁移,progress 回调实时渲染进度条(TTY)或打印百分比(非 TTY)。
        await JsonMigration.run(drizzle({ client: Database.Client().$client }), {
          progress: (event) => {
            const percent = Math.floor((event.current / event.total) * 100)
            if (percent === last && event.current !== event.total) return
            last = percent
            if (tty) {
              const fill = Math.round((percent / 100) * width)
              const bar = `${"■".repeat(fill)}${"･".repeat(width - fill)}`
              process.stderr.write(
                `\r${orange}${bar} ${percent.toString().padStart(3)}%${reset} ${muted}${event.label.padEnd(12)} ${event.current}/${event.total}${reset}`,
              )
              if (event.current === event.total) process.stderr.write("\n")
            } else {
              process.stderr.write(`sqlite-migration:${percent}${EOL}`)
            }
          },
        })
      } finally {
        if (tty) process.stderr.write("\x1b[?25h") // 恢复光标
        else {
          process.stderr.write(`sqlite-migration:done${EOL}`)
        }
      }
      process.stderr.write("Database migration complete." + EOL)
    }

    // ---- 幂等地把 Claude Code 的历史会话导入到 SQLite ----
    // 每个进程树只跑一次(靠环境变量做标记,子进程会继承)。
    // best-effort:即使失败也绝不能阻塞命令启动。
    if (!process.env.MIMOCODE_DISABLE_CLAUDE_IMPORT && !process.env.MIMOCODE_CLAUDE_IMPORTED) {
      process.env.MIMOCODE_CLAUDE_IMPORTED = "1"
      try {
        await ClaudeImport.run()
      } catch (e) {
        Log.Default.warn("claude-import failed", { e: errorMessage(e) })
      }
    }
  })
  .usage("")
  .completion("completion", "generate shell completion script") // 生成 shell 自动补全脚本
  // ---- 注册所有子命令 ----
  .command(AcpCommand)
  .command(McpCommand)
  .command(TuiThreadCommand) // "$0" 默认命令:不带子命令时启动 TUI
  .command(AttachCommand)
  .command(RunCommand)
  .command(GenerateCommand)
  .command(DebugCommand)
  .command(ConsoleCommand)
  .command(ProvidersCommand)
  .command(AgentCommand)
  .command(UpgradeCommand)
  .command(UninstallCommand)
  .command(ServeCommand)
  .command(WebCommand)
  .command(ModelsCommand)
  .command(StatsCommand)
  .command(ExportCommand)
  .command(ImportCommand)
  .command(GithubCommand)
  .command(PrCommand)
  .command(SessionCommand)
  .command(PluginCommand)
  .command(DbCommand)
  // ---- 参数解析失败时的处理:参数类错误显示帮助,其余直接退出 ----
  .fail((msg, err) => {
    if (
      msg?.startsWith("Unknown argument") ||
      msg?.startsWith("Not enough non-option arguments") ||
      msg?.startsWith("Invalid values:")
    ) {
      if (err) throw err
      cli.showHelp(show)
    }
    if (err) throw err
    process.exit(1)
  })
  .strict() // 严格模式:不认识的命令/参数直接报错

// ---- 真正执行解析 ----
try {
  // 带 -h/--help 时,用回调拿到帮助文本再走 show() 渲染(带 logo)。
  if (args.includes("-h") || args.includes("--help")) {
    await cli.parse(args, (err: Error | undefined, _argv: unknown, out: string) => {
      if (err) throw err
      if (!out) return
      show(out)
    })
  } else {
    // 正常情况:解析并执行对应子命令的 handler。
    await cli.parse()
  }
} catch (e) {
  // ---- 统一的致命错误处理:尽量把错误信息结构化后写日志 ----
  let data: Record<string, any> = {}
  // NamedError 是项目自定义错误,带额外结构化数据。
  if (e instanceof NamedError) {
    const obj = e.toObject()
    Object.assign(data, {
      ...obj.data,
    })
  }

  if (e instanceof Error) {
    Object.assign(data, {
      name: e.name,
      message: e.message,
      cause: e.cause?.toString(),
      stack: e.stack,
    })
  }

  // ResolveMessage 是 Bun 的模块解析错误,单独提取它的字段。
  if (e instanceof ResolveMessage) {
    Object.assign(data, {
      name: e.name,
      message: e.message,
      code: e.code,
      specifier: e.specifier,
      referrer: e.referrer,
      position: e.position,
      importKind: e.importKind,
    })
  }
  Log.Default.error("fatal", data)
  // 把错误格式化成用户友好的提示;格式化不出来就给个兜底提示 + 日志路径。
  const formatted = FormatError(e)
  if (formatted) UI.error(formatted)
  if (formatted === undefined) {
    UI.error("Unexpected error, check log file at " + Log.file() + " for more details" + EOL)
    process.stderr.write(errorMessage(e) + EOL)
  }
  process.exitCode = 1
} finally {
  // 有些子进程(比如基于 docker 容器的 MCP server)不响应 SIGTERM 等信号。
  // 这里显式调用 process.exit() 强制退出,避免留下挂起的子进程。
  process.exit()
}
