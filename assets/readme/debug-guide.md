# 调试指南：debug: serve (engine) + attach

本文档记录 MiMoCode 的推荐调试方式：**把「引擎」和「TUI」拆开** —— 引擎作为无界面后台服务启动并挂上调试器（可断点），TUI 在另一个终端 attach 上去交互驱动。这样断点停在引擎里时，TUI 界面不受影响。

> 适用场景：想像真实使用一样操作 TUI，同时在 `prompt.ts` / `registry.ts` 等引擎代码里随时命中断点、反复调试。

---

## 为什么不用 `npm run dev:inspect`

- **Bun 的调试器不是 Node/Chrome 那套协议**：Bun 用的是基于 WebKit 的 inspector，`chrome://inspect` 连不上，必须用 VS Code 的 Bun 扩展或 Bun 打印的 `debug.bun.sh` 网页调试器。
- **`dev:inspect` 直接启动 TUI**：TUI 接管整个终端（stdin/stdout、渲染循环），断点一停界面就冻住，且看不清 inspector URL，体验很差。

因此推荐用下面的「引擎 + attach」方式。

---

## 前置条件

1. **安装 VS Code Bun 扩展**：扩展市场搜索并安装 `oven.bun-vscode`（`.vscode/launch.json` 里 `type: "bun"` 的配置依赖它）。装完重启 VS Code。
2. 确认 `bun` 已安装（本项目使用 `bun@1.3.14`）。
3. 依赖已安装（`bun install`）。

---

## 调试步骤

### 第 1 步：启动引擎（带调试器）

在 VS Code **运行和调试**面板，选择配置 **`debug: serve (engine)`**，按 `F5`。

它等价于执行：

```bash
# cwd: packages/opencode
MIMOCODE_HOME=<workspace>/.dev-home \
  bun run --conditions=browser src/index.ts serve --port 4096
```

此时引擎作为**无界面后台服务**运行在 `http://localhost:4096`，调试器已挂上，进程会一直等待请求。

### 第 2 步：打断点

在引擎侧代码打断点，常用位置：

- `packages/opencode/src/session/prompt.ts` —— `runLoop` 的 `while (true)`、`createUserMessage`、`resolveTools` 等
- `packages/opencode/src/tool/registry.ts` —— 工具解析
- 其它你关心的引擎逻辑

> 断点要在触发前打好；确认断点是「实心」的（已 bind），否则检查 Bun 扩展是否安装并重启。

### 第 3 步：另开终端，用 TUI attach 到引擎

**新开一个终端**，从**仓库根目录**运行（`MIMOCODE_HOME` 必须与引擎一致，否则连不到同一份数据）：

```bash
MIMOCODE_HOME=$PWD/.dev-home \
  bun run --cwd packages/opencode --conditions=browser \
  src/index.ts attach http://localhost:4096
```

`attach` 命令的常用参数：

- `-c, --continue` —— 继续上一个会话
- `-s, --session <id>` —— 继续指定会话
- `--fork` —— 继续时 fork 会话（配合 `--continue` / `--session`）
- `--dir <path>` —— 指定运行目录
- `-p, --password <pwd>` —— basic auth 密码（默认取 `MIMOCODE_SERVER_PASSWORD`）

### 第 4 步：交互并命中断点

在这个 TUI 里正常打字、操作。每次输入都会打到后台引擎，**断点即在 VS Code 中命中**。TUI 在另一个终端，不受断点冻结影响。

---

## 其它调试配置（`.vscode/launch.json`）

| 配置名 | 用途 | 说明 |
|---|---|---|
| **debug: serve (engine)** | 引擎后台服务 + 断点（本文档） | 配合另开终端 `attach` 驱动 |
| debug: mimo run | 调试单条无 TUI 命令 | `args: ["run", "你好"]`，跑完即退；改 args 里的 prompt 可调不同路径。先打断点再 F5 |
| opencode (attach) | attach 到已用 `--inspect=6499` 启动的进程 | `url: ws://localhost:6499/` |

---

## 常见问题

- **断点不是实心（不 bind）**：多半是没装 `oven.bun-vscode` 扩展，或装后未重启 VS Code。
- **attach 连不上 / 看不到数据**：确认 attach 终端的 `MIMOCODE_HOME` 与引擎 `debug: serve (engine)` 用的一致（都是 `<workspace>/.dev-home`）。
- **端口占用**：`serve --port 4096` 与 `attach http://localhost:4096` 的端口要一致；被占用时两处一起改。
- **改了引擎代码**：需重启 `debug: serve (engine)`（重新 F5）让改动生效；TUI 侧 attach 可保持不动，重连即可。
