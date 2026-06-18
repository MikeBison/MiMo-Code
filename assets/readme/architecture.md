# MiMoCode 架构图

## 系统分层架构

整体分为三层:**用户入口层**负责命令解析与分发,**业务逻辑层**承载会话/代理/工具/模型调用的核心流程,**基础设施层**提供事件、配置、存储等通用能力。

```mermaid
flowchart TD
    subgraph L1["用户入口层"]
        CLI["CLI 入口 · src/index.ts<br/>yargs 解析参数 → 分发子命令"]
        CLI --> TUI["TUI 命令<br/>终端交互界面"]
        CLI --> SERVE["Serve 命令<br/>Web 服务器"]
        CLI --> RUN["Run 命令<br/>单次执行"]
    end

    subgraph L2["业务逻辑层"]
        SESSION["session 会话管理<br/>processor · llm · checkpoint · message"]
        AGENT["agent 代理配置<br/>Build · Plan · Compose · Explore"]
        TOOL["tool 工具执行<br/>bash · read · edit · glob · grep · actor · task"]
        PROVIDER["provider 模型抽象<br/>OpenAI · Anthropic · Google · Azure · Bedrock"]
        SESSION --> AGENT
        SESSION --> TOOL
        SESSION --> PROVIDER
    end

    subgraph L3["基础设施层"]
        BUS["bus 事件总线"]
        CONFIG["config 配置管理"]
        STORAGE["storage 数据库"]
        PERMISSION["permission 权限控制"]
        PLUGIN["plugin 插件系统"]
        EFFECT["effect 副作用管理"]
        PROJECT["project 项目管理"]
        GLOBAL["global 全局常量"]
    end

    L1 --> L2
    L2 --> L3
```

## 依赖方向说明

箭头表示「依赖 / 调用」,整体自上而下单向依赖,不反向依赖;模块间通过 `bus` 事件总线解耦。

```mermaid
flowchart LR
    index["index"] --> cli["cli/*"]
    cli --> session["session"]
    session --> agent["agent"]
    session --> provider["provider"]
    session --> tool["tool"]
    agent --> config["config"]
    provider --> LLM["外部 LLM API"]
    tool --> bus["bus"]
    config -.-> provider
```

关键依赖关系:

- `session` 依赖 `agent`(读取代理配置)、`provider`(调用 LLM)、`tool`(执行工具)
- `agent` 依赖 `config`(读取配置)
- `tool` 依赖 `bus`(发布事件)
- `provider` 依赖外部 LLM API

## 核心调用链路

```mermaid
graph TD
    A[用户输入命令] --> B[CLI 入口]
    B --> C[子命令分发]
    C --> D[session 创建/恢复]
    D --> E[加载 agent 配置]
    E --> F[选择 provider]
    F --> G[调用 LLM API]
    G --> H[解析 LLM 响应]
    H --> I{需要工具?}
    I -->|是| J[执行 tool]
    J --> K[返回工具结果]
    K --> H
    I -->|否| L[生成最终回复]
    L --> M[更新 session]
    M --> N[返回给用户]

    style A fill:#e1f5ff
    style G fill:#f0fff0
    style J fill:#fff0f0
    style N fill:#e1f5ff
```

## 模块职责表

| 模块 | 入口文件 | 核心职责 | 关键依赖 |
|------|----------|----------|----------|
| **session** | `src/session/session.ts` | 会话生命周期、消息管理、上下文窗口 | agent, provider, tool, bus, storage |
| **agent** | `src/agent/agent.ts` | 代理角色定义、权限配置、提示词管理 | config, provider |
| **tool** | `src/tool/registry.ts` | 工具注册、执行、结果格式化 | session, agent, bus, permission |
| **provider** | `src/provider/provider.ts` | LLM 调用抽象、模型管理、响应转换 | config, auth, plugin |
| **bus** | `src/bus/index.ts` | 事件发布/订阅、模块间解耦通信 | - |
| **config** | `src/config/config.ts` | 配置读取、层次合并、动态更新 | global |
| **storage** | `src/storage/db.bun.ts` | SQLite 数据库、Drizzle ORM、迁移 | - |
| **permission** | `src/permission/index.ts` | 工具执行权限控制、用户授权 | config |

## 详细调用流程

### 用户发送消息到 LLM 响应

```mermaid
flowchart TD
    U[用户输入] --> CLI["CLI index.ts 解析命令"]
    CLI --> RUN["子命令 cli/cmd/run.ts 创建 session"]
    RUN --> PROC["processor.ts 加载消息历史"]
    PROC --> PROMPT["prompt.ts 构建系统提示词"]
    PROMPT --> AG["agent.ts 读取代理配置"]
    PROMPT --> CFG["config.ts 读取用户配置"]
    PROMPT --> LLMCALL["llm.ts 调用 provider"]
    LLMCALL --> PV["provider.ts 选择模型"]
    PV --> API["调用 LLM API"]
    API --> RESP{响应类型}
    RESP -->|纯文本| DONE["直接返回"]
    RESP -->|工具调用| TOOL["执行 tool"]
    TOOL --> LLMCALL
    DONE --> UPD["更新 session 消息历史"]
    UPD --> EVT["发布 bus 事件"]
    EVT --> OUT[返回给用户]
```

### 工具执行流程

```mermaid
flowchart TD
    A["LLM 返回工具调用"] --> B["processor.ts 识别工具调用"]
    B --> C["registry.ts 查找工具定义"]
    C --> D["permission.ts 权限检查"]
    D -->|需要授权| E["弹出用户确认"]
    D -->|无需授权| F["tool.ts 参数验证 (Zod)"]
    E --> F
    F --> G["执行具体工具<br/>bash / read / edit / actor ..."]
    G --> H["格式化输出"]
    H --> I["发布 bus 事件"]
    I --> J["返回给 LLM 继续对话"]
```

## 技术栈

- **运行时:** Bun
- **语言:** TypeScript
- **数据库:** SQLite (Drizzle ORM)
- **框架:** Effect (副作用管理)
- **UI:** SolidJS — 终端 TUI 经 OpenTUI 渲染,Web 端经 DOM 渲染;服务端 Hono
- **包管理:** Bun workspace

## 目录结构

```
packages/opencode/
├── src/
│   ├── index.ts           # CLI 入口
│   ├── cli/               # 命令定义
│   │   └── cmd/           # 各子命令(含 tui/ 终端界面)
│   ├── session/           # 会话管理(核心)
│   ├── agent/             # 代理配置
│   ├── tool/              # 工具系统
│   ├── provider/          # LLM 提供商
│   ├── config/            # 配置管理
│   ├── storage/           # 数据库
│   ├── bus/               # 事件总线
│   ├── permission/        # 权限控制
│   ├── plugin/            # 插件系统
│   ├── project/           # 项目管理
│   └── util/              # 工具函数
├── migration/             # 数据库迁移
└── test/                  # 测试
```

## 关键设计模式

### 1. 分层架构

- **用户入口层:** CLI 解析、命令分发
- **业务逻辑层:** session / agent / tool / provider
- **基础设施层:** bus / config / storage / permission

### 2. 依赖方向

- 上层依赖下层,不反向依赖
- 通过 bus 事件总线实现松耦合
- 通过 Effect 管理副作用

### 3. 模块职责单一

- session:会话生命周期管理
- agent:代理配置和角色定义
- tool:工具注册和执行
- provider:LLM 调用抽象

## 扩展点

1. **自定义工具:** 实现 `Tool.Def` 接口,注册到 `registry.ts`
2. **自定义代理:** 配置代理角色和提示词
3. **MCP 服务器:** 集成外部工具和服务
4. **插件系统:** 通过 `@mimo-ai/plugin` 扩展
5. **提供商:** 添加新的 LLM 提供商
