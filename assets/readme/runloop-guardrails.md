# runLoop 控制闸门（围栏）速查

> `runLoop`（`packages/opencode/src/session/prompt.ts`）是一个 ReAct 死循环：
> **调模型 → 执行工具 → 结果喂回 → 再来一圈**，直到模型交出最终答案。
> 下面是它为「不可靠的大模型」设的所有围栏。开发 / 调 agent 循环时对照本表即可。
>
> 读法：每个围栏 = **触发信号 → 处理动作 · 调优常量**。围栏靠注入一条
> `synthetic:true` 的合成用户消息来「提醒」模型；`continue`=再转一圈，`break`=收工。

---

## 分类速览

| 类别 | 干什么 | 结果 |
|---|---|---|
| **重试类**（§2） | 拦一种模型抽风，注入提醒重试 | 能救 `continue`；超上限写错误 `break` |
| **上下文类**（§3） | 上下文超长时压缩 / 重建 | `continue` |
| **终态类**（§4） | 过滤 / 报错 / 取消等终态 | 直接 `break` |
| **否决闸门**（§5） | 模型想停时二次把关（仅主 agent） | 未完成则否决、强制 `continue` |

---

## 2. 重试类围栏

### 2.1 输出被截断续写 · `autoContinueOutputLength`
模型长回答撞 token 输出上限被截断（`finish=length`、无工具调用）→ 注入「从中断处继续、别复述」重试 · `OUTPUT_LENGTH_CONTINUATION_LIMIT`=**3** → 超限写 `OutputLengthError` 并 break

### 2.2 只思考 / 空输出 · `autoContinueInvalidOutput`
模型只在思考区想、对用户没给答案也没调工具，或返回全空（`think-only`/`invalid`，非 json_schema）→ 注入「给最终答案或调真实工具」→ `INVALID_OUTPUT_CONTINUATION_LIMIT`=**2** → 超限写 `InvalidOutputError` 并 break

### 2.3 工具调用被写成纯文本 · `autoRetryTextToolCall`
模型把工具调用打印成文本字面量（如吐出 `<invoke name=...>`）而没走结构化通道，工具没执行 → 丢弃坏 turn + 注入「用真实工具通道重发」→ `TEXT_TOOL_CALL_RETRY_LIMIT`=**2** → 超限保留错误并 break
› 注意：丢弃坏 turn 是为了避免对话搁浅在助手 prefill 上（provider 会拒预填充）。

### 2.4 要 json_schema 却没给结构化输出 · `autoRetryStructuredOutput`
要求返回符合 schema 的 JSON，但模型用纯文本/空/只思考应付、没调 StructuredOutput 工具 → 注入「必须调 StructuredOutput 传合规 JSON」→ 超限写 `StructuredOutputError` 并 break
› 注意：上限取**本次请求随附的 `format.retryCount`**（无全局常量），与 §2.2 分开计数。

### 2.5 流内 n-gram 复读 · `handleTextRepeat`
模型在**同一次**流式输出里刷屏同一短语（「让我再检查一下」×N）→ 第 1 次注入 REMIND、第 2 次 REPLAN → `TEXT_NGRAM_MAX_RECOVERY`=**2** → 超限发 `Session.Event.Error` 并 break
› 检测参数：`TEXT_NGRAM_N`=4 块长 · `TEXT_REPEAT_THRESHOLD`=20 重复次数 · `TEXT_WINDOW_TOKENS`=500 窗口。

### 2.6 空 / 无效工具调用循环 · `handleEmptyStep`
模型反复「假装干活」——调工具但参数全空，或每步既不调工具也不说话 → 第 1 次 REMIND、第 2 次 REPLAN → `EMPTY_STEP_MAX_RECOVERY`=**2** → 超限设 `hardHalt` 并 break
› 注意：`hardHalt` 是硬熔断，会**跳过 §5 两道否决闸门**，防止被反复拽回同一空循环。补的是 §2.5 抓不到的盲区（空调用没文本可匹配）。

### 2.7 跨步文本循环 · `detectTextLoop`
模型**跨多步**重复同一套「文本+工具」，整体没进展（§2.5 单步内看不出）→ 第 1 次 MILD、之后 STRONG → `TEXT_LOOP_MAX_RECOVERY`=**2** → 超限发 `Session.Event.Error` 并 break
› 检测参数：`TEXT_LOOP_BUFFER_SIZE`=5 缓冲步 · `TEXT_LOOP_TRIGGER_COUNT`=3 步全同即触发（文本相同但工具不同不算）。

### 2.8 重复步骤提醒 · 工具签名重复（纯软提醒）
模型连续几步用**完全相同参数**调同一工具（反复跑同一命令/读同一文件）→ 注入「原地打转，换思路」一次（去重，不硬 break）→ `REPEATED_STEP_THRESHOLD`=**3**
› 区别 §2.7：这里只看工具签名，不看文本。

---

## 3. 上下文类围栏

### 3.1 高上下文压力提醒 · `pressureLevel`（软提醒）
上下文快占满、临近 checkpoint 丢弃旧内容 → 注入「先把重要结论写进 memory，然后**继续原任务**」→ 触发 `pressureLevel>=2`（>70%，>=3 时 >85%），每段去重一次
› 注意：措辞刻意强调「这是保存工作、不是收尾信号」，否则模型会误判成「要重置了」而提前交回控制权。

### 3.2 上下文溢出 · 主 agent 重建 / 子 agent 压缩
上下文越过硬阈值（`overflowCheck` 或 `maxThresholdCrossed`，也含 provider 返回 `overflow`）→ **子 agent**：`compaction.create` 有损摘要；**主 agent**：优先 `rebuildFromCheckpoint`，无 checkpoint 才回退 compaction → `continue`
› 豁免：有界计算 agent（`native && hidden`：title/summary/checkpoint-writer）不做上下文管理。

### 3.3 compaction 边界路由
最后一条用户消息带 `compaction` part（`/compact` 或自动溢出插入的边界标记）→ 拦下改调 `compaction.process`，不走正常 LLM 流程 → 返回 `stop` 则 break，否则 `continue`

---

## 4. 终态类围栏（直接 break，无重试）

- **内容安全过滤** — `finish=content-filter` → `writeContentFilterError` → break（首次即终止，重发只会再被过滤）
- **模型报错** — `finish=error` 或已带 error → `writeModelError` → break
- **拿到结构化输出** — `structured !== undefined` → 存下 → break
- **模型正常说完** — `result=stop` → break
- **被插件取消** — `session.userQuery.pre` 设 `cancel=true` → 写 `AbortedError` → break
- **fork 快照丢失** — fork agent 的 `forkCtx` 为空 → actor 标记 failure → break（下轮重新派生）
- **硬熔断** — `hardHalt=true`（来自 §2.6）→ 跳过 §5 直接 break

---

## 5. 收工前的两道否决闸门（仅主 agent）

`outcome=break` 时（除非 `hardHalt`），顺序过这两道，可**否决 break、强制 continue**——防止 agent 半途而废。

### 5.1 taskGate（先跑）
还有 `open`/`in_progress` 的未终结任务 → 注入「逐个 `task done`/`task abandon`」→ `MAX_TASK_GATE_MAIN_REACT`=**3** 次后放行（子 agent 用 `MAX_TASK_GATE_SUBAGENT_REACT`=**2**）
› 跳过：`task` 工具被禁用时（否则会因无法 `task done` 死循环）· fail-open：registry 出错按「无任务」放行。

### 5.2 goalGate（后跑，需裁判模型）
存在活跃 `/goal` 停止条件 → 裁判模型读记录判断是否达成，未达成则注入「还差什么」→ `MAX_GOAL_REACT`=**12** 次后放行
› fail-open：裁判出错视为「已满足」放行。上限比 taskGate 高，因为有裁判能判断「是否真在推进」。
› 顺序：taskGate 先于 goalGate（任务状态是确定的 DB 事实，未清任务板会污染 goal 裁决）。

---

## 6. 常量索引（可调优处）

`常量` = 默认值（环境变量）→ 归属围栏

- `OUTPUT_LENGTH_CONTINUATION_LIMIT` = 3（`MIMOCODE_OUTPUT_LENGTH_CONTINUATION_LIMIT`）→ §2.1
- `INVALID_OUTPUT_CONTINUATION_LIMIT` = 2（`MIMOCODE_INVALID_OUTPUT_CONTINUATION_LIMIT`）→ §2.2
- `TEXT_TOOL_CALL_RETRY_LIMIT` = 2（`MIMOCODE_TEXT_TOOL_CALL_RETRY_LIMIT`）→ §2.3
- `TEXT_NGRAM_MAX_RECOVERY` = 2（无 env）→ §2.5
- `TEXT_NGRAM_N` = 4（`MIMOCODE_TEXT_NGRAM_N`）→ §2.5
- `TEXT_REPEAT_THRESHOLD` = 20（`MIMOCODE_TEXT_REPEAT_THRESHOLD`）→ §2.5
- `TEXT_WINDOW_TOKENS` = 500（`MIMOCODE_TEXT_WINDOW_TOKENS`）→ §2.5
- `EMPTY_STEP_MAX_RECOVERY` = 2（`MIMOCODE_EMPTY_STEP_MAX_RECOVERY`）→ §2.6
- `TEXT_LOOP_BUFFER_SIZE` = 5（无 env）→ §2.7
- `TEXT_LOOP_TRIGGER_COUNT` = 3（无 env）→ §2.7
- `TEXT_LOOP_MAX_RECOVERY` = 2（无 env）→ §2.7
- `REPEATED_STEP_THRESHOLD` = 3（无 env）→ §2.8
- `MAX_TASK_GATE_MAIN_REACT` = 3（无 env）→ §5.1
- `MAX_TASK_GATE_SUBAGENT_REACT` = 2（无 env）→ §5.1
- `MAX_GOAL_REACT` = 12（无 env，`prompt.ts` 硬编码）→ §5.2
- （§2.4 无全局常量，用本次请求的 `format.retryCount`）

---

## 7. 开发自查清单

- [ ] 输出被 token 截断 → 能自动续写且有上限（§2.1）
- [ ] 只思考 / 空输出 → 能提醒并有上限（§2.2）
- [ ] 工具调用写成纯文本 → 能丢弃坏 turn 重试（§2.3）
- [ ] 要结构化输出没给 → 能重试（per-request retryCount）（§2.4）
- [ ] 单步复读 / 跨步复读 / 空工具调用 → 三套检测各有软→硬阶梯（§2.5–2.7）
- [ ] 连续相同工具调用 → 有换思路提醒（§2.8）
- [ ] 上下文超长 → 主 agent 重建、子 agent 压缩，有界计算豁免（§3）
- [ ] 过滤 / 报错 / 取消 → 有用户可见错误并干净退出（§4）
- [ ] 模型想停但 task/goal 未完成 → 有否决闸门拽回（§5）
- [ ] 每类重试都有硬上限，绝不无限循环；hardHalt 会跳过否决闸门

---

*来源：`session/prompt.ts`、`classify.ts`、`task/gate.ts`、`goal.ts`、`prompt/text-loop-recovery.ts`、`prompt/text-ngram-detection.ts`、`prompt/empty-step-detection.ts`、`flag/flag.ts`。*
