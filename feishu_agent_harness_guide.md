# 从零理解 AI Agent：逆向解析 Claude Code 架构

> **一句话总结**：模型本身就是 Agent，你写的代码只是为它搭台子（Harness）。

---

## 写在前面：这篇文档适合谁？

如果你是：

- 刚接触 AI Agent 概念的新手
- 听过 Claude Code / Cursor / Copilot 但不知道背后怎么运作的
- 想自己动手搭 Agent 但不知道从哪开始的开发者
- 对"提示词工程"感到困惑的初学者

那这篇文档就是为你写的。我们会用**最直白的语言**，从一个不到 30 行的程序开始，一步步带你理解一个完整 AI Agent 系统的全部架构。

---

## 第零章：先搞清楚两个概念

### Agent 是什么？

很多人以为 Agent 是一段复杂的代码程序。**错。**

**Agent 就是大语言模型本身。** 比如 Claude、GPT-4——它们已经通过海量训练学会了推理、规划、决策。你不需要"教"模型怎么思考，模型已经会了。

### Harness（脚手架）是什么？

既然模型已经会思考了，那我们程序员要做什么？

**搭台子。**

模型很聪明，但它"看不见摸不着"——不能读你的文件、不能跑你的测试、不能改你的代码。**Harness 就是连接模型和真实世界的桥梁**，包括：

- 给模型提供**工具**（读写文件、执行命令）
- 给模型提供**知识**（领域文档、代码规范）
- 给模型提供**记忆**（任务进度、对话历史）
- 给模型设定**边界**（安全沙箱、权限控制）

> 一个类比：模型是一个天才工程师，Harness 是你给他准备的办公室——电脑、工具、文件柜、门禁卡。天才的能力是他自己的，但没有办公室他什么也做不了。

---

## 第一章：Agent 循环——一切的起点

### 问题

大语言模型只能"说话"，它说"请帮我运行 `python test.py`"，但它真的没法自己去执行这个命令。

如果每次模型想用工具，都需要你手动复制命令、粘贴结果，那你自己就是那个"循环"。

### 解决方案：自动化这个循环

整个 Agent 系统的核心，其实就是一个 `while` 循环：

```
你提问 ──→ 模型思考 ──→ 模型调用工具 ──→ 执行工具 ──→ 把结果告诉模型 ──→ 模型继续思考 ──→ ...
                                                                              ↓
                                                                        模型说"我做完了" ──→ 结束
```

用一张图来表示：

```
+--------+      +-------+      +---------+
|  你的   | --→ | 大模型 | --→ | 执行工具  |
|  提问   |      |       |      |（跑命令）|
+--------+      +---+---+      +----+----+
                    ↑                |
                    |   工具执行结果   |
                    +────────────────+
                   （循环，直到模型说"我做完了"）
```

### 核心代码（不到 30 行！）

```python
def agent_loop(query):
    messages = [{"role": "user", "content": query}]   # 你的提问

    while True:
        # 1. 把消息发给大模型
        response = client.messages.create(
            model=MODEL, system=SYSTEM,
            messages=messages, tools=TOOLS,
        )

        # 2. 记录模型的回复
        messages.append({"role": "assistant", "content": response.content})

        # 3. 关键判断：模型还想用工具吗？
        if response.stop_reason != "tool_use":
            return   # 不想了，任务完成！

        # 4. 模型想用工具 → 执行它，把结果送回去
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

### 小白重点理解

1. **谁在控制流程？** 模型。代码只是在听模型的指挥。
2. **循环什么时候结束？** 当模型的 `stop_reason` 不再是 `"tool_use"` 的时候——也就是模型觉得"我做完了"。
3. **代码做了什么？** 只做了三件事：发消息、执行工具、送结果。**没有任何"智能"逻辑**。

> 记住这句话：**循环属于 Agent（模型），代码只是执行者。**

---

## 第二章：工具分发——让 Agent 的手伸得更远

### 问题

只有一个 `bash`（命令行）工具时，模型做什么都得走 shell 命令。比如读文件要用 `cat`，但 `cat` 遇到大文件就截断；编辑文件要用 `sed`，但 `sed` 遇到特殊字符就出错。

更重要的是——**每个 bash 命令都可能执行任意操作**，安全隐患巨大。

### 解决方案：专用工具 + 字典分发

```
模型说"我要用 read_file"
    ↓
代码查字典：{"read_file": 对应的处理函数}
    ↓
调用对应的函数，返回结果
```

这就像一个前台接待员：

```
客人（模型）："我要去 302 房间"
前台（代码）：查房间号 → 带去 302
```

### 核心代码

```python
# 工具分发表——一个 Python 字典搞定
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"]),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# 路径沙箱——防止模型"越狱"访问系统文件
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"路径逃逸工作区: {p}")
    return path
```

在循环里，只需要一行查表：

```python
handler = TOOL_HANDLERS.get(block.name)
output = handler(**block.input)
```

### 小白重点理解

1. **加新工具有多简单？** 在字典里加一行就行。循环代码完全不用动。
2. **沙箱是什么？** 就是限制模型只能在你指定的目录里操作，防止它读 `/etc/passwd` 这类系统文件。
3. **为什么用字典而不是 if/else？** 因为字典可以无限扩展而不会变得臃肿。想象 100 个工具写成 if/elif/elif...

---

## 第三章：TodoWrite——给模型一个记事本

### 问题

让模型做一个 10 步的任务，它做到第 3 步可能就忘了后面还有什么要做。

为什么？因为**对话越长，早期的指令影响力越弱**——它被后面大量的工具输出"淹没"了。这是所有大模型的通病。

### 解决方案：给模型一个待办清单

```
待办清单状态：
[ ] #1: 设置项目结构        ← 还没做
[>] #2: 编写核心逻辑        ← 正在做（同一时间只能有一个！）
[x] #3: 添加测试            ← 做完了
[ ] #4: 编写文档            ← 还没做

(1/4 已完成)
```

同时加一个"催促机制"——如果模型连续 3 轮没更新待办清单，就自动提醒它：

```xml
<reminder>更新你的待办清单。</reminder>
```

### 核心代码

```python
class TodoManager:
    def update(self, items: list) -> str:
        in_progress_count = 0
        for item in items:
            if item["status"] == "in_progress":
                in_progress_count += 1
        # 关键规则：同时只能做一件事！
        if in_progress_count > 1:
            raise ValueError("同一时间只能有一个任务在进行中")
        self.items = items
        return self.render()
```

催促机制：

```python
# 如果模型连续 3 轮没更新待办
if rounds_since_todo >= 3:
    results.insert(0, {
        "type": "text",
        "text": "<reminder>更新你的待办清单。</reminder>"
    })
```

### 小白重点理解

1. **为什么只允许一个 in_progress？** 强制模型聚焦——做完一件再做下一件，不要分心。
2. **催促机制为什么有效？** 它在工具结果中插入提醒文本，模型下次"看到"时就会想起来要更新进度。
3. **这和 system prompt 有什么区别？** system prompt 是静态的"开场白"，而催促机制是动态的"戳一下"，效果更好。

---

## 第四章：子 Agent——大任务拆小，各干各的

### 问题

智能体工作越久，对话历史（messages 数组）就越长。读了 10 个文件、跑了 20 条命令，所有输出都永久留在上下文里。

有时候你只是问"这个项目用什么测试框架？"，模型需要读 5 个文件才能回答，但最终答案只有一个词："pytest"。那 5 个文件的内容全留在父对话里，白白浪费空间。

### 解决方案：子 Agent 用独立的对话

```
父 Agent                          子 Agent
+──────────────────+              +──────────────────+
| messages=[很长...]|              | messages=[] ← 全新！|
|                   |  分配任务     |                   |
| 工具: task        | ──────────→ | while 循环:       |
|   prompt="查一下  |              |   调用工具...      |
|   测试框架"       |              |   记录结果...      |
|                   |  返回摘要     |                   |
|   结果 = "pytest" | ←────────── | 返回最终文本       |
+──────────────────+              +──────────────────+
```

子 Agent 可能读了 5 个文件、跑了 3 条命令，但父 Agent 只收到一句话："pytest"。**子对话结束后直接丢弃**，不污染父对话。

### 核心代码

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # 全新上下文！

    for _ in range(30):  # 安全限制：最多 30 轮
        response = client.messages.create(
            model=MODEL,
            messages=sub_messages,  # 用子对话的 messages
            tools=CHILD_TOOLS,     # 子 Agent 没有 task 工具（防止套娃）
        )
        sub_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break
        # ... 执行工具 ...

    # 只返回最终文本，整个子对话被丢弃
    return "".join(b.text for b in response.content if hasattr(b, "text"))
```

### 小白重点理解

1. **为什么子 Agent 没有 task 工具？** 防止无限套娃——子 Agent 再生子 Agent 再生子 Agent...
2. **"上下文隔离"是什么意思？** 子 Agent 有自己的 messages 数组，和父 Agent 完全独立。做完就丢。
3. **这像什么？** 就像你把一个调研任务交给助理，助理翻了半天资料，最后只给你一页报告。你的桌子还是干净的。

---

## 第五章：技能加载——用到什么知识，临时加载什么

### 问题

你希望 Agent 掌握 10 种专业技能（代码审查、PDF 处理、MCP 构建...），每个技能的说明书有 2000 词。全塞进 system prompt = 20000 词，大部分跟当前任务毫无关系。

这就像**上考场时把整个图书馆背进去**——太重了。

### 解决方案：两层注入

```
第一层（系统提示，永远在）——只放"目录"：
┌──────────────────────────────────┐
│ 你是一个编程 Agent。             │
│ 可用技能：                       │
│   - pdf: 处理 PDF 文件           │  ← 每个技能只花 ~20 词
│   - code-review: 审查代码        │
│   - mcp-builder: 构建 MCP 服务器 │
└──────────────────────────────────┘

第二层（tool_result，按需加载）——模型说"我需要 pdf 技能"时才给：
┌──────────────────────────────────┐
│ <skill name="pdf">               │
│   完整的 PDF 处理说明书...       │  ← ~2000 词，只在需要时加载
│   第一步：...                    │
│   第二步：...                    │
│ </skill>                         │
└──────────────────────────────────┘
```

### 核心代码

```python
class SkillLoader:
    def __init__(self, skills_dir):
        # 扫描 skills/ 目录下所有 SKILL.md 文件
        for f in sorted(skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            self.skills[meta["name"]] = {"meta": meta, "body": body}

    def get_descriptions(self) -> str:
        """第一层：返回简短描述，放进 system prompt"""
        return "\n".join(f"  - {name}: {skill['meta']['description']}"
                         for name, skill in self.skills.items())

    def get_content(self, name: str) -> str:
        """第二层：返回完整内容，放进 tool_result"""
        return f"<skill name=\"{name}\">\n{self.skills[name]['body']}\n</skill>"
```

### 小白重点理解

1. **为什么不直接全放 system prompt？** 浪费 token（=浪费钱），而且会干扰模型理解当前任务。
2. **模型怎么知道什么时候加载技能？** 它看到系统提示里的"目录"，知道有 pdf 技能可用。当遇到 PDF 相关任务时，它自己会调用 `load_skill("pdf")`。
3. **知识注入的关键原则**：通过 `tool_result` 注入，而非 system prompt。这样知识会出现在对话的"最近位置"，模型最容易注意到。

---

## 第六章：上下文压缩——让对话可以无限延续

### 问题

大模型的上下文窗口是有限的（比如 200k token）。读一个 1000 行的文件就要消耗 ~4000 token。读 30 个文件、跑 20 条命令，轻松突破上限。

**不压缩，Agent 根本没法在大项目里长时间工作。**

### 解决方案：三层压缩策略

```
每一轮对话前：
┌──────────────────────┐
│ 第一层：微压缩        │  静默执行，每轮都做
│ 把 3 轮之前的工具结果 │
│ 替换为一句话占位符     │  "之前使用了 read_file" → 只花几个 token
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ 检查：token > 50000？│
│    否 → 继续正常工作  │
│    是 ↓               │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ 第二层：自动压缩      │  把完整对话存到磁盘
│ 让模型总结对话要点    │  然后用摘要替换所有消息
│ 相当于"翻到新的一页"  │
└──────────────────────┘

           +

┌──────────────────────┐
│ 第三层：手动压缩      │  模型觉得太臃肿时
│ 主动调用 compact 工具 │  效果同第二层
└──────────────────────┘
```

### 核心代码

第一层——把旧的工具结果替换为占位符：

```python
def micro_compact(messages):
    # 找到所有 tool_result
    # 保留最近 3 个，其余替换为占位符
    for result in old_results:
        result["content"] = f"[之前使用了 {tool_name}]"
```

第二层——保存完整对话到磁盘，然后让模型总结：

```python
def auto_compact(messages):
    # 1. 保存完整对话到 .transcripts/ 目录（以防万一）
    save_transcript(messages)

    # 2. 让模型总结对话
    summary = client.messages.create(
        messages=[{"role": "user", "content": "请总结这段对话的要点..."}]
    )

    # 3. 用摘要替换所有消息
    return [
        {"role": "user", "content": f"[对话已压缩]\n{summary}"},
        {"role": "assistant", "content": "明白了，我已获得上下文摘要，继续工作。"},
    ]
```

### 小白重点理解

1. **微压缩为什么安全？** 3 轮之前的工具结果通常不再需要完整内容，一句话概括足够了。
2. **自动压缩会丢信息吗？** 完整对话存在磁盘上了（`.transcripts/` 目录），摘要只保留"做了什么、当前状态、关键决定"。
3. **为什么需要手动压缩？** 有时候模型比我们更清楚什么时候该压缩——比如刚完成一个大阶段，要开始新阶段时。

---

## 第七章：任务系统——大目标拆成小任务

### 问题

第三章的 TodoManager 只是内存里的清单，没有依赖关系，压缩后就丢了。

真实的项目目标是有结构的——**任务 B 得等任务 A 做完，任务 C 和 D 可以同时做，任务 E 要等 C 和 D 都做完**。

### 解决方案：磁盘持久化的任务图

```
.tasks/ 目录
├── task_1.json   {"id":1, "status":"completed"}
├── task_2.json   {"id":2, "blockedBy":[1], "status":"pending"}
├── task_3.json   {"id":3, "blockedBy":[1], "status":"pending"}
└── task_4.json   {"id":4, "blockedBy":[2,3], "status":"pending"}

任务依赖图 (DAG)：
                 ┌──────────┐
            ┌──→ │ 任务 2    │ ──┐
            │    │ pending   │   │
┌──────────┐     └──────────┘    ┌──→ ┌──────────┐
│ 任务 1    │                          │ 任务 4    │
│ completed │──→ ┌──────────┐    ┌──→ │ blocked   │
└──────────┘     │ 任务 3    │ ──┘    └──────────┘
                 │ pending   │
                 └──────────┘

当任务 1 完成时 → 自动解锁任务 2 和 3
当任务 2 和 3 都完成时 → 自动解锁任务 4
```

### 核心代码

```python
class TaskManager:
    def create(self, subject):
        task = {"id": self._next_id, "subject": subject,
                "status": "pending", "blockedBy": [], "owner": ""}
        # 存到 .tasks/task_N.json
        self._save(task)

    def _clear_dependency(self, completed_id):
        """任务完成时，自动解锁被它阻塞的任务"""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)
```

### 小白重点理解

1. **为什么存磁盘而不是内存？** 因为上下文压缩（第六章）会清空 messages，但磁盘上的文件不会丢。
2. **每个任务一个 JSON 文件**——简单粗暴但有效，不需要数据库。
3. **自动解锁机制**——任务 1 完成后，代码自动从任务 2、3 的 `blockedBy` 里删掉 1。

---

## 第八章：后台任务——慢操作丢后台

### 问题

有些命令要跑好几分钟：`npm install`、`pytest`、`docker build`。阻塞式循环下，模型只能干等，什么也做不了。

### 解决方案：后台线程 + 通知队列

```
主线程（Agent 循环）             后台线程
┌─────────────────┐             ┌─────────────────┐
│ 继续思考和工作    │             │ subprocess 执行  │
│ ...              │             │ ...              │
│ 每次调用模型前 ←─┼─────────── │ 完成后入队通知    │
│ 先排空通知队列    │             └─────────────────┘
└─────────────────┘

时间线：
Agent ──[丢后台 A]──[丢后台 B]──[继续做别的事]────
             |          |
             v          v
          [A 在跑]   [B 在跑]      （并行！）
             |          |
             +── 结果注入到下一次模型调用前 ──+
```

### 核心代码

```python
class BackgroundManager:
    def run(self, command):
        task_id = str(uuid.uuid4())[:8]
        # 启动守护线程，立即返回
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True)
        thread.start()
        return f"后台任务 {task_id} 已启动"

    def _execute(self, task_id, command):
        result = subprocess.run(command, shell=True, ...)
        # 完成后，把结果放入通知队列
        self._notification_queue.append({"task_id": task_id, "result": result})

    def drain_notifications(self):
        """排空通知队列，返回所有完成的结果"""
        notifications = list(self._notification_queue)
        self._notification_queue.clear()
        return notifications
```

### 小白重点理解

1. **"丢后台"是什么意思？** 就是开一个新线程去执行命令，主线程继续运行。就像你把洗衣机打开后去做饭，不用站在洗衣机旁边等。
2. **通知队列**——后台任务完成后不是直接打断模型，而是放进队列。下次模型调用前，统一取出来告诉它。
3. **daemon=True**——守护线程，主程序退出时自动结束，不会留下"僵尸进程"。

---

## 第九章：Agent 团队——从单兵作战到团队协作

### 问题

子 Agent（第四章）是一次性的：干完活就消失了，没有身份，不能跨任务记忆。

如果你需要一个"开发者 Alice"和一个"测试员 Bob"长期协作呢？

### 解决方案：持久化队友 + 邮箱通信

```
.team/ 目录
├── config.json          ← 团队名册（谁在队里，什么状态）
└── inbox/
    ├── alice.jsonl      ← Alice 的收件箱（追加写入，读后清空）
    ├── bob.jsonl        ← Bob 的收件箱
    └── lead.jsonl       ← 领导的收件箱

通信方式：
    ┌────────┐   发消息到 bob.jsonl   ┌────────┐
    │ Alice  │ ────────────────────→ │  Bob   │
    │ 循环   │                        │  循环   │
    └────────┘                        └────────┘
         ↑                                 │
         │   发消息到 alice.jsonl           │
         └─────────────────────────────────┘
```

### 核心代码

```python
class MessageBus:
    def send(self, sender, to, content):
        msg = {"from": sender, "content": content, "timestamp": time.time()}
        # 追加写入对方的收件箱
        with open(self.dir / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")

    def read_inbox(self, name):
        path = self.dir / f"{name}.jsonl"
        msgs = [json.loads(l) for l in path.read_text().splitlines() if l]
        path.write_text("")  # 读后清空
        return msgs
```

队友生成：

```python
class TeammateManager:
    def spawn(self, name, role, prompt):
        # 在新线程中启动一个完整的 agent loop
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt), daemon=True)
        thread.start()
        return f"已生成队友 '{name}'（角色: {role}）"
```

### 小白重点理解

1. **每个队友就是一个独立的 agent loop**，跑在自己的线程里。
2. **JSONL 邮箱**——每行一条消息，追加写入，读后清空。比数据库简单得多。
3. **生命周期**：`spawn → working → idle → working → ... → shutdown`。不像子 Agent 那样用完就丢。

---

## 第十章：团队协议——队友之间的沟通规矩

### 问题

队友能通信了，但缺少**结构化的协调**：

- **关机**：直接杀线程会留下写了一半的文件。需要"请求-批准"握手。
- **计划审批**：高风险任务（比如删库重构）应该先过审再动手。

### 解决方案：请求-响应模式

两种协议，结构完全一样：

```
关机协议                         计划审批协议
==========                       ============

领导             队友             队友             领导
  |                 |              |                 |
  |──关机请求──→   |              |──计划请求──→   |
  | {req_id:"abc"} |              | {req_id:"xyz"} |
  |                 |              |                 |
  |←──关机响应──   |              |←──审批响应──   |
  | {req_id:"abc", |              | {req_id:"xyz", |
  |  approve:true} |              |  approve:true} |

共享状态机：
  [待处理] ──批准──→ [已批准]
  [待处理] ──拒绝──→ [已拒绝]
```

### 小白重点理解

1. **为什么需要 request_id？** 为了把请求和响应对应起来。就像快递单号——"你回复的是我哪个请求？"
2. **状态机只有三个状态**：pending → approved / rejected。简单但足够。
3. **领导和队友是对等的**——领导可以请求关机，队友也可以请求审批。

---

## 第十一章：自治 Agent——自己找活干

### 问题

到现在为止，队友只在被明确指派时才工作。任务看板上有 10 个未认领的任务，得领导手动分配。这扩展不了。

### 解决方案：空闲轮询 + 自动认领

```
队友的生命周期：

┌───────┐
│ 生成   │
└───┬───┘
    ↓
┌───────┐  模型调工具   ┌───────┐
│ 工作   │ ←──────── │  LLM  │
└───┬───┘              └───────┘
    │
    │ 模型说"我做完了"
    ↓
┌────────┐
│  空闲   │  每 5 秒轮询一次，最多等 60 秒
└───┬────┘
    │
    ├──→ 检查收件箱 → 有消息？ ──────→ 继续工作
    │
    ├──→ 扫描任务板 → 有未认领的？ ──→ 认领 → 继续工作
    │
    └──→ 60 秒超时 ──────────────────→ 自动关机
```

### 核心代码

```python
def _idle_poll(self, name, messages):
    for _ in range(12):  # 60秒 / 5秒 = 12 次
        time.sleep(5)

        # 先检查收件箱
        inbox = BUS.read_inbox(name)
        if inbox:
            return True  # 有消息，回去工作

        # 再扫描任务看板
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            claim_task(unclaimed[0]["id"], name)  # 自动认领
            return True  # 有活干，回去工作

    return False  # 60秒没活 → 自动关机
```

还有一个细节——**身份重注入**。上下文压缩后，模型可能忘了自己是谁：

```python
if len(messages) <= 3:  # 压缩后消息很少，说明刚被压缩过
    messages.insert(0, {"role": "user",
        "content": "<identity>你是 'Alice'，角色: coder</identity>"})
```

### 小白重点理解

1. **自治 = 自己找活干**，不需要领导逐个分配。
2. **空闲轮询两件事**：收件箱（有人找我吗？）+ 任务看板（有没人认领的任务吗？）。
3. **60 秒超时自动关机**——防止空闲线程永远挂着浪费资源。
4. **身份重注入**——压缩后模型可能以为自己是"领导"，需要重新告诉它"你是 Alice"。

---

## 第十二章：Worktree 隔离——各干各的目录

### 问题

所有 Agent 共享一个目录。Alice 在改 `config.py`，Bob 也在改 `config.py`——未提交的改动互相污染，谁也没法干净回滚。

### 解决方案：Git Worktree + 任务绑定

每个任务有自己独立的目录（通过 Git Worktree 实现）：

```
控制面（.tasks/）                执行面（.worktrees/）
┌──────────────────┐             ┌──────────────────────┐
│ task_1.json       │             │ auth-refactor/        │
│   status: 进行中  │ ←──绑定──→ │   branch: wt/auth-... │
│   worktree: "auth"│             │   task_id: 1          │
├──────────────────┤             ├──────────────────────┤
│ task_2.json       │             │ ui-login/             │
│   status: 待处理  │ ←──绑定──→ │   branch: wt/ui-...   │
│   worktree: "ui"  │             │   task_id: 2          │
└──────────────────┘             └──────────────────────┘
```

### 核心代码

创建 worktree 并绑定任务：

```python
def create(self, name, task_id):
    # 创建独立的 Git Worktree
    # git worktree add -b wt/auth-refactor .worktrees/auth-refactor HEAD
    self._run_git(["worktree", "add", "-b", branch, path, "HEAD"])

    # 双向绑定：worktree 记录 task_id，task 记录 worktree
    self.tasks.bind_worktree(task_id, name)
```

在 worktree 中执行命令：

```python
# 命令在独立目录中执行，不影响其他 worktree
subprocess.run(command, shell=True, cwd=worktree_path)
```

收尾时的两个选择：
- **保留**（`worktree_keep`）：目录留着，以后还要用
- **删除**（`worktree_remove`）：删除目录 + 标记任务完成 + 记录事件

### 小白重点理解

1. **Git Worktree 是什么？** Git 自带的功能——一个仓库可以同时 checkout 到多个目录。每个目录有独立的工作区，但共享同一个 `.git`。
2. **为什么需要绑定？** 任务知道"在哪个目录干活"，目录知道"服务于哪个任务"。双向关联。
3. **事件流**（`.worktrees/events.jsonl`）——记录所有生命周期事件，方便审计和调试。

---

## 全景回顾：12 个 Session 的演进路径

```
第一阶段：循环                   第二阶段：规划与知识
==========                       ================
s01 Agent 循环                   s03 TodoWrite 待办管理
  while + stop_reason              TodoManager + 催促机制
  （核心模式，永不改变）             （让模型不丢步骤）
    │                                │
    └→ s02 工具分发                s04 子 Agent
         TOOL_HANDLERS 字典           独立 messages[]
         （加工具只加一行）            （上下文隔离）
                                      │
                                   s05 技能加载
                                     两层注入
                                     （按需加载知识）
                                      │
                                   s06 上下文压缩
                                     三层策略
                                     （无限会话）

第三阶段：持久化                 第四阶段：团队
==========                       ==========
s07 任务系统                     s09 Agent 团队
  文件 CRUD + 依赖图               队友 + JSONL 邮箱
  （任务比对话更长命）               （多模型协作）
    │                                │
s08 后台任务                     s10 团队协议
  守护线程 + 通知队列               请求-响应握手
  （模型不用干等）                   （结构化协调）
                                      │
                                   s11 自治 Agent
                                     空闲轮询 + 自动认领
                                     （自己找活干）
                                      │
                                   s12 Worktree 隔离
                                     任务 + 目录绑定
                                     （并行不冲突）
```

---

## 核心设计模式总结

学完 12 个 Session，你应该记住这 5 条原则：

### 原则 1：循环不变

```
所有 12 个 Session 的核心 while 循环完全一样。
新功能 = 在循环外围加 handler 和 manager。
循环是模型的，代码只是搭台子。
```

### 原则 2：工具分发是字典

```python
TOOL_HANDLERS = {name: handler}
# 加一个工具 = 加一行。100 个工具也不混乱。
```

### 原则 3：知识通过 tool_result 注入

```
不要往 system prompt 塞大段文字。
技能按需加载，通过 <skill>...</skill> 标签包装。
出现在对话最近位置，模型最容易注意到。
```

### 原则 4：子 Agent 隔离上下文

```
子 Agent = 独立的 messages[]
只返回摘要，不污染父对话。
想象成：交给助理一个调研任务，只收报告。
```

### 原则 5：全程文件持久化

```
JSON 存任务（.tasks/）
JSONL 存邮箱（.team/inbox/）
JSONL 存事件（.worktrees/events.jsonl）
不依赖任何数据库。简单就是美。
```

---

## 动手试试

### 环境准备

```bash
# 1. 克隆仓库
git clone <repo-url>
cd learn-claude-code

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 ANTHROPIC_API_KEY 和 MODEL_ID
```

### 从第一个 Session 开始

```bash
# 最简单的 Agent——只有一个 while 循环和一个 bash 工具
python agents/s01_agent_loop.py

# 试试这些 prompt：
# "创建一个 hello.py 文件并运行它"
# "列出当前目录下的所有 Python 文件"
# "查看当前 git 分支"
```

### 逐步进阶

```bash
python agents/s02_tool_use.py      # 多工具 + 路径沙箱
python agents/s03_todo_write.py    # 待办管理 + 催促机制
python agents/s04_subagent.py      # 子 Agent 上下文隔离
python agents/s05_skill_loading.py # 按需加载技能
python agents/s06_context_compact.py # 三层压缩
# ... 以此类推
```

### 跑总集（所有功能整合）

```bash
python agents/s_full.py

# REPL 命令：
# /compact  手动压缩上下文
# /tasks    查看任务看板
# /team     查看团队状态
# /inbox    检查收件箱
```

---

## 最后的话

构建 AI Agent 的核心，不是写多复杂的代码，而是理解**模型需要什么支持**。

- 模型需要**行动能力**？→ 给它工具。
- 模型需要**记忆**？→ 给它待办清单和任务系统。
- 模型需要**知识**？→ 按需加载技能。
- 模型需要**更大的上下文**？→ 压缩旧内容。
- 模型需要**协作**？→ 给它队友和邮箱。
- 模型需要**隔离**？→ 给它独立的工作目录。

**模型是 Agent。你的代码是 Harness。**

> *"Build great harnesses. The agent will do the rest."*
>
> 搭好台子，Agent 自己会演好戏。
