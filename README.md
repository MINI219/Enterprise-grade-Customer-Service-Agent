# 智能客服 Agent 系统 —— 技术文档 

> **项目版本**: v0.4.0  
> **项目定位**: 全栈 AI 智能客服后端系统，涵盖 LLM Agent、RAG 检索增强、长期记忆与用户画像、MCP 标准协议、Web 聊天界面  

---

## 一、项目概述

本项目是一个面向电商场景的**智能客服 Agent 后端服务**，支持用户通过自然语言查询订单、追踪物流、查询投诉工单、检索公司政策知识库。系统同时提供三套对外接口：

| 接口层      | 路径                                  | 用途                                             |
| ----------- | ------------------------------------- | ------------------------------------------------ |
| Web 聊天 UI | `GET /`                               | 浏览器直接使用的图形化聊天界面                   |
| REST API    | `POST /api/chat`                      | 标准 HTTP JSON 接口，供前端 / 第三方调用         |
| MCP SSE     | `GET /mcp/sse` + `POST /mcp/messages` | MCP 标准协议，供 Claude Desktop 等 AI 客户端调用 |

---

## 二、关键技术栈

```
┌─────────────────────────────────────────────────────┐
│                    Web 框架层                        │
│  FastAPI 0.138  +  Uvicorn 0.34  +  Pydantic 2.10  │
├─────────────────────────────────────────────────────┤
│                    AI / Agent 层                     │
│  LangChain 0.3  +  LangChain-OpenAI  +  DeepSeek    │
├─────────────────────────────────────────────────────┤
│                 长期记忆 / 用户画像                    │
│  SQLite (sqlite3)  +  FastAPI BackgroundTasks       │
│  +  LLM 驱动的偏好抽取 Pipeline                      │
├─────────────────────────────────────────────────────┤
│                    RAG 检索层                        │
│  FAISS 1.14  +  Sentence-Transformers 3.4           │
│  (all-MiniLM-L6-v2)  +  LangChain-Community         │
├─────────────────────────────────────────────────────┤
│                    MCP 协议层                        │
│  mcp 1.28  +  sse-starlette 3.4                     │
├─────────────────────────────────────────────────────┤
│                    工具 & 基础设施                    │
│  Loguru (日志)  +  python-multipart  +  httpx        │
└─────────────────────────────────────────────────────┘
```

### 2.1 FastAPI（Web 框架）

**选型理由**: 异步原生支持、自动 OpenAPI 文档生成、Pydantic 深度集成、类型安全。

- `lifespan` 上下文管理器：管理应用启动 / 关闭生命周期
- `CORSMiddleware`：允许跨域（前端调试友好）
- 自定义 HTTP 中间件：为每个请求生成 `request_id` 并注入日志上下文
- 三个子路由：`chat_ui`（HTML 页面）、`chat`（Agent API）、`mcp_router`（SSE 长连接）

### 2.2 LangChain + DeepSeek（LLM Agent）

**选型理由**: LangChain 提供统一的 Agent 抽象层（`create_tool_calling_agent`），DeepSeek 兼容 OpenAI 协议，性价比高。

- **模型**: `deepseek-chat`，通过 `ChatOpenAI(base_url="https://api.deepseek.com")` 调用
- **Agent 模式**: Tool-Calling Agent（原生 Function Calling），而非旧版 ReAct 文本模板
- **AgentExecutor 配置**: `temperature=0.3`（客服场景要求稳定）、`max_iterations=5`、`handle_parsing_errors=True`（自动修复格式错误）
- **会话管理**: `InMemoryChatMessageHistory`，以 `conversation_id` 为 key 存储多轮对话上下文
- **System Prompt**: 精心设计的结构化提示词，定义 persona（"小智"）、工具选择优先级、回复规范

### 2.3 RAG（检索增强生成）

**流程**: `faq.md → TextLoader → RecursiveCharacterTextSplitter → HuggingFaceEmbeddings → FAISS`

| 组件     | 技术选型                         | 说明                                 |
| -------- | -------------------------------- | ------------------------------------ |
| 文档加载 | `TextLoader` (LangChain)         | 读取 Markdown 文件                   |
| 文本切块 | `RecursiveCharacterTextSplitter` | chunk_size=500, overlap=100          |
| 向量化   | `all-MiniLM-L6-v2` (384维)       | 轻量、多语言、CPU 可用               |
| 向量存储 | FAISS (IndexFlatL2)              | Meta 开源的向量检索库，无 GPU 依赖   |
| 持久化   | `faiss_index/` 目录              | `index.faiss` + `index.pkl` 磁盘文件 |

**RAG 为什么重要**: 解决 LLM "幻觉"和"缺乏私有知识"两大痛点。当用户问"退货邮费谁出"时，Agent 不靠记忆编造，而是从向量库中检索《1.3 邮费规则》原文。

### 2.4 MCP（Model Context Protocol）

**选型理由**: Anthropic 提出的 AI 工具调用标准协议，让任何 MCP 客户端（Claude Desktop、Codex 等）都能发现并调用本系统的工具。

- **Server**: `FastMCP`（mcp 包的高级 API）
- **传输层**: SSE（Server-Sent Events）—— `GET /mcp/sse` 建立长连接，`POST /mcp/messages` 接收 JSON-RPC 消息
- **注册工具**: 4 个 `@mcp.tool()`，与 LangChain Tool 共享同一套底层服务

### 2.5 日志系统（Loguru）

- 三路输出：控制台（彩色 DEBUG） + 文件（按天轮转，30 天保留） + 错误文件（90 天保留）
- 全链路追踪：`logger.contextualize(request_id=xxx)` 确保同一请求的所有日志携带相同 ID
- 兜底机制：`logger.configure(extra={"request_id": "-"})` 防止非 HTTP 上下文调用时报 KeyError

### 2.6 长期记忆 & 用户画像（Long-Term Memory）

**选型理由**: 短期记忆（会话历史）随会话结束而消失；长期记忆跨越对话、持续积累用户偏好，让 Agent 从"一次性工具"进化为"越来越懂你的私人管家"。

**架构**:

```
每次对话结束（BackgroundTasks 异步）:
  对话记录 → LLM 分析 → 提取偏好 JSON → SQLite 持久化
                                            │
下次对话开始（run_agent 同步注入）:            │
  Agent 输入 ← 【用户画像】← JSON 反序列化 ←──┘
```

| 组件     | 技术选型                                                 | 说明                                           |
| -------- | -------------------------------------------------------- | ---------------------------------------------- |
| 存储     | Python 内置 `sqlite3`                                    | 零外部依赖，单文件数据库 `data/user_memory.db` |
| 表结构   | `user_profiles(user_id TEXT PK, facts TEXT, updated_at)` | facts 为 JSON 格式，适应画像字段的灵活多变     |
| 抽取方式 | DeepSeek LLM 异步调用                                    | 专用 Prompt 引导 LLM 从对话中提取 6 类特征     |
| 异步机制 | FastAPI `BackgroundTasks`                                | 记忆抽取作为后台任务，**不阻塞 HTTP 响应**     |
| 注入时机 | `run_agent()` 中同步查询                                 | 每次对话前读取 SQLite，将画像注入 Agent 输入   |
| 合并策略 | 新 key 追加，已有 key 覆盖                               | JSON 级别 merge，最新信息优先                  |

**LLM 抽取的 6 个维度**:

1. **物流偏好** — 偏好的快递公司、收货时间偏好
2. **消费习惯** — 常购品类、价格敏感度、是否偏好促销
3. **沟通风格** — 喜欢简洁还是详细、是否急躁
4. **个人信息** — 称呼偏好、所在城市、会员等级
5. **投诉倾向** — 容易因什么问题投诉、投诉频率
6. **其他特征** — 任何值得记录的固定属性

**Prompt 设计要点**:

- 温度设为 `0.1`（极低），确保 JSON 输出稳定可解析
- 传入已有画像避免重复提取
- 强制 JSON 输出格式 `{"facts": {...}}`
- 无有效特征时返回 `{}` 跳过写入

---

## 三、项目架构图

```
                        ┌──────────────────────┐
                        │   MCP Client          │
                        │ (Claude Desktop 等)    │
                        └──────┬───────┬───────┘
                               │ SSE   │ POST
                               ▼       ▼
┌──────────┐   POST /api/chat  ┌──────────────────────────────┐
│  Web UI  │ ────────────────▶ │       FastAPI :8000           │
│  /       │ ◀──────────────── │                              │
└──────────┘   {reply, conv_id}│  /mcp/sse ─── mcp_router ────┤
                               │  /mcp/msg ─── mcp_router ────┤
                               │  /api/chat ── chat router ───┤
                               │  /          ── chat_ui ──────┤
                               └─────────────┬────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │  [同步]                │  [异步 BackgroundTasks] │
                    ▼                        ▼                        │
      ┌──────────────────────┐   ┌──────────────────────┐            │
      │    agent/core.py      │   │  memory_manager.py   │            │
      │  Tool-Calling Agent   │   │  LLM 画像抽取        │            │
      │  + 短期会话历史       │   │  → SQLite 写入       │            │
      │  + 长期画像注入 ←─────┼───│  ← 对话历史读取      │            │
      └──────┬──────┬──────┬──┘   └──────────┬───────────┘            │
             │      │      │                 │                        │
  ┌──────────▼┐ ┌──▼──────▼──┐     ┌────────▼──────────┐             │
  │ RAG Tool  │ │ 业务 Tools  │     │ user_memory.db    │             │
  │ (知识库)   │ │ 订单/物流/投诉│    │ (SQLite 画像存储)  │             │
  └─────┬─────┘ └──────┬──────┘     └───────────────────┘             │
        │              │                                               │
  ┌─────▼─────┐ ┌──────▼────────┐                                     │
  │ FAISS     │ │ mock_services │                                     │
  │ all-MiniLM│ │ (模拟后端数据) │                                     │
  └─────┬─────┘ └───────────────┘                                     │
        │                                                              │
  ┌─────▼─────┐                                                       │
  │faq.md     │                                                       │
  │(政策文档)  │                                                       │
  └───────────┘                                                       │
```

**同步通路（用户感知）+ 异步通路（后台静默）**:

1. **Web UI** → `/api/chat` → `agent/core.py`（画像注入→回复） → BackgroundTasks → `memory_manager.py`（画像抽取）
2. **REST API** → 同上
3. **MCP Client** → `/mcp/sse` → `mcp_server.py` → 直接调用 Services

## 四、项目文件清单

```
AgentService/
├── .env.example              # 环境变量模板（DeepSeek API Key 等）
├── .mcp.json                 # MCP 客户端连接配置
├── requirements.txt          # Python 依赖清单（6 组共 16 个包）
├── run.py                    # 开发环境启动入口
├── mcp_server.py             # MCP Server（4 个 Tool 注册）
├── app/
│   ├── main.py               # FastAPI 入口 + 中间件 + 路由注册
│   ├── core/
│   │   └── logger.py         # Loguru 日志配置（3 路输出 + 全链路追踪）
│   └── routers/
│       ├── chat.py           # /api/chat 对话接口 + /api/chat/session/{id} 会话管理
│       ├── chat_ui.py        # / Web 聊天界面（内联 HTML+CSS+JS）
│       └── mcp_router.py     # /mcp/sse + /mcp/messages MCP SSE 传输
├── agent/
│   ├── core.py               # LangChain Agent 核心（DeepSeek + Tool Calling + 会话管理 + 画像注入）
│   └── memory_manager.py     # 长期记忆管理器（SQLite + LLM 画像抽取 + BackgroundTasks）
├── tools/
│   └── agent_tools.py        # 4 个 LangChain Tool（订单/物流/投诉/RAG）
├── services/
│   └── mock_services.py      # 3 个 Mock 业务函数（模拟后端数据）
├── rag/
│   └── retriever.py          # RAG 管道（FAISS + all-MiniLM-L6-v2）
├── models/
│   └── schemas.py            # Pydantic 数据模型
├── data/
│   ├── faq.md                # 电商 FAQ 知识库（5 大类政策文档）
│   └── user_memory.db        # SQLite 用户画像数据库（运行时生成）
├── faiss_index/              # FAISS 向量索引持久化目录（运行时生成）
└── logs/                     # 日志文件目录（运行时生成）
```

---

## 五、快速开始

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-real-key

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python run.py

# 4. 使用
# - Web UI:          http://localhost:8000/
# - API 文档:        http://localhost:8000/docs
# - MCP 连接配置:    查看 .mcp.json
# - 健康检查:        http://localhost:8000/health

# 5. 测试（需要真实 API Key 才能获得 LLM 回复）
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U10086","message":"退货邮费谁承担？"}'
```
