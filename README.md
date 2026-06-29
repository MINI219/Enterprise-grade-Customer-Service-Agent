# 智能客服 Agent 系统 —— 技术文档 

> **项目版本**: v0.5.0
> **项目定位**: 全栈 AI 智能客服后端系统，涵盖 LLM Agent、RAG 检索增强、长期记忆与用户画像、MCP 标准协议、Web 聊天界面
> **v0.5.0 重大更新**: FAISS → ChromaDB (Client/Server)、all-MiniLM-L6-v2 → BAAI/bge-small-zh-v1.5、Metadata Filtering、弹性设计(重试+熔断)

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
│  FastAPI 0.115  +  Uvicorn 0.34  +  Pydantic 2.10  │
├─────────────────────────────────────────────────────┤
│                    AI / Agent 层                     │
│  LangChain 0.3  +  LangChain-OpenAI  +  DeepSeek    │
├─────────────────────────────────────────────────────┤
│                 长期记忆 / 用户画像                    │
│  SQLite (sqlite3)  +  FastAPI BackgroundTasks       │
│  +  LLM 驱动的偏好抽取 Pipeline                      │
├─────────────────────────────────────────────────────┤
│                    RAG 检索层                        │
│  ChromaDB 0.5 (Docker C/S) + BAAI/bge-small-zh-v1.5 │
│  (512维 中文优化) + LangChain-Chroma + 弹性重试       │
├─────────────────────────────────────────────────────┤
│                    MCP 协议层                        │
│  mcp 1.28  +  sse-starlette 3.4                     │
├─────────────────────────────────────────────────────┤
│                    工具 & 基础设施                    │
│  Loguru (日志)  +  httpx  +  tenacity (重试)         │
│  +  CircuitBreaker (熔断)  +  Docker (ChromaDB)      │
└─────────────────────────────────────────────────────┘
```

### 2.1 FastAPI（Web 框架）

**选型理由**: 异步原生支持、自动 OpenAPI 文档生成、Pydantic 深度集成、类型安全。

- `lifespan` 上下文管理器：管理应用启动 / 关闭生命周期，并在启动时初始化 `VectorStoreService`
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
- **v0.5.0 更新**: `run_agent()` 改为 `async def`，使用 `executor.ainvoke()` 异步执行，避免阻塞事件循环

### 2.3 RAG（检索增强生成）—— v0.5.0 重构

**架构**: Client/Server 模式 —— ChromaDB 通过 Docker 独立部署，Python 后端仅通过 HTTP 通信。

```
faq.md → TextLoader → RecursiveCharacterTextSplitter
→ HuggingFaceEmbeddings (bge-small-zh-v1.5, 512维)
→ ChromaDB Docker Server (localhost:8001)
→ search_faq(query, user_profile=None) → 业务标签过滤 + 语义检索
```

| 组件      | v0.4.0 (旧)               | v0.5.0 (新)                        | 升级理由                                          |
| --------- | ------------------------- | ---------------------------------- | ------------------------------------------------- |
| 向量存储  | FAISS (IndexFlatL2)       | ChromaDB 0.5 (Docker C/S)          | 原生 Metadata 过滤、CRUD 友好、Client/Server 解耦 |
| Embedding | all-MiniLM-L6-v2 (384维)  | BAAI/bge-small-zh-v1.5 (512维)     | 中文语义理解显著提升、BAAI 专为中文优化           |
| 架构      | 嵌入式（本地 FAISS 文件） | Client/Server（Docker HTTP）       | 不阻塞事件循环、独立扩缩容                        |
| 元数据    | 不支持                    | `policy_type` + `level_*` 布尔字段 | 支持业务标签预过滤                                |
| 弹性      | 无                        | 重试(tenacity) + 断路器            | 应对网络抖动、ChromaDB 临时不可用                 |

**Embedding 模型对比**:

| 特性          | all-MiniLM-L6-v2                  | BAAI/bge-small-zh-v1.5 |
| ------------- | --------------------------------- | ---------------------- |
| 开发者        | Microsoft / Sentence-Transformers | BAAI（北京智源研究院） |
| 维度          | 384                               | 512                    |
| 中文优化      | 否（多语言通用）                  | 是（专为中文检索设计） |
| 模型大小      | ~80 MB                            | ~95 MB                 |
| MTEB 中文排名 | 中等                              | 同尺寸最优             |

**Metadata Filtering（元数据过滤）—— v0.5.0 核心特性**:

文档入库时自动打标：

- `policy_type`: `"return_exchange"` / `"invoice"` / `"logistics"` / `"membership"` / `"after_sales"`
- `level_basic` / `level_silver` / `level_gold` / `level_diamond`: 布尔值（会员等级适用性）

检索时先业务标签过滤再语义检索：

```
用户画像 {"card_type": "times_card", "level": "gold"}
    ↓
ChromaDB where={"level_gold": True}   ← 仅检索金卡相关规则
    ↓
语义匹配 (cosine distance)           ← 在过滤结果中做语义排序
    ↓
若结果不足 k 条 → 补充全库检索并去重合并
```

### 2.4 MCP（Model Context Protocol）

**选型理由**: Anthropic 提出的 AI 工具调用标准协议，让任何 MCP 客户端（Claude Desktop、Codex 等）都能发现并调用本系统的工具。

- **Server**: `FastMCP`（mcp 包的高级 API）
- **传输层**: SSE（Server-Sent Events）—— `GET /mcp/sse` 建立长连接，`POST /mcp/messages` 接收 JSON-RPC 消息
- **注册工具**: 4 个 `@mcp.tool()`，与 LangChain Tool 共享同一套底层服务
- **v0.5.0 更新**: `search_knowledge_base` MCP Tool 改为 `async def`，适配异步 ChromaDB 检索

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

**v0.5.0 更新**: 新增 `get_user_facts_raw()` 返回原始 dict，直接注入到 ChromaDB 的 `where` 过滤条件中，实现"用户画像 → 业务标签过滤 → 语义检索"的闭环。

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
                    │  [async]               │  [异步 BackgroundTasks] │
                    ▼                        ▼                        │
      ┌──────────────────────┐   ┌──────────────────────┐            │
      │    agent/core.py      │   │  memory_manager.py   │            │
      │  Tool-Calling Agent   │   │  LLM 画像抽取        │            │
      │  + 短期会话历史       │   │  → SQLite 写入       │            │
      │  + 长期画像注入 ←─────┼───│  ← 对话历史读取      │            │
      │  + ChromaDB 过滤上下文 │   └──────────┬───────────┘            │
      └──────┬──────┬──────┬──┘              │                        │
             │      │      │                 │                        │
  ┌──────────▼┐ ┌──▼──────▼──┐     ┌────────▼──────────┐             │
  │ RAG Tool  │ │ 业务 Tools  │     │ user_memory.db    │             │
  │ (async)   │ │ 订单/物流/投诉│    │ (SQLite 画像存储)  │             │
  └─────┬─────┘ └──────┬──────┘     └───────────────────┘             │
        │              │                                               │
  ┌─────▼──────────┐ ┌─▼──────────────┐                                │
  │ VectorStore    │ │ mock_services  │                                │
  │ Service (async)│ │ (模拟后端数据)  │                                │
  │ ┌────────────┐ │ └────────────────┘                                │
  │ │ bge-small   │ │                                                  │
  │ │ -zh-v1.5    │ │                                                  │
  │ │ (512维)     │ │                                                  │
  │ └─────┬──────┘ │                                                  │
  │       │ HTTP   │                                                  │
  └───────┼────────┘                                                  │
          │                                                           │
  ┌───────▼──────────────────────────────────────┐                    │
  │  ChromaDB Docker Container                   │                    │
  │  chromadb/chroma:0.5.23                      │                    │
  │  Port: 8001 (host) → 8000 (container)        │                    │
  │  Volume: agent_chroma_data                   │                    │
  │  ┌─────────────────────────────────────────┐ │                    │
  │  │ Collection: cs_faq                      │ │                    │
  │  │ Metadata: policy_type, level_*, card_type│ │                    │
  │  └─────────────────────────────────────────┘ │                    │
  └──────────────────────────────────────────────┘                    │
                                                                      │
  ┌──────────────────────────────────────────────┐                    │
  │  弹性层 (services/retry.py)                   │                    │
  │  tenacity 重试 (指数退避+抖动) + CircuitBreaker│                    │
  └──────────────────────────────────────────────┘                    │
```

**数据流（完整链路）**:

1. **Web UI / REST API** → `/api/chat` → `await run_agent()`
   - 读取 SQLite 用户画像 → 注入 System Prompt（自然语言）
   - 读取 SQLite 用户画像 → `set_current_profile()`（ChromaDB 过滤上下文）
   - `await executor.ainvoke()` → Agent 选择工具
   - RAG Tool → `await search_faq(query)` → 自动从上下文读取 profile → 构建 ChromaDB where 过滤 → `VectorStoreService.search()` → `asyncio.to_thread()` → ChromaDB HTTP 查询
   - 返回回复 → BackgroundTasks 触发 LLM 画像抽取
2. **MCP Client** → `/mcp/sse` → `mcp_server.py` → 直接调用底层 Services + RAG

## 四、项目文件清单

```
AgentService-v2.0/
├── .env.example              # 环境变量模板（DeepSeek + ChromaDB + Embedding 配置）
├── .mcp.json                 # MCP 客户端连接配置
├── docker-compose.yml        # ChromaDB Docker 独立部署（v0.5.0 新增）
├── requirements.txt          # Python 依赖清单（17 个包 + 补丁）
├── run.py                    # 开发环境启动入口
├── mcp_server.py             # MCP Server（4 个 Tool，async 知识库检索）
├── PROJECT_DOCS.md           # 本文档
├── app/
│   ├── main.py               # FastAPI 入口 + lifespan（初始化 VectorStoreService）
│   ├── core/
│   │   └── logger.py         # Loguru 日志配置（3 路输出 + 全链路追踪）
│   └── routers/
│       ├── chat.py           # /api/chat (async run_agent) + 会话管理
│       ├── chat_ui.py        # / Web 聊天界面（内联 HTML+CSS+JS）
│       └── mcp_router.py     # /mcp/sse + /mcp/messages MCP SSE 传输
├── agent/
│   ├── core.py               # LangChain Agent 核心（async + 画像注入 + ChromaDB 过滤上下文）
│   └── memory_manager.py     # 长期记忆管理器（SQLite + LLM 画像抽取 + get_user_facts_raw）
├── tools/
│   └── agent_tools.py        # 4 个 LangChain Tool（async search_knowledge_base）
├── services/
│   ├── mock_services.py      # 3 个 Mock 业务函数（订单/物流/投诉）
│   └── retry.py              # 弹性设计（tenacity 重试 + 断路器 CircuitBreaker）[v0.5.0 新增]
├── rag/
│   └── retriever.py          # RAG 管道 [v0.5.0 重构]
│       │                     #   - VectorStoreService（async ChromaDB 客户端）
│       │                     #   - Metadata 自动打标（policy_type + level_*）
│       │                     #   - build_where_filter（画像 → ChromaDB where）
│       │                     #   - search_faq（两阶段：过滤检索 + 全库补充）
├── models/
│   └── schemas.py            # Pydantic 数据模型
├── data/
│   ├── faq.md                # 电商 FAQ 知识库（5 大类政策文档）
│   └── user_memory.db        # SQLite 用户画像数据库（运行时生成）
└── logs/                     # 日志文件目录（运行时生成）
```

**v0.4.0 → v0.5.0 变更摘要**:

| 文件                      | 变更     | 说明                                                         |
| ------------------------- | -------- | ------------------------------------------------------------ |
| `docker-compose.yml`      | **新增** | ChromaDB 0.5.23 Docker 独立部署                              |
| `requirements.txt`        | 修改     | `faiss-cpu` 移除；`chromadb`, `langchain-chroma`, `tenacity` 新增 |
| `.env.example`            | 修改     | 新增 `CHROMA_HOST/PORT/COLLECTION`, `EMBEDDING_MODEL/DEVICE` |
| `rag/retriever.py`        | **重写** | FAISS → ChromaDB + VectorStoreService + Metadata Filtering   |
| `services/retry.py`       | **新增** | tenacity 重试 + CircuitBreaker 断路器                        |
| `agent/core.py`           | 修改     | `run_agent()` async；用户画像双通道注入                      |
| `agent/memory_manager.py` | 修改     | 新增 `get_user_facts_raw()`                                  |
| `tools/agent_tools.py`    | 修改     | `search_knowledge_base` 改为 async                           |
| `mcp_server.py`           | 修改     | `search_knowledge_base` MCP Tool 改为 async                  |
| `app/routers/chat.py`     | 修改     | `await run_agent()`                                          |
| `app/main.py`             | 修改     | lifespan 中初始化 VectorStoreService                         |
| `data/faiss_index/`       | **废弃** | 替换为 Docker Volume `agent_chroma_data`                     |

---

## 五、快速开始

```bash
# 0. 前置条件：Docker 已安装并运行

# 1. 启动 ChromaDB 容器
docker compose up -d
# 验证: curl http://localhost:8001/api/v2/heartbeat

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-real-key

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务（首次自动从 FAQ 构建向量索引）
python run.py

# 5. 使用
# - Web UI:          http://localhost:8000/
# - API 文档:        http://localhost:8000/docs
# - MCP 连接配置:    查看 .mcp.json
# - 健康检查:        http://localhost:8000/health

# 6. 测试（需要真实 API Key 才能获得 LLM 回复）
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U10086","message":"退货邮费谁承担？"}'

# 7. 测试 Metadata Filtering（先写入用户画像，再检索）
# 写入金卡会员画像
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U20000","message":"我是金卡会员，想了解一下退货政策"}'
# Agent 会自动从 SQLite 读取 level=gold 画像，
# ChromaDB 检索时会优先返回金卡/钻石卡相关的规则
```
