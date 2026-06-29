"""
Agent 核心模块
基于 LangChain + DeepSeek API，实现 ReAct 风格的 Tool-Calling Agent：

  用户消息 → LLM 理解意图 → 选择工具 → 执行获取结果 → 汇总自然语言回答
                ↑                                              |
                └────────── 多轮对话上下文追踪 ─────────────────┘

支持：
  - DeepSeek API（OpenAI 兼容协议）
  - 多轮对话上下文（InMemoryChatMessageHistory）
  - 工具调用失败时的友好兜底话术
"""
import os
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

from app.core.logger import logger
from tools.agent_tools import AGENT_TOOLS
from agent.memory_manager import get_user_facts, get_user_facts_raw
from rag.retriever import set_current_profile

# ── DeepSeek API 配置 ─────────────────────────────────────
# 通过环境变量注入，支持 .env 文件
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── System Prompt ─────────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业、友好、耐心的智能客服助手，名字叫「小智」。

## 你的职责
- 帮助用户查询订单状态、物流信息、投诉/工单记录
- 回答用户关于退换货政策、发票规则、物流政策、会员制度等通用业务问题
- 用自然、温暖的语气与用户沟通
- 从用户的自然语言中理解他们的真实需求

## 你拥有的工具
1. **search_knowledge_base** ⭐ 知识库检索 — 查询退换货政策、发票规则、会员制度、物流条款、售后流程等
2. **query_order_status** — 根据订单号查询订单状态
3. **query_logistics** — 根据快递单号查询物流信息
4. **query_complaint** — 根据用户ID查询投诉/工单记录

## 工具选择规则（重要！）
- 用户问「退货规则」「邮费谁出」「发票怎么开」「会员等级」「配送时效」等**政策/规则类问题**：
  → 必须优先调用 **search_knowledge_base**，不要自己编造规则
- 用户问「我的订单 ORD-xxx 到哪了」「查一下快递 SFxxx」「我的投诉」等**具体数据类问题**：
  → 调用对应的 query_* 工具
- 用户只是打招呼、感谢、道别 → 直接友好回复

## 工作流程
1. 仔细阅读用户的消息，判断他们想要什么
2. 如果是政策/规则类问题，优先检索知识库；如果是具体数据问题，调用对应工具
3. 获取工具返回的结果后，用自然语言总结给用户
4. 如果你不确定用户意图，可以先问候并引导用户说明需求

## 回复规范
- 用中文回复，语气亲切但不啰嗦
- 引用知识库规则时，明确指出依据（如「根据我们的退换货政策…」）
- 涉及数据时，把关键信息（订单号、状态、时间等）清晰地列出来
- 如果查询结果为空（如无投诉记录），如实告知并安抚用户
- 如果用户没有提供必要信息（如没有订单号），主动询问
- 回复末尾可以加上「还有其他需要帮您的吗？」之类的话

## 当前对话中的用户信息
用户的 user_id 会在每次对话中提供给你。当需要查询投诉记录时，请使用该 user_id。
"""

# ── 会话存储（生产环境应替换为 Redis / DB） ──────────────

# conversation_id → InMemoryChatMessageHistory
_session_store: Dict[str, InMemoryChatMessageHistory] = {}


def _get_or_create_session(conversation_id: str) -> InMemoryChatMessageHistory:
    """获取或创建会话历史"""
    if conversation_id not in _session_store:
        _session_store[conversation_id] = InMemoryChatMessageHistory()
        logger.info(f"[Session] 新建会话 | conversation_id={conversation_id}")
    return _session_store[conversation_id]


# ── Agent 构建（单例） ────────────────────────────────────

_agent_executor: Optional[AgentExecutor] = None


def _build_agent() -> AgentExecutor:
    """
    构建 Tool-Calling Agent。

    使用 create_tool_calling_agent（原生 Function Calling），
    而非旧版 ReAct 模板 —— 更稳定，参数提取更准确。
    """
    # 初始化 DeepSeek LLM（OpenAI 兼容协议）
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,          # 客服场景，偏低以保持稳定
        max_tokens=2048,
        timeout=30,
        max_retries=1,            # 快速失败，由外层兜底
    )

    # 构建 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 创建 Agent
    agent = create_tool_calling_agent(llm, AGENT_TOOLS, prompt)

    # 创建 Executor
    executor = AgentExecutor(
        agent=agent,
        tools=AGENT_TOOLS,
        verbose=False,                     # 生产环境关闭 verbose
        handle_parsing_errors=True,        # LLM 输出格式错误时自动重试
        max_iterations=5,                  # 最多 5 轮工具调用
        return_intermediate_steps=False,   # 只返回最终结果
    )

    logger.info(f"[Agent] Agent 初始化完成 | model={DEEPSEEK_MODEL} | tools={len(AGENT_TOOLS)}")
    return executor


def get_agent() -> AgentExecutor:
    """获取 Agent 单例"""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = _build_agent()
    return _agent_executor


# ── 对外接口 ──────────────────────────────────────────────

# 友好的兜底话术池
FALLBACK_REPLIES = [
    "非常抱歉，我暂时遇到了一些技术问题，无法处理您的请求。请您稍等片刻再试，或者联系人工客服获取帮助。给您带来不便非常抱歉！🙏",
    "哎呀，系统好像开了个小差～您别着急，稍后重试一下，或者直接联系人工客服，我们会优先为您处理。",
    "很抱歉，当前服务繁忙，我没能成功处理您的请求。建议您稍后再试，或转接人工客服为您服务。感谢您的耐心！",
]


async def run_agent(
    message: str,
    user_id: str,
    conversation_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    执行 Agent 对话。

    Args:
        message:         用户输入的自然语言消息
        user_id:         用户唯一标识（用于工具调用和上下文）
        conversation_id: 会话 ID，None 时自动创建

    Returns:
        (reply_text, conversation_id)

        成功时 reply_text 为 Agent 的自然语言回复；
        失败时 reply_text 为友好的兜底话术。

    ChromaDB 集成：
      在执行 Agent 前，从 SQLite 读取用户画像原始 dict 并设置到
      rag.retriever 的请求级上下文，供 search_faq() 的
      metadata 过滤使用。
    """
    import random
    import uuid

    # 确保有会话 ID
    if not conversation_id:
        conversation_id = str(uuid.uuid4())[:12]

    session = _get_or_create_session(conversation_id)

    logger.info(
        f"[Agent] 收到消息 | user_id={user_id} | "
        f"conversation_id={conversation_id} | message={message[:120]}"
    )

    # ── 构建输入 ──
    # 查询用户长期记忆 / 画像，动态注入上下文
    user_facts = get_user_facts(user_id)

    enriched_message = f"（当前用户 user_id: {user_id}）\n"

    if user_facts:
        enriched_message += (
            "【当前用户画像与长期记忆】请参考以下用户偏好提供个性化服务：\n"
            f"{user_facts}\n\n"
        )
        logger.info(f"[Agent] 已注入用户画像 | user_id={user_id}")

    enriched_message += f"用户消息: {message}"

    agent_input = {
        "input": enriched_message,
        "chat_history": session.messages,
    }

    # ── 设置 ChromaDB 元数据过滤上下文 ──
    # 从 SQLite 读取画像的原始 dict，用于构建 ChromaDB where 过滤条件
    profile_dict = get_user_facts_raw(user_id)
    set_current_profile(profile_dict)
    if profile_dict:
        logger.info(
            f"[Agent] 已设置 ChromaDB 过滤上下文 | profile={profile_dict}"
        )

    try:
        executor = get_agent()
        # 使用 ainvoke（异步），避免阻塞事件循环
        result = await executor.ainvoke(agent_input)

        reply = result.get("output", "").strip()

        if not reply:
            logger.warning("[Agent] Agent 返回空回复，使用兜底")
            reply = random.choice(FALLBACK_REPLIES)

        # ── 将本轮对话写入会话历史 ──
        session.add_message(HumanMessage(content=enriched_message))
        session.add_message(AIMessage(content=reply))

        logger.info(
            f"[Agent] 回复完成 | conversation_id={conversation_id} | "
            f"reply_length={len(reply)}"
        )

        return reply, conversation_id

    except Exception as exc:
        # ── 统一异常兜底 ──
        logger.error(f"[Agent] Agent 执行异常 | error={exc}")

        # 若为 API Key 未配置等明显配置问题，日志中提醒
        if "api_key" in str(exc).lower() or "authentication" in str(exc).lower():
            logger.error(
                "[Agent] DeepSeek API Key 无效或未配置！"
                "请设置环境变量 DEEPSEEK_API_KEY"
            )

        fallback = random.choice(FALLBACK_REPLIES)
        # 兜底回复也写入历史，保持上下文连贯
        session.add_message(HumanMessage(content=enriched_message))
        session.add_message(AIMessage(content=fallback))

        return fallback, conversation_id

    finally:
        # 清理 ChromaDB 过滤上下文
        set_current_profile(None)


def clear_session(conversation_id: str) -> bool:
    """清除指定会话"""
    if conversation_id in _session_store:
        del _session_store[conversation_id]
        logger.info(f"[Session] 清除会话 | conversation_id={conversation_id}")
        return True
    return False
