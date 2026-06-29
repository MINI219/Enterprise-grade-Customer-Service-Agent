"""
长期记忆管理器（Long-Term Memory & User Profiling）

架构：
  对话历史 → LLM 抽取偏好 → JSON 格式 facts → SQLite 持久化
                                                │
  下次对话 ← 动态注入 System Prompt ← 读取 ──────┘

使用 Python 内置 sqlite3，零外部依赖。
"""
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.logger import logger

# ── 路径与配置 ────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "user_memory.db"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── Prompt: 从对话中提取用户画像 ──────────────────────────

EXTRACTION_PROMPT = """你是一个用户画像分析专家。请阅读以下客服对话记录，提取用户的**个人偏好、业务习惯、固定属性**。

## 提取规则
1. 只提取明确、可长期复用的信息，忽略临时的、一次性的内容
2. 关注以下维度：
   - **物流偏好**：偏好的快递公司、收货时间偏好
   - **消费习惯**：常购品类、价格敏感度、是否偏好促销
   - **沟通风格**：喜欢简洁还是详细、是否急躁易投诉
   - **个人信息**：称呼偏好、所在城市、会员等级
   - **投诉倾向**：容易因为什么问题投诉、投诉频率
   - **其他特征**：任何值得记录的用户固定属性
3. 如果对话中没有可提取的长期特征，返回空的 facts 对象 {}
4. 只返回 JSON，不要包含任何解释文字

## 已有画像（避免重复）
{existing_facts}

## 对话记录
{chat_history}

## 输出格式（严格 JSON）
{{
  "facts": {{
    "物流偏好": "顺丰快递",
    "沟通风格": "喜欢详细解释",
    "所在城市": "深圳",
    "...": "..."
  }}
}}

请输出 JSON:"""


# ── 数据库初始化 ──────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """获取数据库连接（自动建表）"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id   TEXT PRIMARY KEY,
            facts     TEXT DEFAULT '{}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


# ── 读取记忆 ──────────────────────────────────────────────

def get_user_facts(user_id: str) -> Optional[str]:
    """
    查询用户的长期记忆 / 画像。

    Args:
        user_id: 用户 ID

    Returns:
        格式化的自然语言画像描述，无记忆时返回 None
    """
    facts_dict = get_user_facts_raw(user_id)
    if not facts_dict:
        return None

    lines = []
    for key, value in facts_dict.items():
        lines.append(f"- {key}: {value}")

    logger.info(f"[Memory] 读取用户画像 | user_id={user_id} | fields={len(lines)}")
    return "\n".join(lines)


def get_user_facts_raw(user_id: str) -> Optional[Dict[str, Any]]:
    """
    查询用户的长期记忆 / 画像（原始 dict 格式）。

    供 ChromaDB metadata 过滤使用 —— 需要结构化的 key-value。

    Args:
        user_id: 用户 ID

    Returns:
        画像 dict，如 {"card_type": "times_card", "level": "gold"}
        无记忆时返回 None
    """
    try:
        conn = _get_connection()
        row = conn.execute(
            "SELECT facts FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        conn.close()

        if not row:
            return None

        facts_raw = row[0]
        if not facts_raw or facts_raw == "{}":
            return None

        facts_dict = json.loads(facts_raw)
        if not facts_dict:
            return None

        return facts_dict

    except Exception as exc:
        logger.error(f"[Memory] 读取记忆失败 | user_id={user_id} | error={exc}")
        return None


# ── 写入 / 更新记忆 ───────────────────────────────────────

def _merge_facts(user_id: str, new_facts: dict) -> dict:
    """
    将新提取的特征合并到已有画像中。

    合并策略：
    - 新 key 直接追加
    - 已有 key 用新值覆盖（最新信息优先）
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT facts FROM user_profiles WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    existing = {}
    if row and row[0]:
        try:
            existing = json.loads(row[0])
        except json.JSONDecodeError:
            existing = {}

    # 合并：新值覆盖旧值
    merged = {**existing, **new_facts}

    conn.execute(
        """INSERT INTO user_profiles (user_id, facts, updated_at)
           VALUES (?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(user_id) DO UPDATE SET
               facts = ?,
               updated_at = CURRENT_TIMESTAMP""",
        (user_id, json.dumps(merged, ensure_ascii=False),
         json.dumps(merged, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()

    logger.info(
        f"[Memory] 画像更新 | user_id={user_id} | "
        f"existing={len(existing)} fields | new={len(new_facts)} fields | "
        f"merged={len(merged)} fields"
    )
    return merged


# ── 核心：LLM 抽取 + 写入 ─────────────────────────────────

async def extract_and_update_memory(user_id: str, chat_history: str) -> bool:
    """
    异步分析对话记录，提取用户偏好并持久化。

    此函数设计为在 FastAPI BackgroundTasks 中执行，
    不会阻塞主线程的 HTTP 响应。

    Args:
        user_id:      用户 ID
        chat_history: 完整的对话记录文本

    Returns:
        True 表示成功提取并写入，False 表示跳过（无需更新或无新信息）
    """
    if not chat_history or not chat_history.strip():
        logger.info(f"[Memory] 跳过抽取：对话历史为空 | user_id={user_id}")
        return False

    # 读取已有画像，传给 LLM 避免重复提取
    existing_raw = get_user_facts(user_id)
    existing_str = existing_raw if existing_raw else "（暂无已有画像）"

    prompt = EXTRACTION_PROMPT.format(
        existing_facts=existing_str,
        chat_history=chat_history,
    )

    logger.info(
        f"[Memory] 开始抽取用户画像 | user_id={user_id} | "
        f"history_length={len(chat_history)}"
    )

    try:
        llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.1,       # 抽取任务，温度极低以保证稳定
            max_tokens=1024,
            timeout=20,
            max_retries=1,
        )

        messages = [
            SystemMessage(content="你是一个用户画像分析专家。只返回 JSON，不要包含任何解释。"),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)

        raw_text = response.content.strip()

        # 清理可能包裹的 markdown 代码块
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        logger.debug(f"[Memory] LLM 返回 | raw={raw_text[:300]}")

        # 解析 JSON
        parsed = json.loads(raw_text)
        facts = parsed.get("facts", {})

        if not facts or not isinstance(facts, dict):
            logger.info(f"[Memory] 未提取到新特征 | user_id={user_id}")
            return False

        # 过滤掉空值
        facts = {k: v for k, v in facts.items() if v and str(v).strip()}

        if not facts:
            logger.info(f"[Memory] 过滤后无有效特征 | user_id={user_id}")
            return False

        # 合并写入
        _merge_facts(user_id, facts)
        logger.info(f"[Memory] 画像持久化完成 | user_id={user_id} | facts={facts}")
        return True

    except json.JSONDecodeError as exc:
        logger.warning(f"[Memory] LLM 返回非 JSON | user_id={user_id} | raw={raw_text[:200]}")
        return False
    except Exception as exc:
        logger.error(f"[Memory] 抽取失败 | user_id={user_id} | error={exc}")
        return False


# ── 会话历史导出 ──────────────────────────────────────────

def get_chat_history_text(user_id: str, conversation_id: str) -> str:
    """
    从 Agent 会话存储中导出对话记录文本。

    供 BackgroundTasks 中调用 extract_and_update_memory 时使用。
    """
    from agent.core import _get_or_create_session

    session = _get_or_create_session(conversation_id)
    messages = session.messages

    if not messages:
        return ""

    lines = []
    for msg in messages:
        role = "用户" if msg.type == "human" else "客服"
        lines.append(f"[{role}]: {msg.content}")

    return "\n".join(lines)
