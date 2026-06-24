"""
/api/chat 对话路由（Agent 驱动）

流程：
  用户消息 → Agent（LLM 理解意图 → 选择工具 → 执行 → 汇总回答）
           → 返回自然语言回复 + conversation_id
           → BackgroundTasks: 异步抽取用户画像写入长期记忆

支持多轮对话：客户端传入 conversation_id 即可在上下文中连续对话。
"""
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.logger import logger
from models.schemas import ChatRequest, ChatResponse
from agent.core import run_agent, clear_session
from agent.memory_manager import extract_and_update_memory, get_chat_history_text

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    智能客服对话接口（Agent 模式）

    用户直接发送自然语言消息，Agent 自动：
      1. 查询用户画像，注入个性化上下文
      2. 理解用户意图
      3. 选择合适的工具（查订单 / 查物流 / 查投诉 / 查知识库）
      4. 执行并获取结果
      5. 汇总为自然语言回复
      6. 后台异步：从对话中抽取用户偏好，写入长期记忆

    ### 长期记忆机制
    每次对话结束后，后台异步分析对话内容，提取用户的偏好和习惯
    （如偏好的快递公司、沟通风格、所在城市等），存储到 SQLite。
    下次同一 user_id 对话时，Agent 会自动加载这些画像提供个性化服务。

    ### 多轮对话示例

    **第一轮**（不传 conversation_id）：
    ```json
    {"user_id": "U10086", "message": "帮我查一下 ORD-20240623-001 订单"}
    ```
    返回 `conversation_id: "a1b2c3d4e5f6"`

    **第二轮**（传入 conversation_id）：
    ```json
    {"user_id": "U10086", "message": "那物流到哪了？", "conversation_id": "a1b2c3d4e5f6"}
    ```
    Agent 会结合上文理解"那"指的是刚才查的订单。
    """
    request_id = str(uuid.uuid4())[:8]

    with logger.contextualize(request_id=request_id):
        # ── 入参日志 ──
        logger.info(
            f"[Request] 收到对话请求 | user_id={request.user_id} | "
            f"conversation_id={request.conversation_id or '(new)'} | "
            f"message={request.message[:150]}"
        )

        # ── 基本参数校验 ──
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=422, detail="message 不能为空")

        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=422, detail="user_id 不能为空")

        # ── 调用 Agent ──
        reply, conv_id = run_agent(
            message=request.message.strip(),
            user_id=request.user_id.strip(),
            conversation_id=request.conversation_id,
        )

        # ── 判断是否为兜底回复 ──
        # （兜底回复以特定前缀开头，这种情况标记 code=500 提醒前端）
        is_fallback = reply.startswith("非常抱歉") or reply.startswith("哎呀") or reply.startswith("很抱歉")

        # ── 后台任务：异步抽取用户画像 ──
        # 只在非兜底、成功对话时触发记忆抽取
        if not is_fallback:
            uid = request.user_id.strip()
            cid = conv_id
            background_tasks.add_task(
                _background_extract_memory,
                user_id=uid,
                conversation_id=cid,
            )
            logger.info(f"[Memory] 后台记忆抽取已调度 | user_id={uid}")

        logger.info(
            f"[Response] 回复完成 | conversation_id={conv_id} | "
            f"fallback={is_fallback} | reply_preview={reply[:80]}"
        )

        return ChatResponse(
            code=500 if is_fallback else 200,
            message="fallback" if is_fallback else "success",
            reply=reply,
            conversation_id=conv_id,
        )


async def _background_extract_memory(user_id: str, conversation_id: str):
    """
    后台任务：从对话历史中抽取用户画像。

    此函数在 BackgroundTasks 中异步执行，
    不会阻塞 HTTP 响应返回给用户。
    """
    logger.info(f"[Memory-BG] 开始后台记忆抽取 | user_id={user_id}")
    try:
        chat_text = get_chat_history_text(user_id, conversation_id)
        if chat_text:
            success = await extract_and_update_memory(user_id, chat_text)
            logger.info(
                f"[Memory-BG] 记忆抽取{'成功' if success else '跳过'} | user_id={user_id}"
            )
        else:
            logger.info(f"[Memory-BG] 无对话历史可抽取 | user_id={user_id}")
    except Exception as exc:
        logger.error(f"[Memory-BG] 后台记忆抽取异常 | user_id={user_id} | error={exc}")



@router.delete("/chat/session/{conversation_id}", tags=["chat"])
async def delete_session(conversation_id: str):
    """
    清除指定会话的上下文（结束对话）
    """
    ok = clear_session(conversation_id)
    if ok:
        logger.info(f"[Session] 会话已清除 | conversation_id={conversation_id}")
        return {"code": 200, "message": "会话已清除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")
