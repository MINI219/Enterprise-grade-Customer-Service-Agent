"""
Agent 工具集
将 Mock 业务函数封装为 LangChain 标准 Tool，
每个参数携带详细 description，供 LLM Function Calling 精确识别。
"""
import json
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.logger import logger
from services.mock_services import (
    get_order_status,
    get_logistics_info,
    get_complaint_record,
)
from rag.retriever import search_faq


# ── Pydantic Schema —— 严格约束 LLM 传入的参数 ──────────────

class OrderStatusInput(BaseModel):
    """查询订单状态的参数"""
    order_id: str = Field(
        ...,
        description=(
            "订单编号，格式通常为 'ORD-' 开头后跟日期和序号，"
            "例如 'ORD-20240623-001'。从用户的输入中提取订单号。"
        ),
    )


class LogisticsInput(BaseModel):
    """查询物流信息的参数"""
    tracking_number: str = Field(
        ...,
        description=(
            "快递单号 / 运单号，通常为字母+数字的组合，"
            "例如 'SF1234567890' 或 'YT9876543210'。从用户的输入中提取快递单号。"
        ),
    )


class ComplaintInput(BaseModel):
    """查询投诉记录或工单的参数"""
    user_id: str = Field(
        ...,
        description=(
            "用户唯一标识，格式通常为 'U' 开头后跟数字，"
            "例如 'U10086'。当用户询问'我的投诉''我的工单''投诉进度'时，"
            "用当前对话关联的 user_id 作为参数。"
        ),
    )


class KnowledgeBaseInput(BaseModel):
    """知识库检索的参数"""
    query: str = Field(
        ...,
        description=(
            "要在知识库中检索的完整问题或关键词。"
            "应尽可能保留用户问题的完整语义，避免拆词。"
            "例如用户问「退货邮费谁出」→ query='退货邮费谁承担'；"
            "用户问「发票怎么开」→ query='发票开具流程和要求'。"
        ),
    )


# ── Tool 定义 ──────────────────────────────────────────────

@tool(args_schema=OrderStatusInput)
def query_order_status(order_id: str) -> str:
    """
    查询用户的订单状态和进度。

    适用场景：
    - 用户询问「我的订单到哪了」「查一下订单」「订单状态」「什么时候到货」
    - 用户提供了订单编号

    返回该订单的当前状态、商品信息、物流进度等详细数据。
    """
    logger.info(f"[Tool] query_order_status 被调用 | order_id={order_id}")
    try:
        result = get_order_status(order_id)
        logger.info(f"[Tool] query_order_status 成功 | status={result.get('status')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"[Tool] query_order_status 失败 | error={exc}")
        return json.dumps({"error": f"查询订单失败: {exc}"}, ensure_ascii=False)


@tool(args_schema=LogisticsInput)
def query_logistics(tracking_number: str) -> str:
    """
    查询快递 / 物流的实时运输信息。

    适用场景：
    - 用户询问「快递到哪了」「查一下物流」「包裹到哪里了」
    - 用户提供了快递单号或运单号

    返回物流公司、当前位置、完整轨迹节点等详细信息。
    """
    logger.info(f"[Tool] query_logistics 被调用 | tracking_number={tracking_number}")
    try:
        result = get_logistics_info(tracking_number)
        logger.info(f"[Tool] query_logistics 成功 | status={result.get('current_status')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"[Tool] query_logistics 失败 | error={exc}")
        return json.dumps({"error": f"查询物流失败: {exc}"}, ensure_ascii=False)


@tool(args_schema=ComplaintInput)
def query_complaint(user_id: str) -> str:
    """
    查询用户的投诉 / 工单记录。

    适用场景：
    - 用户询问「我的投诉」「投诉进度」「工单处理得怎么样了」
    - 用户未提供投诉编号，需要查该用户的所有投诉

    返回该用户所有投诉工单的列表，包括编号、主题、状态、处理时间线等。
    """
    logger.info(f"[Tool] query_complaint 被调用 | user_id={user_id}")
    try:
        result = get_complaint_record(user_id)
        logger.info(f"[Tool] query_complaint 成功 | total={result.get('total')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"[Tool] query_complaint 失败 | error={exc}")
        return json.dumps({"error": f"查询投诉失败: {exc}"}, ensure_ascii=False)


@tool(args_schema=KnowledgeBaseInput)
async def search_knowledge_base(query: str) -> str:
    """
    检索公司内部知识库，查询退换货政策、发票规则、会员制度、物流条款等通用业务规定。

    ⚠️ 重要：当用户询问以下类型问题时，必须优先调用此工具：
    - 退换货规则：「能不能退」「退货流程」「换货条件」「无理由退货」「7天退货」
    - 邮费政策：「退货邮费谁出」「换货运费」「运费报销」
    - 发票规则：「发票怎么开」「开票流程」「电子发票」「发票修改」「补开发票」
    - 物流政策：「配送时效」「配送范围」「偏远地区」「包裹破损处理」
    - 会员制度：「会员等级」「银卡金卡钻石卡」「优惠券规则」
    - 售后政策：「投诉流程」「客服联系方式」「客服工作时间」

    只有当用户的问题明确属于以上业务规则类问题时才调用此工具。
    如果用户问的是具体的订单、物流单号、个人投诉，请使用对应的 query_* 工具。

    💡 此工具会自动读取当前用户的画像数据（如会员等级、偏好等），
       优先检索与该用户相关的业务规则后再进行语义匹配。
    """
    logger.info(f"[Tool] search_knowledge_base 被调用 | query={query[:120]}")
    try:
        result = await search_faq(query)
        logger.info(f"[Tool] search_knowledge_base 完成")
        return result
    except Exception as exc:
        logger.error(f"[Tool] search_knowledge_base 失败 | error={exc}")
        return f"知识库检索失败: {exc}"


# ── 汇总导出 ──────────────────────────────────────────────

AGENT_TOOLS = [
    search_knowledge_base,   # RAG 工具放首位，优先暴露给 Agent
    query_order_status,
    query_logistics,
    query_complaint,
]
