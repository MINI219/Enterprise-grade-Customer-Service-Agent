"""
MCP (Model Context Protocol) Server

将订单查询、物流追踪、投诉查询、RAG 知识库检索注册为标准 MCP Tools，
通过 SSE 传输层暴露给 MCP 客户端（如 Claude Desktop、Codex 等）。

架构：
  MCP Client ←── SSE ──→ FastAPI (/mcp/sse, /mcp/messages)
                              │
                         mcp_server.py
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              订单/物流   投诉查询   RAG 检索
"""
import json
import logging
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from app.core.logger import logger
from services.mock_services import (
    get_order_status,
    get_logistics_info,
    get_complaint_record,
)
from rag.retriever import search_faq

# ── 初始化 FastMCP 服务器 ─────────────────────────────────

mcp = FastMCP(
    name="智能客服 Agent MCP Server",
    instructions="提供订单查询、物流追踪、投诉查询、知识库检索等智能客服能力。",
)


# ═══════════════════════════════════════════════════════════
# MCP Tools 注册
# ═══════════════════════════════════════════════════════════

@mcp.tool()
def query_order_status(order_id: str) -> str:
    """
    查询用户的订单状态和进度。

    适用场景：用户询问「我的订单到哪了」「查一下订单」「订单状态」等。
    参数 order_id 为订单编号，格式通常为 'ORD-' 开头后跟日期和序号，
    例如 'ORD-20240623-001'。

    返回该订单的当前状态、商品信息、物流进度等详细 JSON 数据。
    """
    logger.info(f"[MCP] Tool: query_order_status | order_id={order_id}")
    try:
        result = get_order_status(order_id)
        logger.info(f"[MCP] query_order_status 成功 | status={result.get('status')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"[MCP] query_order_status 失败 | error={exc}")
        return json.dumps({"error": f"查询订单失败: {exc}"}, ensure_ascii=False)


@mcp.tool()
def query_logistics(tracking_number: str) -> str:
    """
    查询快递 / 物流的实时运输信息。

    适用场景：用户询问「快递到哪了」「查一下物流」「包裹到哪里了」等。
    参数 tracking_number 为快递单号 / 运单号，通常为字母+数字的组合，
    例如 'SF1234567890' 或 'YT9876543210'。

    返回物流公司、当前位置、完整轨迹节点等详细 JSON 数据。
    """
    logger.info(f"[MCP] Tool: query_logistics | tracking_number={tracking_number}")
    try:
        result = get_logistics_info(tracking_number)
        logger.info(f"[MCP] query_logistics 成功 | status={result.get('current_status')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"[MCP] query_logistics 失败 | error={exc}")
        return json.dumps({"error": f"查询物流失败: {exc}"}, ensure_ascii=False)


@mcp.tool()
def query_complaint(user_id: str) -> str:
    """
    查询用户的投诉 / 工单记录。

    适用场景：用户询问「我的投诉」「投诉进度」「工单处理得怎么样了」等。
    参数 user_id 为用户唯一标识，格式通常为 'U' 开头后跟数字，例如 'U10086'。

    返回该用户所有投诉工单的列表，包括编号、主题、状态、处理时间线等。
    """
    logger.info(f"[MCP] Tool: query_complaint | user_id={user_id}")
    try:
        result = get_complaint_record(user_id)
        logger.info(f"[MCP] query_complaint 成功 | total={result.get('total')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"[MCP] query_complaint 失败 | error={exc}")
        return json.dumps({"error": f"查询投诉失败: {exc}"}, ensure_ascii=False)


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """
    检索公司内部知识库，查询退换货政策、发票规则、会员制度、物流条款等通用业务规定。

    IMPORTANT: 当用户询问以下类型问题时，请优先调用此工具：
    - 退换货规则：「能不能退」「退货流程」「换货条件」「7天无理由退货」
    - 邮费政策：「退货邮费谁出」「换货运费」「运费报销上限」
    - 发票规则：「发票怎么开」「开票流程」「电子发票」「补开发票时效」
    - 物流政策：「配送时效」「配送范围」「偏远地区」「包裹破损处理」
    - 会员制度：「会员等级」「银卡金卡钻石卡」「优惠券规则」
    - 售后政策：「投诉流程」「客服联系方式」「客服工作时间」

    参数 query 为用户问题的完整语义表达，应尽量保留原意、避免拆词。

    返回知识库中最相关的文本段落及相关度评分。
    """
    logger.info(f"[MCP] Tool: search_knowledge_base | query={query[:120]}")
    try:
        result = search_faq(query)
        logger.info("[MCP] search_knowledge_base 完成")
        return result
    except Exception as exc:
        logger.error(f"[MCP] search_knowledge_base 失败 | error={exc}")
        return f"知识库检索失败: {exc}"


# ═══════════════════════════════════════════════════════════
# 工具列表（供外部引用）
# ═══════════════════════════════════════════════════════════

MCP_TOOLS = [
    query_order_status,
    query_logistics,
    query_complaint,
    search_knowledge_base,
]
