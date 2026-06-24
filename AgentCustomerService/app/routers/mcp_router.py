"""
MCP SSE 路由

将 MCP Server 通过 SSE (Server-Sent Events) 传输层暴露：
  - GET  /mcp/sse       — 建立 SSE 长连接
  - POST /mcp/messages  — 接收客户端 JSON-RPC 消息

MCP 客户端（如 Claude Desktop）通过这两个端点发现和调用工具。
"""
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app.core.logger import logger
from mcp_server import mcp  # FastMCP 实例

router = APIRouter(prefix="/mcp", tags=["mcp"])

# ── SSE Transport 初始化 ─────────────────────────────────

from mcp.server.sse import SseServerTransport

# "/mcp/messages" 是客户端 POST 消息的目标路径
_sse_transport = SseServerTransport("/mcp/messages")


@router.get("/sse")
async def mcp_sse_endpoint(request: Request):
    """
    建立 MCP SSE 长连接。

    客户端连接到此端点后，服务器会：
    1. 发送 endpoint 事件告知消息投递地址
    2. 通过该 SSE 通道流式返回 Tool 调用结果
    """
    request_id = str(uuid.uuid4())[:8]

    logger.info(f"[MCP-SSE] 新 SSE 连接 | request_id={request_id} | client={request.client}")

    async with _sse_transport.connect_sse(
        request.scope,
        request.receive,
        request._send,
    ) as (read_stream, write_stream):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )

    logger.info(f"[MCP-SSE] SSE 连接关闭 | request_id={request_id}")
    # SSE 是流式响应，返回空 Response 避免 FastAPI 报错
    return Response()


@router.post("/messages")
async def mcp_messages_endpoint(request: Request):
    """
    接收 MCP 客户端的 JSON-RPC 消息。

    客户端通过此端点发送 Tool 调用请求（如 tool/list, tool/call），
    服务器处理后通过 SSE 通道返回结果。
    """
    logger.debug(f"[MCP-MSG] 收到消息 | client={request.client}")

    await _sse_transport.handle_post_message(
        request.scope,
        request.receive,
        request._send,
    )

    return Response()
