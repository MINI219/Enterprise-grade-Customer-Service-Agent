"""
Pydantic 请求 / 响应模型

V2 升级说明：
  - ChatRequest  从「意图 + 参数」改为「自然语言消息 + 会话ID」
  - ChatResponse 从「意图 + 数据」改为「自然语言回复 + 会话ID」
  - 保留旧模型用于向后兼容和内部使用
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── V2 请求（Agent 驱动） ─────────────────────────────────

class ChatRequest(BaseModel):
    """
    对话接口请求体（Agent 模式）

    用户发送自然语言消息，由 Agent 自动理解意图并调用工具。
    """
    user_id: str = Field(
        ...,
        description="用户唯一标识，如 U10086",
        examples=["U10086"],
    )
    message: str = Field(
        ...,
        description="用户输入的自然语言消息，无需指定意图或参数",
        examples=["帮我查一下订单 ORD-20240623-001 到哪了"],
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="会话 ID，用于多轮对话上下文追踪。首次对话不传，后续传入返回的 conversation_id",
        examples=["a1b2c3d4e5f6"],
    )


# ── V2 响应（Agent 驱动） ─────────────────────────────────

class ChatResponse(BaseModel):
    """对话接口响应体（Agent 模式）"""
    code: int = Field(default=200, description="业务状态码，200=成功，500=异常兜底")
    message: str = Field(default="success", description="状态描述")
    reply: str = Field(
        ...,
        description="Agent 的自然语言回复，已包含完整信息（无需前端解析结构化数据）",
    )
    conversation_id: str = Field(
        ...,
        description="会话 ID，后续对话需传入此值以保持上下文连续",
    )


# ── 系统 ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="ok")
    version: str = Field(default="0.4.0")


# ── Mock 业务数据模型（方便后续扩展） ──────────────────────

class OrderStatus(BaseModel):
    order_id: str
    status: str
    product_name: str
    price: float
    created_at: str
    estimated_delivery: str
    progress: List[Dict[str, str]]


class LogisticsInfo(BaseModel):
    tracking_number: str
    carrier: str
    current_status: str
    origin: str
    destination: str
    checkpoints: List[Dict[str, str]]


class ComplaintRecord(BaseModel):
    complaint_id: str
    user_id: str
    subject: str
    status: str
    filed_at: str
    resolution: Optional[str] = None
    timeline: List[Dict[str, str]]
