"""
FastAPI 应用入口
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logger import logger
from app.routers.chat import router as chat_router
from app.routers.chat_ui import router as ui_router
from app.routers.mcp_router import router as mcp_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("[App] ========== 智能客服 Agent 服务启动 ==========")
    yield
    logger.info("[App] ========== 智能客服 Agent 服务关闭 ==========")


app = FastAPI(
    title="智能客服 Agent",
    description="基于 FastAPI + LangChain + DeepSeek + RAG + MCP 的智能客服 Agent 后端服务",
    version="0.4.0",
    lifespan=lifespan,
)

# CORS —— 允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由（UI 路由放前面，确保 / 不被其他路由拦截）
app.include_router(ui_router)
app.include_router(chat_router)
app.include_router(mcp_router)


@app.get("/health", tags=["system"])
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": "0.4.0"}


@app.middleware("http")
async def log_every_request(request, call_next):
    """中间件：为每个 HTTP 请求注入 request_id"""
    request_id = str(uuid.uuid4())[:8]
    with logger.contextualize(request_id=request_id):
        logger.info(f"[Middleware] --> {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(
            f"[Middleware] <-- {request.method} {request.url.path} | status={response.status_code}"
        )
        return response
