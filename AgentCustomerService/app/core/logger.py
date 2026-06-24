"""
日志记录器配置
使用 loguru 记录每次接口调用的入参、出参以及业务状态
"""
import sys
from pathlib import Path

from loguru import logger


def _safe_format(record):
    """自定义格式化函数：安全获取 extra 字段，避免 KeyError"""
    rid = record["extra"].get("request_id", "-")
    # 将 request_id 注入 record 以便 format 中使用
    record["extra"]["request_id"] = rid
    return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[request_id]} | {message}\n"


# 移除默认 handler
logger.remove()

# 全局默认 extra —— 避免非 HTTP 上下文调用时 KeyError
logger.configure(extra={"request_id": "-"})

# 控制台输出 —— 彩色、结构化
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[request_id]}</cyan> | "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

# 确保 logs 目录存在
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 文件输出
logger.add(
    LOG_DIR / "api_{time:YYYY-MM-DD}.log",
    format=_safe_format,
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    encoding="utf-8",
)

# 错误日志单独记录
logger.add(
    LOG_DIR / "error_{time:YYYY-MM-DD}.log",
    format=_safe_format,
    level="ERROR",
    rotation="00:00",
    retention="90 days",
    encoding="utf-8",
)


def get_logger():
    """返回配置好的 logger 实例"""
    return logger


__all__ = ["logger", "get_logger"]
