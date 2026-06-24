"""
开发环境启动入口
用法: python run.py

启动前请确保已配置环境变量（复制 .env.example 为 .env 并填入 API Key）：
  DEEPSEEK_API_KEY=sk-xxx
"""
import os
from pathlib import Path

import uvicorn

# 自动加载项目根目录的 .env 文件（如果存在）
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    print(f"[run.py] 已加载环境变量: {_env_path}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
