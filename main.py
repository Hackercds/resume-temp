"""
简历 RAG 智能问答系统 - FastAPI 入口
本地 embedding + Elasticsearch 混合检索 + 在线 LLM

启动方式：
  python main.py              → 后端 + 前端一起（dev 模式）
  APP_MODE=dev python main.py → 热重载 + 前端调试
"""
import sys
import os
import uuid
import time
import signal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# 确保项目根在 path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.pkg.errors import RAGException
from internal.handler.api_handler import router


# ---------- 应用生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    logger = get_logger()
    trace_id = "startup"
    logger.info(trace_id, "main", f"服务启动... {cfg.app.name} v1.0.0",
                host=cfg.app.host, port=cfg.app.port, mode=cfg.app.mode)

    # 后台预热 embedding 模型
    try:
        from internal.service.embedding_service import get_embedding_service
        import threading, os
        emb_service = get_embedding_service()
        logger.info(trace_id, "main",
                    "Embedding 模型预热中（后台线程）",
                    model=cfg.embedding.model_name,
                    cache=cfg.embedding.cache_folder,
                    log_level=cfg.log.level)
        threading.Thread(target=lambda: emb_service.load_model(), daemon=True).start()
    except Exception as e:
        logger.warn(trace_id, "main", "Embedding 模型预热失败，首次请求时加载",
                    error=str(e))

    yield
    logger.info(trace_id, "main", "服务关闭")


# ---------- FastAPI 应用 ----------
cfg = get_config()
app = FastAPI(
    title=cfg.app.name,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — dev 模式放宽，production 严格
if cfg.app.mode == "dev":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# API 路由
app.include_router(router)

# 托管前端静态文件（前面板直接开箱可用）
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.is_dir():
    # 静态资源
    src_dir = frontend_dir / "src"
    if src_dir.is_dir():
        app.mount("/src", StaticFiles(directory=str(src_dir)), name="frontend_src")

    # 根路径 → 前端口
    @app.get("/app", include_in_schema=False)
    @app.get("/app/", include_in_schema=False)
    async def frontend_spa():
        return FileResponse(str(frontend_dir / "index.html"))

    # 已经是 index.html 内部资源
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": cfg.app.name,
            "version": "1.0.0",
            "frontend": "/app",
            "docs": "/docs",
            "health": "/health"
        }


# ---------- 全局异常处理 ----------
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    logger = get_logger()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    logger.error(request_id, "exception_handler", exc.message,
                 code=exc.code, path=request.url.path)
    return JSONResponse(
        status_code=200,
        content={
            "code": exc.code,
            "message": exc.message,
            "data": None,
            "requestId": request_id,
            "timestamp": int(time.time() * 1000)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger = get_logger()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    logger.error(request_id, "exception_handler", str(exc),
                 type=type(exc).__name__, path=request.url.path)
    return JSONResponse(
        status_code=200,
        content={
            "code": 50004,
            "message": "内部服务错误",
            "data": None,
            "requestId": request_id,
            "timestamp": int(time.time() * 1000)
        }
    )


# ---------- 优雅退出 ----------
def setup_graceful_shutdown():
    def handler(signum, frame):
        logger = get_logger()
        logger.info("shutdown", "main", "收到退出信号，优雅关闭...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


if __name__ == "__main__":
    import uvicorn

    setup_graceful_shutdown()

    print(f"""
╔══════════════════════════════════════════╗
║   简历 RAG 智能问答系统 v1.0.0       ║
╠══════════════════════════════════════════╣
║  API:     http://{cfg.app.host}:{cfg.app.port}         ║
║  Swagger: http://{cfg.app.host}:{cfg.app.port}/docs    ║
║  前面板:  http://{cfg.app.host}:{cfg.app.port}/app     ║
║  Health:  http://{cfg.app.host}:{cfg.app.port}/health  ║
╚══════════════════════════════════════════╝
""")

    uvicorn.run(
        "main:app",
        host=cfg.app.host,
        port=cfg.app.port,
        reload=cfg.app.mode == "dev",
        log_level=cfg.log.level.lower(),
    )
