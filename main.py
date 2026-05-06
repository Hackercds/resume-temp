"""
简历 RAG 智能问答系统 - FastAPI 入口
本地 embedding + Elasticsearch 混合检索 + 在线 LLM

面试核心项目：演示完整 RAG 链路
"""
import sys
import os
import uuid
import time
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 确保项目根在 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.pkg.errors import RAGException
from internal.handler.api_handler import router


# ---------- 应用生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """面试点：启动时预热模型，关闭时优雅退出"""
    cfg = get_config()
    logger = get_logger()
    trace_id = "startup"
    logger.info(trace_id, "main", f"服务启动中... {cfg.app.name} v1.0.0",
                host=cfg.app.host, port=cfg.app.port, mode=cfg.app.mode)

    # 预热 embedding 模型（后台线程，不阻塞启动）
    try:
        from internal.service.embedding_service import get_embedding_service
        emb_service = get_embedding_service()
        import threading
        threading.Thread(target=lambda: emb_service.load_model(), daemon=True).start()
        logger.info(trace_id, "main", "Embedding 模型预热已提交后台线程",
                    model=cfg.embedding.model_name)
    except Exception as e:
        logger.warn(trace_id, "main", "Embedding 模型预热失败，将在首次请求时加载",
                    error=str(e))

    yield

    logger.info(trace_id, "main", "服务正在关闭...")


# ---------- FastAPI 应用 ----------
cfg = get_config()
app = FastAPI(
    title=cfg.app.name,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置（面试点：禁止 *，指定可信域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ---------- 全局异常处理 ----------
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """面试点：统一异常处理，返回友好错误信息"""
    logger = get_logger()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    logger.error(request_id, "exception_handler", exc.message,
                 code=exc.code, path=request.url.path)
    return JSONResponse(
        status_code=200,  # 业务异常返回200，通过code区分
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
    """兜底异常处理 - 面试点：不暴露堆栈信息"""
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
    """面试点：捕获 SIGTERM，优雅关闭"""
    def handler(signum, frame):
        logger = get_logger()
        logger.info("shutdown", "main", "收到退出信号，正在优雅关闭...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


if __name__ == "__main__":
    import uvicorn

    setup_graceful_shutdown()

    uvicorn.run(
        "main:app",
        host=cfg.app.host,
        port=cfg.app.port,
        reload=cfg.app.mode == "dev",
        log_level=cfg.log.level.lower(),
    )
