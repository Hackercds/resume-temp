"""
API 接口层 - 面试点：Handler 层只做参数校验和路由，不写业务逻辑
遵循 MICRO_SERVICE_SPEC.md 统一响应格式：{code, message, data, requestId, timestamp}

路由一览：
- POST /api/query          → RAG 问答
- POST /api/knowledge/upload → 文档上传入库
- GET  /api/knowledge/documents → 文档列表
- DELETE /api/knowledge/documents/{file_name} → 删除文档
- GET  /api/stats          → 知识库统计
- GET  /health             → 健康检查
"""
import time
import uuid
from urllib.parse import unquote

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from typing import Optional

from internal.model.dto import (
    APIResponse, QueryRequest, QueryResult, SourceItem, TimingInfo,
    UploadResult, DeleteResult, StatsResult, HealthResult, DocumentInfo
)
from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.pkg.errors import (
    RAGException, UnsupportedFileTypeError, FileTooLargeError,
    EmptyRetrievalError, DocumentNotFoundError
)
from internal.service.rag_service import RAGService
from internal.service.embedding_service import get_embedding_service
from internal.service.es_service import ESService
from internal.service.knowledge_base_service import KnowledgeBaseService
from internal.service.rerank_service import RerankService


router = APIRouter()
logger = get_logger()


# ---------- 中间件：为每个请求生成 traceId ----------
def _generate_request_id(request: Request) -> str:
    """生成请求ID并绑定到request.state"""
    rid = str(uuid.uuid4())[:8]
    request.state.request_id = rid
    return rid


# ---------- 服务单例（延迟初始化）----------
_rag_service = None
_kb_service = None
_es_service = None


def _get_es_service() -> ESService:
    global _es_service
    if _es_service is None:
        _es_service = ESService()
    return _es_service


def _get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        cfg = get_config()
        rerank = None
        if cfg.rerank.enabled:
            rerank = RerankService()
        _rag_service = RAGService(
            embedding_service=get_embedding_service(),
            es_service=_get_es_service(),
            rerank_service=rerank
        )
    return _rag_service


def _get_kb_service() -> KnowledgeBaseService:
    global _kb_service
    if _kb_service is None:
        _kb_service = KnowledgeBaseService(
            embedding_service=get_embedding_service(),
            es_service=_get_es_service()
        )
    return _kb_service


# ---------- 健康检查 ----------
@router.get("/health")
async def health_check():
    """
    健康检查接口 - 面试点：为什么需要健康检查？
    - K8s/Docker 的 liveness probe 需要
    - 检查核心依赖：embedding 模型 + ES 连接
    """
    rid = str(uuid.uuid4())[:8]
    es_connected = False
    embedding_loaded = False

    try:
        es_service = _get_es_service()
        es_connected = es_service.is_connected()
    except Exception:
        pass

    try:
        emb_service = get_embedding_service()
        embedding_loaded = emb_service.is_loaded
    except Exception:
        pass

    status = "ok" if (es_connected or embedding_loaded) else "degraded"

    result = HealthResult(
        status=status,
        embedding_loaded=embedding_loaded,
        es_connected=es_connected
    )

    return APIResponse.success(
        data=result.model_dump(),
        request_id=rid
    )


# ---------- POST /api/query ----------
@router.post("/api/query")
async def rag_query(request: Request, body: QueryRequest):
    """
    RAG 问答 - 面试点：全链路 traceId 贯穿
    统一响应格式 + 错误码规范
    """
    rid = _generate_request_id(request)
    logger.info(rid, "api_handler", "收到查询请求",
                question=body.question[:100],
                provider=body.provider, top_k=body.top_k)

    if not body.question.strip():
        return APIResponse.error(40001, "问题不能为空", rid).model_dump()

    try:
        rag_service = _get_rag_service()
        result = rag_service.query(
            question=body.question,
            api_key=body.api_key,
            provider=body.provider,
            model=body.model,
            top_k=body.top_k
        )

        # 构造 sources
        sources = [
            SourceItem(**s) for s in result.get("sources", [])
        ]
        timing = None
        if result.get("timing"):
            timing = TimingInfo(**result["timing"])

        query_result = QueryResult(
            answer=result["answer"],
            sources=sources,
            trace_id=result.get("trace_id", rid),
            timing=timing
        )

        logger.info(rid, "api_handler", "查询返回成功",
                    sources=len(sources))
        return APIResponse.success(
            data=query_result.model_dump(),
            request_id=rid
        ).model_dump()

    except RAGException as e:
        return APIResponse.error(e.code, e.message, rid).model_dump()
    except Exception as e:
        logger.error(rid, "api_handler", "查询异常",
                     error=str(e))
        return APIResponse.error(50004, f"查询失败: {str(e)}", rid).model_dump()


# ---------- POST /api/knowledge/upload ----------
@router.post("/api/knowledge/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = Form(default=400),
    overlap: int = Form(default=50),
):
    """
    文档上传入库 - 面试点：完整入库流程
    parse → chunk → embed → ES bulk insert
    """
    rid = _generate_request_id(request)
    logger.info(rid, "api_handler", "收到上传请求",
                file_name=file.filename, chunk_size=chunk_size)

    # 安全校验
    if not file.filename:
        return APIResponse.error(40001, "文件名不能为空", rid).model_dump()

    # 校验文件格式
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    allowed = {".pdf", ".txt", ".csv"}
    if ext not in allowed:
        return APIResponse.error(40002,
                                 f"不支持的文件格式: {ext}，仅支持 {allowed}",
                                 rid).model_dump()

    # 读取文件内容
    content = await file.read()

    # 文件大小校验（10MB）
    if len(content) > 10 * 1024 * 1024:
        return APIResponse.error(40003, "文件大小超过 10MB 限制", rid).model_dump()

    try:
        kb_service = _get_kb_service()
        result = kb_service.ingest_content(
            content=content,
            file_name=file.filename,
            chunk_size=chunk_size,
            overlap=overlap
        )

        upload_result = UploadResult(**result)
        logger.info(rid, "api_handler", "上传成功",
                    file_name=file.filename,
                    chunks_created=result["chunks_created"])
        return APIResponse.success(
            data=upload_result.model_dump(),
            request_id=rid
        ).model_dump()

    except ValueError as e:
        return APIResponse.error(40001, str(e), rid).model_dump()
    except RAGException as e:
        return APIResponse.error(e.code, e.message, rid).model_dump()
    except Exception as e:
        logger.error(rid, "api_handler", "上传异常",
                     file_name=file.filename, error=str(e))
        return APIResponse.error(50004, f"上传失败: {str(e)}", rid).model_dump()


# ---------- GET /api/knowledge/documents ----------
@router.get("/api/knowledge/documents")
async def list_documents(request: Request):
    """
    列出知识库所有文档 - 面试点：ES 聚合查询，不拉全量数据
    """
    rid = _generate_request_id(request)

    try:
        kb_service = _get_kb_service()
        documents = kb_service.list_documents()
        doc_infos = [DocumentInfo(**d) for d in documents]

        logger.info(rid, "api_handler", "文档列表查询",
                    count=len(doc_infos))
        return APIResponse.success(
            data={"documents": [d.model_dump() for d in doc_infos]},
            request_id=rid
        ).model_dump()

    except Exception as e:
        logger.error(rid, "api_handler", "文档列表查询失败",
                     error=str(e))
        return APIResponse.error(50004, f"查询失败: {str(e)}", rid).model_dump()


# ---------- DELETE /api/knowledge/documents/{file_name} ----------
@router.delete("/api/knowledge/documents/{file_name:path}")
async def delete_document(request: Request, file_name: str):
    """
    删除文档 - 面试点：按 file_name 删除所有 chunk
    URL 编码的文件名需要先解码
    """
    rid = _generate_request_id(request)
    file_name = unquote(file_name)
    logger.info(rid, "api_handler", "收到删除请求",
                file_name=file_name)

    try:
        kb_service = _get_kb_service()
        result = kb_service.delete_document(file_name)

        delete_result = DeleteResult(**result)
        logger.info(rid, "api_handler", "删除成功",
                    file_name=file_name,
                    deleted_chunks=result["deleted_chunks"])
        return APIResponse.success(
            data=delete_result.model_dump(),
            request_id=rid
        ).model_dump()

    except Exception as e:
        logger.error(rid, "api_handler", "删除失败",
                     file_name=file_name, error=str(e))
        return APIResponse.error(50004, f"删除失败: {str(e)}", rid).model_dump()


# ---------- GET /api/stats ----------
@router.get("/api/stats")
async def get_stats(request: Request):
    """
    知识库统计 - 面试点：ES count + aggregation
    """
    rid = _generate_request_id(request)

    try:
        kb_service = _get_kb_service()
        stats = kb_service.get_stats()
        stats_result = StatsResult(**stats)

        logger.info(rid, "api_handler", "统计查询",
                    total_chunks=stats["total_chunks"],
                    total_documents=stats["total_documents"])
        return APIResponse.success(
            data=stats_result.model_dump(),
            request_id=rid
        ).model_dump()

    except Exception as e:
        logger.error(rid, "api_handler", "统计查询失败",
                     error=str(e))
        return APIResponse.error(50004, f"统计查询失败: {str(e)}", rid).model_dump()


# ---------- GET / ----------
@router.get("/")
async def root():
    """根路径重定向到 API 文档"""
    return {
        "service": "resume-rag-service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
