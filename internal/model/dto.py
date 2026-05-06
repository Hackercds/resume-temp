"""
数据传输对象 - Request/Response Models
面试点：Pydantic 自动校验输入，防止非法数据进入后端
遵循 MICRO_SERVICE_SPEC.md 统一响应格式：{code, message, data, requestId, timestamp}
"""
import uuid
import time
from typing import Optional, List, Any
from pydantic import BaseModel, Field


# ---------- 统一响应包装 ----------
class APIResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: Any = None
    requestId: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

    @classmethod
    def success(cls, data: Any = None, request_id: str = None) -> "APIResponse":
        return cls(
            code=0,
            message="success",
            data=data,
            requestId=request_id or str(uuid.uuid4())[:8],
            timestamp=int(time.time() * 1000)
        )

    @classmethod
    def error(cls, code: int, message: str, request_id: str = None) -> "APIResponse":
        return cls(
            code=code,
            message=message,
            data=None,
            requestId=request_id or str(uuid.uuid4())[:8],
            timestamp=int(time.time() * 1000)
        )


# ---------- 查询请求 ----------
class QueryRequest(BaseModel):
    """面试点：API Key 透传，后端不存储"""
    question: str = Field(..., min_length=1, max_length=2000,
                          description="用户问题")
    api_key: str = Field(..., min_length=1,
                         description="LLM API Key，后端不存储，仅透传")
    provider: str = Field(default="openai", description="LLM Provider: openai / anthropic / custom")
    model: Optional[str] = Field(default=None, description="模型名，支持自由输入")
    base_url: Optional[str] = Field(default=None, description="自定义 API 地址，如 https://api.deepseek.com/v1")
    top_k: int = Field(default=5, ge=1, le=20, description="检索Top-K")


# ---------- 查询响应 ----------
class SourceItem(BaseModel):
    content: str
    file_name: str
    score: float = 0.0


class TimingInfo(BaseModel):
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    llm_s: float = 0.0


class QueryResult(BaseModel):
    answer: str
    sources: List[SourceItem] = []
    trace_id: str = ""
    timing: Optional[TimingInfo] = None


# ---------- 文档管理 ----------
class DocumentInfo(BaseModel):
    file_name: str
    file_type: str = "unknown"
    chunk_count: int = 0
    upload_time: Optional[str] = None


class UploadResult(BaseModel):
    file_name: str
    file_type: str
    chunks_created: int
    total_chunks: int


class DeleteResult(BaseModel):
    deleted_chunks: int
    remaining_chunks: int


class StatsResult(BaseModel):
    total_chunks: int
    total_documents: int
    documents: List[DocumentInfo] = []


# ---------- 健康检查 ----------
class HealthResult(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    embedding_loaded: bool = False
    es_connected: bool = False
