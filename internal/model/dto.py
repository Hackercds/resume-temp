"""
数据传输对象 - Request/Response Models
面试点：Pydantic 自动校验输入，防止非法数据进入后端
遵循 MICRO_SERVICE_SPEC.md 统一响应格式：{code, message, data, requestId, timestamp}
"""
import uuid
import time
from typing import Optional, List, Any, Dict
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
class ChatMessage(BaseModel):
    """对话历史消息"""
    role: str = Field(..., pattern=r"^(user|assistant|system)$",
                      description="消息角色: user / assistant / system")
    content: str = Field(..., min_length=1, description="消息内容")


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
    history: List[ChatMessage] = Field(default_factory=list, description="历史对话消息")
    session_id: Optional[str] = Field(default=None, description="会话ID，可选")
    retrieve_full_doc: bool = Field(default=False, description="是否强制召回整篇文档")


# ---------- 查询响应 ----------
class SourceItem(BaseModel):
    content: str
    file_name: str
    score: float = 0.0
    section_title: str = ""
    chunk_index: int = 0
    is_full_doc: bool = False


class TimingInfo(BaseModel):
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    llm_s: float = 0.0


class QueryTrace(BaseModel):
    """检索过程追踪，用于调试与前端展示"""
    trace_id: str = ""
    intent: Optional[str] = None
    original_question: str = ""
    expanded_question: str = ""
    context_question: str = ""
    rewrite_method: str = "none"
    candidates_before_boost: int = 0
    candidates_after_boost: int = 0
    source_boosts: Dict[str, float] = Field(default_factory=dict)
    full_doc_requested: bool = False
    error: Optional[str] = None


class QueryResult(BaseModel):
    answer: str
    sources: List[SourceItem] = []
    trace_id: str = ""
    timing: Optional[TimingInfo] = None
    trace: Optional[QueryTrace] = None
    suggestion: Optional[str] = None
    empty_retrieval: bool = False
    fallback_context: Optional[str] = None


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
    default_api_key_configured: bool = False
