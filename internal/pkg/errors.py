"""
集中化错误定义 - 面试点：为什么要有统一错误体系？
答：1) 便于前端统一处理  2) 日志可追踪  3) 错误码区间化管理
遵循 MICRO_SERVICE_SPEC.md 错误码规范
"""


class RAGException(Exception):
    """RAG 系统基础异常"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# ---------- 系统级错误 50001-50099 ----------
class ModelLoadError(RAGException):
    """Embedding 模型加载失败"""
    def __init__(self, detail: str = "Embedding模型加载失败"):
        super().__init__(50001, detail)


class ESConnectionError(RAGException):
    """ES 连接失败"""
    def __init__(self, detail: str = "Elasticsearch 连接失败"):
        super().__init__(50002, detail)


class LLMAPIError(RAGException):
    """LLM API 调用失败"""
    def __init__(self, detail: str = "LLM API 调用失败"):
        super().__init__(50003, detail)


class InternalError(RAGException):
    """内部未知错误"""
    def __init__(self, detail: str = "内部服务错误"):
        super().__init__(50004, detail)


# ---------- 参数错误 40001-40099 ----------
class InvalidParameterError(RAGException):
    """参数校验失败"""
    def __init__(self, detail: str = "参数错误"):
        super().__init__(40001, detail)


class UnsupportedFileTypeError(RAGException):
    """不支持的文件格式"""
    def __init__(self, detail: str = "不支持的文件格式，仅支持 PDF/TXT/CSV"):
        super().__init__(40002, detail)


class FileTooLargeError(RAGException):
    """文件过大"""
    def __init__(self, max_mb: int = 10):
        super().__init__(40003, f"文件大小超过 {max_mb}MB 限制")


# ---------- 资源不存在 40401-40499 ----------
class EmptyRetrievalError(RAGException):
    """知识库检索为空"""
    def __init__(self, detail: str = "未找到相关内容"):
        super().__init__(40401, detail)


class DocumentNotFoundError(RAGException):
    """文档不存在"""
    def __init__(self, file_name: str = ""):
        super().__init__(40402, f"文档不存在: {file_name}")


EmptyKnowledgeBaseError = EmptyRetrievalError  # 别名兼容
