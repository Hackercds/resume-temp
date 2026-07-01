"""
配置加载模块 - 面试点：配置优先级 环境变量 > config.yaml > 默认值
遵循 MICRO_SERVICE_SPEC.md 规范：所有配置通过 config.yaml 注入，禁止硬编码
"""
import os
import yaml
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "resume-rag-service"
    host: str = "0.0.0.0"
    port: int = 8080
    mode: str = "release"
    default_api_key: str = ""  # 团队部署时注入，用户无需填写


class LogConfig(BaseModel):
    level: str = "info"
    path: str = "./log"
    filename: str = "app.log"
    maxSize: int = 100
    maxBackups: int = 30
    maxAge: int = 7
    compress: bool = True


class ElasticsearchConfig(BaseModel):
    hosts: List[str] = ["http://localhost:9200"]  # 默认本地。Docker 内用 -e ES_HOST=http://elasticsearch:9200
    index: str = "pdf_knowledge_base"
    vector_dim: int = 512   # BGE-small-zh-v1.5 = 512，BGE-base = 1024
    request_timeout: int = 30


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-zh-v1.5"
    cache_folder: str = "./models"
    max_concurrent: int = 4
    batch_size: int = 32


class ChunkConfig(BaseModel):
    chunk_size: int = 400
    overlap: int = 50
    strategy: str = "fixed"  # fixed | semantic | hybrid
    # CSV 分块时为每行注入表头，保持每行语义完整
    csv_inject_header: bool = True


class LLMConfig(BaseModel):
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"
    timeout: int = 60
    temperature: float = 0.3
    max_tokens: int = 1000
    presets: List[Dict] = Field(default_factory=lambda: [
        {"name": "OpenAI GPT-4o-mini", "provider": "openai", "model": "gpt-4o-mini", "base_url": ""},
        {"name": "硅基流动 DeepSeek-V3", "provider": "openai", "model": "deepseek-chat", "base_url": "https://api.siliconflow.cn/v1"},
        {"name": "MiniMax", "provider": "openai", "model": "abab6.5s-chat", "base_url": "https://api.minimax.chat/v1"},
    ])


class RerankConfig(BaseModel):
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6V2"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    min_score: float = 0.0  # RRF 融合后分数 ≤ 0.033；阈值默认关闭，由 RRF 排名决定
    enable_full_document: bool = True
    # 邻域上下文扩展：命中某 chunk 后，召回同文档相邻 chunk 拼接上下文
    enable_neighbor_expansion: bool = True
    neighbor_window: int = 1  # 前后各扩展 N 个 chunk
    # 文档多样性（MMR 简化版）：确保 top_k 覆盖多个文档，避免单文档垄断答案
    enable_doc_diversity: bool = True
    max_chunks_per_doc: int = 2  # 单文档在 primary 中的最大 chunk 数
    # 上下文内容预算（字符数）：保护模型上下文窗口，按文档分配额度
    context_char_budget: int = 6000


class ConversationConfig(BaseModel):
    max_history: int = 5
    context_window: int = 8000
    enable_full_document: bool = True
    enable_query_rewrite: bool = True
    source_boost: float = 0.15
    source_boost_decay: int = 5
    summary_threshold: int = 3
    enable_intent_llm: bool = False
    intent_rule_confidence_threshold: float = 0.85
    intent_model: str = "gpt-4o-mini"
    # 追问向量检索用 expanded（纯实体消解）而非 context（含对话噪音）
    follow_up_vector_use_expanded: bool = True
    # 查询向量缓存：相同问题文本复用向量，避免重复推理
    enable_query_vector_cache: bool = True
    query_vector_cache_size: int = 128


class CorsConfig(BaseModel):
    allowed_origins: List[str] = [
        "http://localhost:8080",
        "http://localhost:5000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5000",
    ]


class Config(BaseModel):
    app: AppConfig = AppConfig()
    log: LogConfig = LogConfig()
    elasticsearch: ElasticsearchConfig = ElasticsearchConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunk: ChunkConfig = ChunkConfig()
    llm: LLMConfig = LLMConfig()
    rerank: RerankConfig = RerankConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    conversation: ConversationConfig = ConversationConfig()
    cors: CorsConfig = CorsConfig()


def load_config(config_path: str = "config/config.yaml") -> Config:
    """加载配置：config.yaml 为底，环境变量覆盖"""
    config_data = {}

    # 1. 加载 YAML 文件
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}

    # 2. 环境变量覆盖（遵循 ${ENV_VAR} 规范）
    # ES_HOST → elasticsearch.hosts[0]
    if os.getenv("ES_HOST"):
        config_data.setdefault("elasticsearch", {}).setdefault("hosts", [])
        config_data["elasticsearch"]["hosts"][0] = os.getenv("ES_HOST")
    if os.getenv("ES_INDEX"):
        config_data.setdefault("elasticsearch", {})["index"] = os.getenv("ES_INDEX")
    if os.getenv("APP_PORT"):
        config_data.setdefault("app", {})["port"] = int(os.getenv("APP_PORT"))
    if os.getenv("APP_MODE"):
        config_data.setdefault("app", {})["mode"] = os.getenv("APP_MODE")
    if os.getenv("DEFAULT_API_KEY"):
        config_data.setdefault("app", {})["default_api_key"] = os.getenv("DEFAULT_API_KEY")
    if os.getenv("LOG_LEVEL"):
        config_data.setdefault("log", {})["level"] = os.getenv("LOG_LEVEL")

    return Config(**config_data)


# 全局配置实例（进程级单例）
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str = "config/config.yaml") -> Config:
    """重新加载配置（模型更新等场景）"""
    global _config
    _config = load_config(config_path)
    return _config
