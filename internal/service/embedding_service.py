"""
本地 Embedding 服务 - 核心模块
面试核心问题：
1. 为什么用本地 embedding？数据隐私、离线可用、零成本、自主可控
2. 为什么选 BGE-small-zh？中文优化、CPU推理快、1024维平衡、社区活跃
3. 为什么用 sentence-transformers？封装了 mean pooling + L2 归一化，开箱即用
"""
import time
from typing import List, Optional, Dict
import numpy as np

from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.pkg.errors import ModelLoadError


class EmbeddingService:
    """
    本地 Embedding 服务
    面试点：
    - 模型选择：BAAI/bge-small-zh-v1.5（智源开源，中文C-MTEB排行榜Top）
    - small 版本：专为 CPU 推理优化，单条 < 100ms
    - 1024 维度：ES 存储和检索效率平衡
    """

    def __init__(self, model_name: str = None, cache_folder: str = None,
                 max_concurrent: int = None):
        cfg = get_config()
        emb_cfg = cfg.embedding
        self.model_name = model_name or emb_cfg.model_name
        self.cache_folder = cache_folder or emb_cfg.cache_folder
        self.max_concurrent = max_concurrent or emb_cfg.max_concurrent
        self.model = None
        self.dim = 1024  # BGE-small-zh 输出维度
        self._cache: Dict[str, np.ndarray] = {}  # 文本→向量缓存
        self._loaded = False
        self.logger = get_logger()

    def load_model(self):
        """
        模型加载 - 面试点：首次加载慢（~10-30s），如何优化？
        - 启动时后台线程预热
        - 模型文件持久化到本地磁盘
        - Docker 构建时预下载（COPY 到镜像内）
        """
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # 自动检测设备：GPU > CPU
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

            start = time.time()
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=self.cache_folder
            )
            # 验证模型维度
            self.dim = self.model.get_sentence_embedding_dimension()
            elapsed = time.time() - start

            self._loaded = True
            self.logger.info("load_model", "embedding",
                             f"模型加载完成: {self.model_name}",
                             device=device, dim=self.dim, elapsed_s=round(elapsed, 1))
        except Exception as e:
            self.logger.error("load_model", "embedding",
                              f"模型加载失败: {self.model_name}",
                              error=str(e))
            raise ModelLoadError(f"模型加载失败: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        批量向量化 - 面试点：
        - batch 处理：多条文本一起推理，CPU SIMD 指令加速
        - L2 归一化：余弦相似度等价于内积，检索更快
        """
        self.load_model()
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        查询向量化 - 面试点：查询和文档用同一个模型，保证向量空间一致
        """
        return self.encode([query])[0]

    def encode_cached(self, texts: List[str]) -> np.ndarray:
        """
        带缓存的向量化 - 面试点：相同文本避免重复推理
        适用于文档入库时可能有重复 chunk 的场景
        """
        self.load_model()
        result = np.zeros((len(texts), self.dim), dtype=np.float32)

        # 检查缓存
        to_encode_indices = []
        to_encode_texts = []
        for i, text in enumerate(texts):
            if text in self._cache:
                result[i] = self._cache[text]
            else:
                to_encode_indices.append(i)
                to_encode_texts.append(text)

        # 批量推理未缓存的
        if to_encode_texts:
            embeddings = self.encode(to_encode_texts)
            for j, (idx, text) in enumerate(zip(to_encode_indices, to_encode_texts)):
                result[idx] = embeddings[j]
                self._cache[text] = embeddings[j]

        return result

    def encode_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        分批向量化 - 面试点：为什么分批？
        - 避免单批过大导致 CPU 打满/OOM
        - 每批 32-100 条，CPU 推理吞吐最优
        """
        if batch_size is None:
            batch_size = get_config().embedding.batch_size

        self.load_model()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(np.array(embeddings, dtype=np.float32))

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def clear_cache(self):
        """清空向量缓存"""
        self._cache.clear()


# 全局单例
_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
