"""
本地 Embedding 服务 - 核心模块
面试核心问题：
1. 为什么用本地 embedding？数据隐私、离线可用、零成本、自主可控
2. 为什么选 BGE-small-zh？中文优化、CPU推理快、1024维平衡、社区活跃
3. 为什么用 sentence-transformers？封装了 mean pooling + L2 归一化，开箱即用
"""
import time
from typing import List, Optional, Dict
from collections import OrderedDict
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
        self.dim = 512  # BGE-small-zh-v1.5 实际输出 512 维（BGE-base 才是 1024）
        self._cache: Dict[str, np.ndarray] = {}  # 文本→向量缓存（入库用）
        self._query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()  # 查询向量 LRU 缓存
        self._query_cache_size: int = 128
        self._loaded = False
        self.logger = get_logger()

    def load_model(self):
        """
        模型加载 — 优先本地路径，找不到再下载。DEBUG 日志打印完整诊断。
        """
        if self.model is not None:
            self.logger.debug("load_model", "embedding", "模型已加载，跳过")
            return

        try:
            from sentence_transformers import SentenceTransformer
            import os
            import glob

            self.logger.debug("load_model", "embedding",
                              "===== 模型加载诊断 =====",
                              cache_folder=self.cache_folder,
                              model_name=self.model_name,
                              cwd=os.getcwd(),
                              abs_cache=os.path.abspath(self.cache_folder))

            # 自动检测设备
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            self.logger.debug("load_model", "embedding",
                              "设备检测", device=device)

            # 优先用 Docker 预下载的本地模型
            # 路径 1: ./models/bge-model （Docker 预下载）
            # 路径 2: ./models/models--BAAI--bge-small-zh-v1.5/snapshots/<hash>/ （HF 默认缓存）
            local_path = os.path.join(self.cache_folder, "bge-model")
            abs_local = os.path.abspath(local_path)
            self.logger.debug("load_model", "embedding",
                              "检查本地模型路径",
                              path=abs_local,
                              exists=os.path.isdir(abs_local))

            if not os.path.isdir(abs_local):
                # 检查 HuggingFace 默认快照目录
                hf_cache = os.path.join(
                    self.cache_folder,
                    f"models--{self.model_name.replace('/', '--')}",
                    "snapshots"
                )
                if os.path.isdir(hf_cache):
                    snapshots = [d for d in os.listdir(hf_cache)
                                 if os.path.isdir(os.path.join(hf_cache, d))]
                    if snapshots:
                        abs_local = os.path.join(hf_cache, snapshots[0])
                        self.logger.info("load_model", "embedding",
                                         "✓ 使用 HuggingFace 缓存快照",
                                         path=abs_local)

            if os.path.isdir(abs_local):
                # 列出路径下的文件
                try:
                    top_files = os.listdir(abs_local)[:20]
                    self.logger.debug("load_model", "embedding",
                                      "本地路径内容", files=top_files)
                except Exception:
                    pass

                # 检查关键文件
                for fname in ['config.json', 'model.safetensors', 'pytorch_model.bin',
                               'tokenizer.json', 'sentence_bert_config.json']:
                    fp = os.path.join(abs_local, fname)
                    exists = os.path.isfile(fp)
                    size = os.path.getsize(fp) if exists else 0
                    self.logger.debug("load_model", "embedding",
                                      f"  关键文件: {fname}",
                                      exists=exists, size=size)

            if os.path.isdir(abs_local) and os.path.isfile(os.path.join(abs_local, "config.json")):
                model_source = abs_local
                self.logger.info("load_model", "embedding",
                                 "✓ 使用预下载本地模型", path=abs_local)
            else:
                model_source = self.model_name
                self.logger.info("load_model", "embedding",
                                 "本地模型不可用，从 HuggingFace 下载",
                                 model=self.model_name,
                                 reason="本地路径不存在或缺少 config.json")

            start = time.time()
            self.logger.debug("load_model", "embedding",
                              "开始加载模型...", source=str(model_source)[:80])
            self.model = SentenceTransformer(
                model_source,
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
        带 LRU 缓存：相同问题文本复用向量，避免重试/相似追问时重复推理。
        面试点：为什么查询要单独缓存？
        答：查询是高频且高度重复的——同一用户重试、相似追问、多路检索
            会反复 encode 同一段文本；CPU 推理 ~50ms，缓存命中即 0ms。
        """
        cfg = get_config()
        conv_cfg = cfg.conversation
        if not conv_cfg.enable_query_vector_cache:
            return self.encode([query])[0]

        # 规范化 key：去首尾空白，避免"张三？"和"张三？ "判为不同
        key = query.strip()
        if not key:
            return self.encode([query])[0]

        cache = self._query_cache
        if key in cache:
            cache.move_to_end(key)  # 命中即提升为最近使用
            return cache[key]

        vec = self.encode([key])[0]
        cache[key] = vec
        size = conv_cfg.query_vector_cache_size or self._query_cache_size
        while len(cache) > size:
            cache.popitem(last=False)  # 淘汰最久未用
        return vec

    def clear_query_cache(self):
        """清空查询向量缓存"""
        self._query_cache.clear()

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
        self._query_cache.clear()


# 全局单例
_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
