"""
Rerank 重排服务（可选扩展）- 面试核心问题：
1. 为什么不直接用向量检索 TopK 结果？
   答：向量检索是双编码器，分别编码 query + doc，捕捉粗粒度语义；
       交叉编码器把 query+doc 一起编码，捕捉细粒度交互信息，排序更准确
2. 本地重排用什么模型？
   答：cross-encoder/ms-marco-MiniLM-L-6V2，CPU可跑，延迟<50ms/candidate
3. 为什么重排是可选模块？
   答：追求极致准确率时开启，一般场景向量检索已经够用
"""
from typing import List, Dict, Optional

from internal.model.config import get_config
from internal.pkg.logger import get_logger


class RerankService:
    """
    本地重排服务 - 面试点：
    - 交叉编码器 vs 双编码器：双编码器速度快但粗粒度，交叉编码器慢但精细
    - 流水线：粗排(TopK*2) → 精排(TopK)，兼顾效率与准确率
    """

    def __init__(self, model_name: str = None):
        cfg = get_config()
        rerank_cfg = cfg.rerank
        self.model_name = model_name or rerank_cfg.model_name
        self.model = None
        self.logger = get_logger()

    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self.logger.info("load_model", "rerank",
                             f"重排模型加载完成: {self.model_name}")
        except ImportError:
            self.logger.warn("load_model", "rerank",
                             "sentence-transformers 未安装，重排功能不可用")

    def rerank(self, query: str, candidates: List[Dict],
               top_k: int = 3) -> List[Dict]:
        """
        重排 - 面试点：为什么只重排 TopK 而不是全部？
        答：向量检索已经做了粗排，重排是对 TopK*2 候选做精排
            只关心最相关的几个，不用所有候选都过一遍交叉编码器
        """
        if not candidates or len(candidates) <= top_k:
            return candidates

        # 延迟加载模型
        self._load_model()

        if self.model is None:
            # 模型未加载，返回原序
            return candidates[:top_k]

        # 交叉编码器打分
        pairs = [(query, cand["content"]) for cand in candidates]
        scores = self.model.predict(pairs)

        # 按分数排序
        scored = list(candidates)
        for i, score in enumerate(scores):
            scored[i] = {**scored[i], "rerank_score": float(score)}

        scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return scored[:top_k]

    @property
    def is_available(self) -> bool:
        """检查重排是否可用"""
        self._load_model()
        return self.model is not None
