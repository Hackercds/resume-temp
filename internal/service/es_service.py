"""
Elasticsearch 检索服务 - 面试核心问题：
1. 为什么用 ES 而不是纯向量数据库？已有ES环境，零部署成本，支持混合检索
2. 混合检索原理？向量语义匹配 + BM25关键词匹配，RRF融合取交集
3. RRF 为什么好？不受两个子系统分数分布差异影响，学术界和工业界验证有效
"""
from typing import List, Dict, Optional
import numpy as np

from internal.model.config import get_config
from internal.repository.es_repository import ESRepository, get_es_repository
from internal.pkg.logger import get_logger


class ESService:
    """ES 检索服务 - 在 Repository 之上封装业务逻辑（混合检索、RRF融合）"""

    def __init__(self, repo: ESRepository = None):
        self.repo = repo or get_es_repository()
        self.logger = get_logger()
        self.index = self.repo.index

    def search_hybrid(self, query_vector: np.ndarray, query_text: str = "",
                      top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """
        混合检索：向量相似度 + BM25 关键词，RRF 融合
        面试点：
        - 为什么用 hybrid search？纯向量检索对专有名词不敏感，BM25 兜底
        - RRF (k=60) 公认参数，学术论文验证

        全链路：query → 向量检索(TopK*2) + BM25(TopK*2) → RRF融合 → 取TopK
        """
        # Step 1: 向量检索
        vector_results = self.repo.search_by_vector(query_vector, size=top_k * 3)

        # Step 2: BM25 关键词检索
        keyword_results = []
        if query_text:
            keyword_results = self.repo.search_by_keyword(query_text, size=top_k * 2)

        # Step 3: RRF 融合
        if not keyword_results:
            # 只有向量结果，直接返回
            return self._filter_by_score(vector_results, min_score)[:top_k]

        if not vector_results:
            return keyword_results[:top_k]

        fused = self._rrf_fusion(vector_results, keyword_results, k=60)
        return self._filter_by_score(fused, min_score)[:top_k]

    def search_vector_only(self, query_vector: np.ndarray,
                           top_k: int = 5) -> List[Dict]:
        """纯向量检索"""
        return self.repo.search_by_vector(query_vector, size=top_k)

    def search_keyword_only(self, query_text: str,
                            top_k: int = 5) -> List[Dict]:
        """
        BM25 降级检索 - 面试点：embedding 不可用时的降级方案
        ES text 字段默认 BM25 算法
        """
        return self.repo.search_by_keyword(query_text, size=top_k)

    def _rrf_fusion(self, vector_results: List[Dict],
                    keyword_results: List[Dict],
                    k: int = 60) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合 - 面试核心问题：RRF 计算公式？
        score(d) = Σ 1/(k + rank_i(d))
        其中 k=60 是学术界公认参数

        面试话术：
        "RRF 对排名取倒数再融合，不受两个子系统分数分布差异影响。
         向量检索的分数是 [0,2]，BM25 的分数可能是 (0, 30+)，
         如果直接加权，需要先做归一化，RRF 天然不受影响"
        """
        scores = {}
        for rank, item in enumerate(vector_results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)

        for rank, item in enumerate(keyword_results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)

        # 合并元数据
        all_items = {}
        for item in vector_results + keyword_results:
            cid = item["chunk_id"]
            if cid not in all_items:
                all_items[cid] = item

        # 按 RRF 分数排序
        sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [all_items[cid] for cid in sorted_cids]

    def _filter_by_score(self, results: List[Dict], min_score: float) -> List[Dict]:
        """过滤低分结果"""
        if min_score <= 0:
            return results
        return [r for r in results if r.get("score", 0) >= min_score]

    # ---------- 索引管理 ----------
    def ensure_index(self) -> bool:
        """确保索引存在，不存在则创建"""
        if not self.repo.index_exists():
            self.logger.info("ensure_index", "es_service",
                             f"索引不存在，正在创建: {self.index}")
            self.repo.create_index()
            return False
        return True

    def create_index(self, index_name: str = None) -> Dict:
        return self.repo.create_index(index_name)

    def bulk_insert(self, chunks: List[Dict], vectors: np.ndarray) -> int:
        return self.repo.bulk_insert(chunks, vectors)

    def delete_by_file_name(self, file_name: str) -> int:
        return self.repo.delete_by_file_name(file_name)

    def list_file_names(self) -> List[Dict]:
        return self.repo.list_file_names()

    def count(self) -> Dict:
        return self.repo.count()

    def switch_alias(self, new_index: str, old_index: str) -> Dict:
        return self.repo.switch_alias(new_index, old_index)

    def is_connected(self) -> bool:
        return self.repo.is_connected()
