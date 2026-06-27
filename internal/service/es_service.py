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

        全链路：query → 向量检索 + BM25 → RRF融合 → 取TopK
        检索量：每路取 max(20, top_k*5)，确保向量与关键词命中都能进入候选。
        """
        # Step 1: 向量检索
        vector_size = max(20, top_k * 5)
        vector_results = self.repo.search_by_vector(query_vector, size=vector_size)

        # Step 2: BM25 关键词检索
        keyword_results = []
        if query_text:
            keyword_results = self.repo.search_by_keyword(query_text, size=vector_size)

        # Step 3: RRF 融合
        if not keyword_results:
            # 只有向量结果，直接返回（向量 score 已经是 cosineSimilarity）
            return self._filter_by_score(vector_results, min_score)[:top_k]

        if not vector_results:
            # 只有 BM25 结果（min_score 直接用于 BM25 score，因为没有融合）
            return keyword_results[:top_k]

        fused = self._rrf_fusion(vector_results, keyword_results, k=60)
        # min_score 在融合前过滤太激进（RRF 分数最大 0.033，远小于 0.5）。
        # 这里改为对 RRF 分数过滤，并写入 _score_rrf 字段。
        return self._filter_by_rrf_score(fused, min_score)[:top_k]

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

        把每个 doc 的 RRF 分数写入 _rrf_score 字段并作为排序依据。
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

        # 按 RRF 分数排序，并把 RRF 分数写入每条结果
        sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result = []
        for cid in sorted_cids:
            item = dict(all_items[cid])  # 复制避免污染原 dict
            item["_rrf_score"] = round(scores[cid], 6)
            result.append(item)
        return result

    def _filter_by_score(self, results: List[Dict], min_score: float) -> List[Dict]:
        """过滤低分结果（用于纯向量或纯 BM25 检索）"""
        if min_score <= 0:
            return results
        return [r for r in results if r.get("score", 0) >= min_score]

    def _filter_by_rrf_score(self, results: List[Dict], min_score: float) -> List[Dict]:
        """
        RRF 融合后的过滤：min_score 应该按 RRF 分数阈值处理。
        RRF 分数最大 1/(60+1)*2 ≈ 0.033，所以 min_score 默认为 0 即可。
        但为避免极弱命中干扰 top_k，可设 0.0001 这种极小阈值。
        """
        if min_score <= 0:
            return results
        return [r for r in results if r.get("_rrf_score", 0) >= min_score]

    def retrieve_full_document(self, file_name: str) -> Optional[Dict]:
        """通过 file_name 召回整篇文档"""
        return self.repo.get_full_document(file_name)

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
