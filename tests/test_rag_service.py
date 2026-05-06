"""
RAG 核心服务单元测试 - 面试点：Mock 全链路
遵循 MICRO_SERVICE_SPEC.md 表格驱动测试法
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestRRFFusion:
    """RRF 融合算法测试 - 面试核心算法"""

    def test_rrf_basic(self):
        """基本 RRF 融合：两个列表各有不同 chunk"""
        from internal.service.es_service import ESService
        es = ESService.__new__(ESService)

        vector_results = [
            {"chunk_id": "a", "content": "A", "score": 0.9},
            {"chunk_id": "b", "content": "B", "score": 0.8},
            {"chunk_id": "c", "content": "C", "score": 0.5},
        ]
        keyword_results = [
            {"chunk_id": "c", "content": "C", "score": 5.0},
            {"chunk_id": "d", "content": "D", "score": 3.0},
        ]

        fused = es._rrf_fusion(vector_results, keyword_results, k=60)

        # c 在两个列表中都出现，应排第一
        assert fused[0]["chunk_id"] == "c"
        # 结果应包含所有唯一 chunk
        all_ids = {r["chunk_id"] for r in fused}
        assert all_ids == {"a", "b", "c", "d"}

    def test_rrf_empty_keyword(self):
        """只有向量结果时不应报错"""
        from internal.service.es_service import ESService
        es = ESService.__new__(ESService)

        vector_results = [
            {"chunk_id": "a", "content": "A", "score": 0.9},
        ]
        fused = es._rrf_fusion(vector_results, [], k=60)
        assert len(fused) == 1
        assert fused[0]["chunk_id"] == "a"

    def test_rrf_same_rank_both(self):
        """相同排名，都在两个列表中"""
        from internal.service.es_service import ESService
        es = ESService.__new__(ESService)

        # 两个列表排名相同
        v = [{"chunk_id": f"c{i}", "content": f"C{i}", "score": 1.0} for i in range(5)]
        k = [{"chunk_id": f"c{i}", "content": f"C{i}", "score": 1.0} for i in range(5)]

        fused = es._rrf_fusion(v, k, k=60)
        # 每个 chunk 在两个列表中都出现，RRF 分数翻倍
        assert len(fused) == 5


class TestRAGService:
    """RAG 核心服务测试"""

    @patch('internal.service.rag_service.EmbeddingService')
    @patch('internal.service.rag_service.ESService')
    @patch('internal.service.rag_service.LLMService')
    def test_query_success(self, mock_llm, mock_es, mock_emb):
        """正常查询流程"""
        from internal.service.rag_service import RAGService

        # Mock embedding
        mock_emb_instance = MagicMock()
        mock_emb_instance.encode_query.return_value = np.random.randn(1024).astype(np.float32)
        mock_emb.return_value = mock_emb_instance

        # Mock ES
        mock_es_instance = MagicMock()
        mock_es_instance.search_hybrid.return_value = [
            {"chunk_id": "c1", "content": "张成都熟悉Python开发",
             "file_name": "resume.pdf", "score": 0.89}
        ]
        mock_es.return_value = mock_es_instance

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "张成都熟悉Python开发，曾使用FastAPI..."
        mock_llm.return_value = mock_llm_instance

        rag = RAGService(
            embedding_service=mock_emb_instance,
            es_service=mock_es_instance,
            llm_service=mock_llm_instance
        )

        result = rag.query(
            question="张成都有哪些Python经验？",
            api_key="sk-test"
        )

        assert "answer" in result
        assert "sources" in result
        assert "trace_id" in result
        assert "timing" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["file_name"] == "resume.pdf"

    @patch('internal.service.rag_service.EmbeddingService')
    @patch('internal.service.rag_service.ESService')
    @patch('internal.service.rag_service.LLMService')
    def test_query_empty_retrieval(self, mock_llm, mock_es, mock_emb):
        """空检索兜底测试"""
        from internal.service.rag_service import RAGService

        mock_emb_instance = MagicMock()
        mock_emb_instance.encode_query.return_value = np.random.randn(1024).astype(np.float32)
        mock_emb.return_value = mock_emb_instance

        mock_es_instance = MagicMock()
        mock_es_instance.search_hybrid.return_value = []  # 空结果
        mock_es.return_value = mock_es_instance

        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        rag = RAGService(
            embedding_service=mock_emb_instance,
            es_service=mock_es_instance,
            llm_service=mock_llm_instance
        )

        result = rag.query(question="这个问题没有答案", api_key="sk-test")

        # 应该返回兜底回答
        assert "未找到" in result["answer"] or "知识库中" in result["answer"]
        assert len(result["sources"]) == 0

    def test_build_context(self):
        """上下文拼接测试"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)

        candidates = [
            {"chunk_id": "c1", "content": "熟悉Python、Go语言开发",
             "file_name": "a.pdf", "score": 0.9},
            {"chunk_id": "c2", "content": "使用FastAPI构建服务",
             "file_name": "a.pdf", "score": 0.8},
        ]

        context = rag._build_context(candidates)

        assert "【来源1】" in context
        assert "【来源2】" in context
        assert "a.pdf" in context
        assert "熟悉Python" in context
        assert "FastAPI" in context


class TestRateLimiter:
    """限流器测试"""

    def test_acquire_release(self):
        """基本获取和释放"""
        from internal.service.rate_limiter import RateLimiter

        limiter = RateLimiter(max_concurrent=2)
        assert limiter.acquire(timeout=1.0) is True
        assert limiter.acquire(timeout=1.0) is True
        # 第三个获取应立即失败
        assert limiter.acquire(timeout=0.1) is False

        # 释放一个后可以获取
        limiter.release()
        assert limiter.acquire(timeout=0.1) is True

    def test_context_manager(self):
        """上下文管理器测试"""
        from internal.service.rate_limiter import RateLimiter

        limiter = RateLimiter(max_concurrent=2)
        with limiter:
            assert limiter.available == 1
            with limiter:
                assert limiter.available == 0

        assert limiter.available == 2
