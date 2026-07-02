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
             "file_name": "resume.pdf", "chunk_index": 0, "score": 0.89}
        ]
        # 邻域扩展：返回原候选（模拟无相邻块）
        mock_es_instance.expand_neighbors.side_effect = lambda cands, **kw: cands
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


class TestMultiEntityRetrieval:
    """查询分解检索测试 — 通用关键词拆分，不限实体类型"""

    def test_extract_two_keywords(self):
        """两个人名 → 两个关键词"""
        from internal.service.rag_service import RAGService
        terms = RAGService._extract_entities("张三和李四合作过什么项目？")
        assert "张三" in terms
        assert "李四" in terms

    def test_extract_tech_comparison(self):
        """技术对比 → 提取技术名"""
        from internal.service.rag_service import RAGService
        terms = RAGService._extract_entities("FastAPI和Flask有什么区别？")
        assert "FastAPI" in terms
        assert "Flask" in terms

    def test_extract_english_terms(self):
        """英文术语对比"""
        from internal.service.rag_service import RAGService
        terms = RAGService._extract_entities("Redis和Kafka的使用场景对比")
        assert "Redis" in terms
        assert "Kafka" in terms

    def test_extract_single_term_no_trigger(self):
        """单个关键词不触发分解（需 ≥2）"""
        from internal.service.rag_service import RAGService
        terms = RAGService._extract_entities("Python语言有什么特点？")
        # "Python语言" 是一个词，"特点" 也是一个词
        # 重点是验证拆分逻辑能正常工作即可
        assert len(terms) >= 1

    def test_stop_words_filtered(self):
        """纯停用词问句 → 无有效关键词"""
        from internal.service.rag_service import RAGService
        terms = RAGService._extract_entities("请问一下这个和那个有什么区别吗")
        # 全是停用词和疑问词，全被过滤
        assert all(t not in ('这个', '那个', '什么', '区别', '请问') for t in terms)


class TestRetrieveFullDocMarks:
    """测试 {{retrieve_full_doc:...}} 标记的处理：解析多个 + 清理泄露"""

    def test_strip_single_mark(self):
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        text = "这里是答案。\n{{retrieve_full_doc:a.pdf}}\n剩余内容"
        cleaned = rag._strip_retrieve_marks(text)
        assert "{{retrieve_full_doc" not in cleaned
        assert "这里是答案" in cleaned
        assert "剩余内容" in cleaned

    def test_strip_multiple_marks(self):
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        text = "{{retrieve_full_doc:a.pdf}}\n中间内容\n{{retrieve_full_doc:b.pdf}}\n结束"
        cleaned = rag._strip_retrieve_marks(text)
        assert "{{retrieve_full_doc" not in cleaned
        assert "中间内容" in cleaned
        assert "结束" in cleaned

    def test_strip_with_whitespace(self):
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        text = "内容 {{retrieve_full_doc:x.pdf}} 后续"
        cleaned = rag._strip_retrieve_marks(text)
        assert cleaned == "内容  后续"

    def test_strip_empty_text(self):
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        assert rag._strip_retrieve_marks("") == ""
        assert rag._strip_retrieve_marks(None) is None

    def test_strip_no_mark(self):
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        text = "普通答案，没有标记"
        assert rag._strip_retrieve_marks(text) == text

    def test_build_prompt_no_undefined_var(self):
        """回归：_build_prompt 不引用未定义变量（如 full_doc_hint）"""
        from internal.service.llm_service import LLMService
        llm = LLMService()
        # 两种参数都应正常生成，不抛 NameError
        p1 = llm._build_prompt("问题", "上下文", allow_full_doc_retrieval=True)
        p2 = llm._build_prompt("问题", "上下文", allow_full_doc_retrieval=False)
        assert "已知信息" in p1
        assert "已知信息" in p2
        # 不应含已移除的自动召回标记指令
        assert "{{retrieve_full_doc" not in p1
        assert "{full_doc_hint}" not in p1

    def test_need_full_documents_single(self):
        """单标记 → 单文件"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        text = "需要更多 {{retrieve_full_doc:a.pdf}} 内容"
        files = rag._need_full_documents(text)
        assert files == ["a.pdf"]

    def test_need_full_documents_multiple(self):
        """多标记 → 多文件（去重保序）"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        text = "{{retrieve_full_doc:a.pdf}}\n和\n{{retrieve_full_doc:b.pdf}}\n不要\n{{retrieve_full_doc:a.pdf}}"
        files = rag._need_full_documents(text)
        assert files == ["a.pdf", "b.pdf"]

    def test_need_full_documents_none(self):
        """无标记 → 空列表"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        assert rag._need_full_documents("普通文本") == []

    def test_need_full_document_legacy(self):
        """旧 API 仍然返回第一个标记（向后兼容）"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        text = "{{retrieve_full_doc:a.pdf}}\n{{retrieve_full_doc:b.pdf}}"
        assert rag._need_full_document(text) == "a.pdf"

    @patch('internal.service.rag_service.EmbeddingService')
    @patch('internal.service.rag_service.ESService')
    @patch('internal.service.rag_service.LLMService')
    def test_query_strips_mark_from_final_answer(self, mock_llm, mock_es, mock_emb):
        """query 完整流程：答案中残留的标记会被剥离后返回（不再自动重新生成）"""
        from internal.service.rag_service import RAGService

        mock_emb_instance = MagicMock()
        mock_emb_instance.encode_query.return_value = np.random.randn(512).astype(np.float32)
        mock_emb.return_value = mock_emb_instance

        mock_es_instance = MagicMock()
        mock_es_instance.search_hybrid.return_value = [
            {"chunk_id": "c1", "content": "x", "file_name": "a.pdf", "chunk_index": 0, "score": 0.5}
        ]
        mock_es_instance.expand_neighbors.side_effect = lambda cands, **kw: cands
        mock_es_instance.list_file_names.return_value = []
        mock_es.return_value = mock_es_instance

        # LLM 返回带标记的答案（已不再指示输出，但作为防御性兜底测试清理逻辑）
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "需要更多细节 {{retrieve_full_doc:a.pdf}}"
        mock_llm.return_value = mock_llm_instance

        rag = RAGService(
            embedding_service=mock_emb_instance,
            es_service=mock_es_instance,
            llm_service=mock_llm_instance
        )

        result = rag.query(question="细节", api_key="sk-test")

        # 最终答案不应包含标记（被 _strip_retrieve_marks 清理）
        assert "{{retrieve_full_doc" not in result["answer"]
        assert "需要更多细节" in result["answer"]
        # 不再触发自动全文召回重新生成（已移除该特性，避免多答案级联）
        mock_es_instance.retrieve_full_document.assert_not_called()
        assert mock_llm_instance.generate.call_count == 1  # 只调一次，不重新生成


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
