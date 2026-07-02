"""
检索召回与对话质量升级 - 专项单元测试
覆盖：
- IK 中文分词探测与降级
- 全文父文档不参与常规检索（防污染）
- 邻域上下文扩展
- 查询向量 LRU 缓存
- 追问向量去污染（follow_up 用 expanded）
- CSV 表头继承
- 指代消解增强（多代词）
- 历史压缩保留 assistant 答案
- LLM 意图识别真实可用
- 空检索动态建议
- context 邻域块标注
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ==================== IK 中文分词探测 ====================
class TestIKAnalyzerDetection:
    """IK 分词器探测与降级"""

    def test_detect_ik_available_true(self):
        """ES 已装 analysis-ik → 返回 True"""
        from internal.repository.es_repository import ESRepository
        repo = ESRepository.__new__(ESRepository)
        repo.logger = MagicMock()
        mock_es = MagicMock()
        mock_es.cat.plugins.return_value = [
            {"component": "analysis-ik", "name": "analysis-ik"}
        ]
        assert repo._detect_ik_available(mock_es) is True

    def test_detect_ik_available_false(self):
        """ES 未装 ik → 返回 False"""
        from internal.repository.es_repository import ESRepository
        repo = ESRepository.__new__(ESRepository)
        repo.logger = MagicMock()
        mock_es = MagicMock()
        mock_es.cat.plugins.return_value = [
            {"component": "analysis-icu", "name": "analysis-icu"}
        ]
        assert repo._detect_ik_available(mock_es) is False

    def test_detect_ik_exception_fallback(self):
        """探测异常 → 降级 False，不抛错"""
        from internal.repository.es_repository import ESRepository
        repo = ESRepository.__new__(ESRepository)
        repo.logger = MagicMock()
        mock_es = MagicMock()
        mock_es.cat.plugins.side_effect = Exception("network")
        assert repo._detect_ik_available(mock_es) is False


# ==================== 全文父文档不参与检索 ====================
class TestFullDocExclusion:
    """父文档（is_full_doc=true）不应参与向量/关键词检索"""

    def test_search_by_vector_excludes_full_doc(self):
        """向量检索用 must_not(term:true) 排除父文档（兼容字段缺失的常规 chunk）"""
        from internal.repository.es_repository import ESRepository
        repo = ESRepository.__new__(ESRepository)
        repo.logger = MagicMock()
        repo.index = "test_idx"

        mock_es = MagicMock()
        mock_es.search.return_value = {"hits": {"hits": []}}
        repo.connect = MagicMock(return_value=mock_es)

        qv = np.random.randn(512).astype(np.float32)
        repo.search_by_vector(qv, size=5)

        body = mock_es.search.call_args.kwargs.get("body") or mock_es.search.call_args.args[1]
        query = body["query"]
        assert "script_score" in query
        inner = query["script_score"]["query"]
        # 应是 bool.must_not(term:true)，而非 term:false（后者会误排除字段缺失的常规 chunk）
        assert "bool" in inner
        assert {"term": {"is_full_doc": True}} in inner["bool"]["must_not"]

    def test_search_by_vector_can_include_full_doc(self):
        """exclude_full_doc=False 时不加过滤（用于全文召回路径）"""
        from internal.repository.es_repository import ESRepository
        repo = ESRepository.__new__(ESRepository)
        repo.logger = MagicMock()
        repo.index = "test_idx"

        mock_es = MagicMock()
        mock_es.search.return_value = {"hits": {"hits": []}}
        repo.connect = MagicMock(return_value=mock_es)

        qv = np.random.randn(512).astype(np.float32)
        repo.search_by_vector(qv, size=5, exclude_full_doc=False)

        body = mock_es.search.call_args.kwargs.get("body") or mock_es.search.call_args.args[1]
        inner = body["query"]["script_score"]["query"]
        # match_all 表示没加过滤
        assert inner == {"match_all": {}}

    def test_search_by_keyword_excludes_full_doc(self):
        """BM25 检索用 must_not(term:true) 排除父文档"""
        from internal.repository.es_repository import ESRepository
        repo = ESRepository.__new__(ESRepository)
        repo.logger = MagicMock()
        repo.index = "test_idx"

        mock_es = MagicMock()
        mock_es.search.return_value = {"hits": {"hits": []}}
        repo.connect = MagicMock(return_value=mock_es)

        repo.search_by_keyword("张三", size=5)

        body = mock_es.search.call_args.kwargs.get("body") or mock_es.search.call_args.args[1]
        bool_body = body["query"]["bool"]
        assert {"term": {"is_full_doc": True}} in bool_body["must_not"]

    def test_regular_chunks_without_field_are_not_excluded(self):
        """
        回归测试：常规 chunk 入库时未写 is_full_doc（字段缺失），
        检索必须仍能命中——不能用 term:false（会误排除缺失字段文档）。
        这是 v1.2 升级的关键数据语义：must_not(term:true) 而非 term:false。
        """
        from internal.repository.es_repository import ESRepository
        repo = ESRepository.__new__(ESRepository)
        repo.logger = MagicMock()
        repo.index = "test_idx"

        captured_bodies = []
        mock_es = MagicMock()
        # 模拟 ES 返回一条常规 chunk（_source 里没有 is_full_doc 字段）
        mock_es.search.side_effect = lambda *a, **k: (
            captured_bodies.append(k.get("body") or (a[1] if len(a) > 1 else None)),
            {"hits": {"hits": [{"_source": {"chunk_id": "c1", "content": "x",
                                            "file_name": "a.pdf", "chunk_index": 0},
                                 "_score": 1.5}]}})[1]
        repo.connect = MagicMock(return_value=mock_es)

        qv = np.random.randn(512).astype(np.float32)
        results = repo.search_by_vector(qv, size=5)
        # 常规 chunk（字段缺失）应被返回，不被 must_not 误排除
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        # 验证查询体用的是 must_not(term:true)，不是 term:false
        body = captured_bodies[0]
        inner = body["query"]["script_score"]["query"]
        assert "bool" in inner
        assert "must_not" in inner["bool"]
        assert {"term": {"is_full_doc": True}} in inner["bool"]["must_not"]
        # 关键：filter 里不能有 term:is_full_doc=false
        filters = inner["bool"].get("filter", [])
        assert not any(f == {"term": {"is_full_doc": False}} for f in filters)


# ==================== 邻域上下文扩展 ====================
class TestNeighborExpansion:
    """邻域扩展：命中 chunk 后召回同文档相邻 chunk"""

    def test_expand_neighbors_adds_adjacent(self):
        """命中 chunk_index=2，window=1 → 召回 1,3"""
        from internal.service.es_service import ESService
        es = ESService.__new__(ESService)
        es.logger = MagicMock()

        candidates = [
            {"chunk_id": "a_2", "content": "C2", "file_name": "a.pdf",
             "chunk_index": 2, "score": 0.8},
        ]
        neighbors_returned = [
            {"chunk_id": "a_1", "content": "C1", "file_name": "a.pdf",
             "chunk_index": 1, "section_title": "", "score": 0.0},
            {"chunk_id": "a_3", "content": "C3", "file_name": "a.pdf",
             "chunk_index": 3, "section_title": "", "score": 0.0},
        ]
        es.repo = MagicMock()
        es.repo.search_neighbor_chunks.return_value = neighbors_returned

        result = es.expand_neighbors(candidates, window=1)

        # 候选 1 + 邻域 2 = 3
        assert len(result) == 3
        ids = {c["chunk_id"] for c in result}
        assert ids == {"a_2", "a_1", "a_3"}
        # 邻域块应追加在末尾
        assert result[0]["chunk_id"] == "a_2"

    def test_expand_neighbors_dedup(self):
        """已命中块不重复召回"""
        from internal.service.es_service import ESService
        es = ESService.__new__(ESService)
        es.logger = MagicMock()

        candidates = [
            {"chunk_id": "a_2", "content": "C2", "file_name": "a.pdf",
             "chunk_index": 2, "score": 0.8},
        ]
        # 邻域查询返回了已命中的 a_2
        es.repo = MagicMock()
        es.repo.search_neighbor_chunks.return_value = [
            {"chunk_id": "a_2", "content": "C2", "file_name": "a.pdf",
             "chunk_index": 2, "score": 0.0},
        ]
        result = es.expand_neighbors(candidates, window=1)
        assert len(result) == 1  # 去重后只剩 1

    def test_expand_neighbors_skip_full_doc(self):
        """父文档候选不触发邻域扩展"""
        from internal.service.es_service import ESService
        es = ESService.__new__(ESService)
        es.logger = MagicMock()
        es.repo = MagicMock()

        candidates = [
            {"chunk_id": "a__full", "content": "全文", "file_name": "a.pdf",
             "chunk_index": -1, "score": 0.0, "is_full_doc": True},
        ]
        result = es.expand_neighbors(candidates, window=1)
        # 无邻域查询
        es.repo.search_neighbor_chunks.assert_not_called()
        assert result == candidates

    def test_expand_neighbors_window_zero(self):
        """window=0 → 不扩展"""
        from internal.service.es_service import ESService
        es = ESService.__new__(ESService)
        es.logger = MagicMock()
        es.repo = MagicMock()
        candidates = [{"chunk_id": "a_0", "content": "x", "file_name": "a.pdf",
                       "chunk_index": 0, "score": 0.5}]
        result = es.expand_neighbors(candidates, window=0)
        assert result == candidates

    def test_search_neighbor_chunks_builds_correct_query(self):
        """邻域查询按 file_name + terms(chunk_index) + must_not(is_full_doc=true) 过滤"""
        from internal.repository.es_repository import ESRepository
        repo = ESRepository.__new__(ESRepository)
        repo.logger = MagicMock()
        repo.index = "idx"
        mock_es = MagicMock()
        mock_es.search.return_value = {"hits": {"hits": []}}
        repo.connect = MagicMock(return_value=mock_es)

        repo.search_neighbor_chunks("a.pdf", [1, 3])
        body = mock_es.search.call_args.kwargs.get("body") or mock_es.search.call_args.args[1]
        bool_body = body["query"]["bool"]
        filters = bool_body["filter"]
        assert {"term": {"file_name": "a.pdf"}} in filters
        # is_full_doc 排除应放在 must_not（兼容字段缺失），不在 filter
        assert {"term": {"is_full_doc": True}} in bool_body["must_not"]
        # terms 含 1 和 3
        terms_filter = next(f for f in filters if "terms" in f)
        assert set(terms_filter["terms"]["chunk_index"]) == {1, 3}


# ==================== 查询向量 LRU 缓存 ====================
class TestQueryVectorCache:
    """查询向量缓存：相同文本复用，避免重复推理"""

    def test_cache_hit_avoids_recompute(self):
        from internal.service.embedding_service import EmbeddingService
        emb = EmbeddingService()
        emb._query_cache.clear()

        fake_cfg = MagicMock()
        fake_cfg.conversation.enable_query_vector_cache = True
        fake_cfg.conversation.query_vector_cache_size = 128
        with patch('internal.service.embedding_service.get_config', return_value=fake_cfg):
            call_count = {"n": 0}
            def fake_encode(texts, normalize=True):
                call_count["n"] += 1
                return np.ones((1, 512), dtype=np.float32) * call_count["n"]
            emb.encode = fake_encode

            v1 = emb.encode_query("张三的项目")
            v2 = emb.encode_query("张三的项目")
            assert call_count["n"] == 1  # 第二次命中缓存
            assert np.array_equal(v1, v2)

    def test_cache_miss_different_text(self):
        from internal.service.embedding_service import EmbeddingService
        emb = EmbeddingService()
        emb._query_cache.clear()

        fake_cfg = MagicMock()
        fake_cfg.conversation.enable_query_vector_cache = True
        fake_cfg.conversation.query_vector_cache_size = 128
        with patch('internal.service.embedding_service.get_config', return_value=fake_cfg):
            emb.encode = lambda texts, normalize=True: np.random.randn(1, 512).astype(np.float32)
            v1 = emb.encode_query("问题A")
            v2 = emb.encode_query("问题B")
            assert not np.array_equal(v1, v2)

    def test_cache_lru_eviction(self):
        """超过容量淘汰最久未用"""
        from internal.service.embedding_service import EmbeddingService
        from collections import OrderedDict
        emb = EmbeddingService()
        emb._query_cache = OrderedDict()

        fake_cfg = MagicMock()
        fake_cfg.conversation.enable_query_vector_cache = True
        fake_cfg.conversation.query_vector_cache_size = 2
        with patch('internal.service.embedding_service.get_config', return_value=fake_cfg):
            counter = {"n": 0}
            def fake_encode(texts, normalize=True):
                counter["n"] += 1
                return np.ones((1, 512), dtype=np.float32) * counter["n"]
            emb.encode = fake_encode

            emb.encode_query("q1")  # cache: q1
            emb.encode_query("q2")  # cache: q1, q2
            emb.encode_query("q3")  # 淘汰 q1, cache: q2, q3
            assert len(emb._query_cache) == 2
            assert "q1" not in emb._query_cache
            assert "q3" in emb._query_cache

    def test_cache_disabled(self):
        """关闭缓存时每次都重新 encode"""
        from internal.service.embedding_service import EmbeddingService
        emb = EmbeddingService()
        emb._query_cache.clear()

        fake_cfg = MagicMock()
        fake_cfg.conversation.enable_query_vector_cache = False
        fake_cfg.conversation.query_vector_cache_size = 128
        with patch('internal.service.embedding_service.get_config', return_value=fake_cfg):
            call_count = {"n": 0}
            def fake_encode(texts, normalize=True):
                call_count["n"] += 1
                return np.ones((1, 512), dtype=np.float32)
            emb.encode = fake_encode

            emb.encode_query("same")
            emb.encode_query("same")
            assert call_count["n"] == 2  # 无缓存，调用两次
            assert len(emb._query_cache) == 0


# ==================== 追问向量去污染 ====================
class TestFollowUpVectorDepollution:
    """follow_up 主向量用 expanded（纯实体消解）而非 context（含对话套话）"""

    @patch('internal.service.rag_service.EmbeddingService')
    @patch('internal.service.rag_service.ESService')
    @patch('internal.service.rag_service.LLMService')
    def test_follow_up_uses_expanded_for_vector(self, mock_llm, mock_es, mock_emb):
        """追问时 encode_query 收到的是 expanded，不是 context"""
        from internal.service.rag_service import RAGService

        encoded_questions = []
        def fake_encode_query(q):
            encoded_questions.append(q)
            return np.random.randn(512).astype(np.float32)

        mock_emb_instance = MagicMock()
        mock_emb_instance.encode_query.side_effect = fake_encode_query
        mock_emb.return_value = mock_emb_instance

        mock_es_instance = MagicMock()
        mock_es_instance.search_hybrid.return_value = [
            {"chunk_id": "c1", "content": "x", "file_name": "a.pdf",
             "chunk_index": 0, "score": 0.5}
        ]
        mock_es_instance.expand_neighbors.side_effect = lambda c, **kw: c
        mock_es.return_value = mock_es_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "答案"
        mock_llm.return_value = mock_llm_instance

        rag = RAGService(
            embedding_service=mock_emb_instance,
            es_service=mock_es_instance,
            llm_service=mock_llm_instance
        )

        # 模拟追问：有历史，问题含「他」
        history = [
            {"role": "user", "content": "张成都的项目经历？"},
            {"role": "assistant", "content": "张成都做过项目A", "sources": []},
        ]
        rag.query(question="他的项目用了什么技术？", api_key="sk-test", history=history)

        # encode_query 应被调用；传入的应是 expanded（含「张成都」），而非 context（含「基于之前」）
        assert len(encoded_questions) >= 1
        # 至少有一次 encode 用的是去污染版本（不含「基于之前关于」套话）
        assert any("基于之前关于" not in q for q in encoded_questions)


# ==================== CSV 表头继承 ====================
class TestCSVHeaderInheritance:
    """CSV 每行注入表头"""

    def test_header_injected_into_rows(self):
        from internal.service.chunk_service import ChunkService
        svc = ChunkService()
        csv_content = "name,age,city\nAlice,25,Beijing\nBob,30,Shanghai"
        chunks = svc.chunk_csv(csv_content, "test.csv")
        assert len(chunks) == 2
        assert "name: Alice" in chunks[0]["content"]
        assert "age: 25" in chunks[0]["content"]
        assert "city: Beijing" in chunks[0]["content"]
        assert "name: Bob" in chunks[1]["content"]

    def test_single_column_no_header_mode(self):
        """单列文本不当作有表头，按行分块"""
        from internal.service.chunk_service import ChunkService
        svc = ChunkService()
        chunks = svc.chunk_csv("line1\nline2\nline3", "t.txt")
        assert len(chunks) == 3
        # 内容就是行本身，不注入"表头:"
        assert chunks[0]["content"] == "line1"
        assert chunks[2]["content"] == "line3"

    def test_quoted_field_with_comma(self):
        """带引号的 CSV 字段含逗号正确解析"""
        from internal.service.chunk_service import ChunkService
        svc = ChunkService()
        csv_content = 'name,desc\nAlice,"年龄,30"\nBob,工程师'
        chunks = svc.chunk_csv(csv_content, "t.csv")
        assert len(chunks) == 2
        assert "desc: 年龄,30" in chunks[0]["content"]

    def test_empty_values_skipped(self):
        """空值字段跳过，不生成「col: 」"""
        from internal.service.chunk_service import ChunkService
        svc = ChunkService()
        csv_content = "a,b,c\n1,,3"
        chunks = svc.chunk_csv(csv_content, "t.csv")
        assert "a: 1" in chunks[0]["content"]
        assert "c: 3" in chunks[0]["content"]
        assert "b:" not in chunks[0]["content"]


# ==================== 指代消解增强 ====================
class TestReferenceResolutionEnhanced:
    """指代消解：多代词/多次出现"""

    def test_single_pronoun_multiple_occurrences(self):
        """「他的项目和他的经验」→ 两处「他」都替换"""
        from internal.service.intent_service import IntentService
        resolved = IntentService.resolve_references(
            "他的项目和他的经验是什么？", ["张成都"])
        # 单实体场景：所有「他」应被替换
        assert "张成都的项目" in resolved
        assert "张成都的经验" in resolved

    def test_this_that_with_noun(self):
        """「这个项目」→ 主实体+项目"""
        from internal.service.intent_service import IntentService
        resolved = IntentService.resolve_references(
            "这个项目用了什么技术？", ["DCS缓存"])
        assert "DCS缓存项目" in resolved

    def test_gai_ci_prefix(self):
        """「该项目」→ 主实体项目"""
        from internal.service.intent_service import IntentService
        resolved = IntentService.resolve_references(
            "该项目进展如何", ["DCS缓存"])
        assert "DCS缓存项目" in resolved

    def test_multi_entity_conservative(self):
        """多实体：只替换句首代词，避免错配"""
        from internal.service.intent_service import IntentService
        resolved = IntentService.resolve_references(
            "他和她的关系", ["张三", "李四"])
        # 句首「他」替换为张三，第二个「她」保留（避免错配）
        assert resolved.startswith("张三")
        assert "她" in resolved

    def test_qitai_not_misreplaced(self):
        """「其他」中的「他」不被误替换"""
        from internal.service.intent_service import IntentService
        resolved = IntentService.resolve_references(
            "他了解其他技术吗", ["张三"])
        assert "张三了解其他技术" in resolved
        # 「其他」保持完整
        assert "其他" in resolved

    def test_no_entities_returns_original(self):
        """无实体 → 原样返回"""
        from internal.service.intent_service import IntentService
        assert IntentService.resolve_references("他的项目", []) == "他的项目"


# ==================== 历史压缩保留 assistant 答案 ====================
class TestHistoryCompressionKeepsAnswers:
    """历史压缩：保留更早的「问→答」配对摘要，不只 user 问句"""

    def test_compression_keeps_assistant_answer(self):
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        # 8 条消息 = 4 轮，threshold=3（>3*2=6 触发压缩）
        history = [
            {"role": "user", "content": "张成都是谁？"},
            {"role": "assistant", "content": "张成都是5年测试开发工程师，熟悉Python"},
            {"role": "user", "content": "他的项目？"},
            {"role": "assistant", "content": "做过DCS缓存系统，优化了响应时间"},
            {"role": "user", "content": "最近问什么1？"},
            {"role": "assistant", "content": "最近答1"},
            {"role": "user", "content": "最近问2？"},
            {"role": "assistant", "content": "最近答2"},
        ]
        compressed = rag._compress_history(history, threshold=3)
        # 应有 1 个 system 摘要 + 最近 4 条
        assert compressed[0]["role"] == "system"
        summary = compressed[0]["content"]
        # 摘要应包含 assistant 答案的要点（旧版只保留 user 问句）
        assert "测试开发" in summary or "DCS" in summary
        # 最近 2 轮完整保留
        assert len(compressed) == 5  # 1 system + 4 recent

    def test_short_history_not_compressed(self):
        """短历史不压缩"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        assert rag._compress_history(history, threshold=3) == history


# ==================== LLM 意图识别真实可用 ====================
class TestLLMIntentClassification:
    """LLM 意图识别不再是 NotImplementedError"""

    def test_llm_classify_parses_json(self):
        """LLM 返回合法 JSON → 正确解析意图"""
        from internal.service.intent_service import IntentService
        svc = IntentService.__new__(IntentService)
        svc.logger = MagicMock()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"intent": "follow_up", "confidence": 0.9}'
        svc.llm = mock_llm

        with patch('internal.service.intent_service.get_config') as mock_cfg:
            full = MagicMock()
            full.conversation.intent_model = "gpt-4o-mini"
            full.app.default_api_key = ""
            mock_cfg.return_value = full

            intent, conf = svc._llm_classify(
                "他的项目", [{"role": "user", "content": "张三"},
                            {"role": "assistant", "content": "张三是..."}],
                {"api_key": "sk-test"}
            )
        assert intent == "follow_up"
        assert 0.85 <= conf <= 0.95

    def test_llm_classify_unparseable_falls_back(self):
        """LLM 返回非 JSON → 抛错（由 classify 捕获回退规则）"""
        from internal.service.intent_service import IntentService
        svc = IntentService.__new__(IntentService)
        svc.logger = MagicMock()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "我觉得这是追问"
        svc.llm = mock_llm

        with patch('internal.service.intent_service.get_config') as mock_cfg:
            full = MagicMock()
            full.conversation.intent_model = "gpt-4o-mini"
            full.app.default_api_key = ""
            mock_cfg.return_value = full

            with pytest.raises(Exception):
                svc._llm_classify(
                    "他的项目", [{"role": "user", "content": "张三"}],
                    {"api_key": "sk-test"}
                )

    def test_llm_classify_invalid_intent_corrected(self):
        """LLM 返回非法 intent 值 → 回退到 follow_up"""
        from internal.service.intent_service import IntentService
        svc = IntentService.__new__(IntentService)
        svc.logger = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"intent": "unknown_thing", "confidence": 0.8}'
        svc.llm = mock_llm

        with patch('internal.service.intent_service.get_config') as mock_cfg:
            full = MagicMock()
            full.conversation.intent_model = "gpt-4o-mini"
            full.app.default_api_key = ""
            mock_cfg.return_value = full
            intent, _ = svc._llm_classify(
                "x", [{"role": "user", "content": "y"}], {"api_key": "sk-test"})
        assert intent == "follow_up"

    def test_classify_with_llm_kwargs_passed_through(self):
        """classify 透传 llm_kwargs 到 _llm_classify"""
        from internal.service.intent_service import IntentService
        svc = IntentService()
        svc.llm = MagicMock()
        svc.llm.generate.return_value = '{"intent": "summarize", "confidence": 0.7}'

        with patch('internal.service.intent_service.get_config') as mock_cfg:
            conv = MagicMock()
            conv.enable_intent_llm = True
            conv.intent_rule_confidence_threshold = 0.99  # 强制走 LLM
            conv.intent_model = "gpt-4o-mini"
            full = MagicMock()
            full.conversation = conv
            full.app.default_api_key = ""
            mock_cfg.return_value = full

            intent, _ = svc.classify(
                "随便问一个低置信度的问题xyz",
                [{"role": "user", "content": "上文"}, {"role": "assistant", "content": "答"}],
                llm_kwargs={"api_key": "sk-test", "provider": "openai"}
            )
        # 应调用了 LLM generate
        svc.llm.generate.assert_called_once()
        assert intent == "summarize"


# ==================== 空检索动态建议 ====================
class TestEmptyRetrievalSuggestion:
    """空检索时返回知识库实际文档名"""

    def test_suggestion_lists_documents(self):
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        rag.es = MagicMock()
        rag.es.list_file_names.return_value = [
            {"file_name": "张三简历.pdf"},
            {"file_name": "项目文档.md"},
        ]
        suggestion = rag._build_empty_retrieval_suggestion()
        assert "张三简历.pdf" in suggestion
        assert "项目文档.md" in suggestion

    def test_suggestion_truncates_many_docs(self):
        """超过5份文档显示「等共N份」"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        rag.es = MagicMock()
        rag.es.list_file_names.return_value = [
            {"file_name": f"doc{i}.pdf"} for i in range(8)
        ]
        suggestion = rag._build_empty_retrieval_suggestion()
        assert "共 8 份" in suggestion or "等共" in suggestion

    def test_suggestion_empty_kb(self):
        """空知识库 → 返回基础建议"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        rag.es = MagicMock()
        rag.es.list_file_names.return_value = []
        suggestion = rag._build_empty_retrieval_suggestion()
        assert "尝试用不同关键词" in suggestion

    def test_suggestion_es_error_fallback(self):
        """ES 异常 → 返回基础建议，不抛错"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        rag.es = MagicMock()
        rag.es.list_file_names.side_effect = Exception("es down")
        suggestion = rag._build_empty_retrieval_suggestion()
        assert "尝试用不同关键词" in suggestion


# ==================== context 邻域块标注 ====================
class TestContextNeighborLabeling:
    """_build_context 标注邻域补充块"""

    def test_neighbor_blocks_labeled(self):
        """邻域块标注为【上下文补充N】"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        candidates = [
            {"chunk_id": "a_2", "content": "主命中", "file_name": "a.pdf",
             "chunk_index": 2, "score": 0.8, "section_title": ""},
            {"chunk_id": "a_1", "content": "邻域前", "file_name": "a.pdf",
             "chunk_index": 1, "score": 0.0, "section_title": ""},
            {"chunk_id": "a_3", "content": "邻域后", "file_name": "a.pdf",
             "chunk_index": 3, "score": 0.0, "section_title": ""},
        ]
        ctx = rag._build_context(candidates)
        # 主命中标【来源1】
        assert "【来源1】" in ctx
        # 邻域标【上下文补充】
        assert "【上下文补充" in ctx
        assert "邻域前" in ctx
        assert "邻域后" in ctx

    def test_full_doc_label(self):
        """完整文档标【完整文档】"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        candidates = [
            {"chunk_id": "a__full", "content": "全文", "file_name": "a.pdf",
             "chunk_index": -1, "score": 0.0, "is_full_doc": True}
        ]
        ctx = rag._build_context(candidates)
        assert "【完整文档】" in ctx

    def test_empty_candidates(self):
        """空候选 → 空字符串"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        assert rag._build_context([]) == ""


# ==================== 文档多样性选择 ====================
class TestDocumentDiversity:
    """文档多样性：per-doc cap + MMR 多文档覆盖"""

    def _make(self, file_name, idx, score):
        return {"chunk_id": f"{file_name}_{idx}", "content": f"c{idx}",
                "file_name": file_name, "chunk_index": idx, "score": score}

    def test_caps_per_document(self):
        """5 个候选全来自同一文档，max_per_doc=2 → 第一轮取2，回退补齐到 top_k"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        candidates = [
            self._make("a.pdf", 0, 0.9),
            self._make("a.pdf", 1, 0.8),
            self._make("a.pdf", 2, 0.7),
            self._make("a.pdf", 3, 0.6),
            self._make("a.pdf", 4, 0.5),
        ]
        primary = rag._diversify_by_document(candidates, top_k=3, max_per_doc=2)
        assert len(primary) == 3  # 回退补齐到 top_k
        # 第一轮取了前 2（最高分），回退补第 3
        assert primary[0]["chunk_id"] == "a.pdf_0"
        assert primary[1]["chunk_id"] == "a.pdf_1"

    def test_multi_doc_coverage(self):
        """候选来自多个文档，确保每个文档都进入 primary（回答「其他文档一次引用」）"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        # 3 个文档各有 2 个 chunk
        candidates = [
            self._make("a.pdf", 0, 0.95),
            self._make("a.pdf", 1, 0.90),
            self._make("b.pdf", 0, 0.85),
            self._make("b.pdf", 1, 0.80),
            self._make("c.pdf", 0, 0.75),
            self._make("c.pdf", 1, 0.70),
        ]
        primary = rag._diversify_by_document(candidates, top_k=3, max_per_doc=1)
        # top_k=3, max_per_doc=1 → 3 个不同文档各取 1
        docs = {c["file_name"] for c in primary}
        assert docs == {"a.pdf", "b.pdf", "c.pdf"}
        # 取的是各文档最高分
        assert primary[0]["chunk_id"] == "a.pdf_0"

    def test_max_per_doc_2_with_multiple_docs(self):
        """多文档 + max_per_doc=2：优先铺开文档，每文档至多2"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        candidates = [
            self._make("a.pdf", 0, 0.95), self._make("a.pdf", 1, 0.90),
            self._make("a.pdf", 2, 0.85),
            self._make("b.pdf", 0, 0.80), self._make("b.pdf", 1, 0.75),
            self._make("c.pdf", 0, 0.70),
        ]
        primary = rag._diversify_by_document(candidates, top_k=4, max_per_doc=2)
        assert len(primary) == 4
        from collections import Counter
        doc_counts = Counter(c["file_name"] for c in primary)
        # a 最多2，b 最多2，c 至少1
        assert doc_counts["a.pdf"] <= 2
        assert doc_counts["b.pdf"] <= 2

    def test_fallback_fills_when_docs_few(self):
        """候选文档数 < top_k：回退忽略 cap 补齐，保证信息量"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        candidates = [
            self._make("a.pdf", 0, 0.9),
            self._make("a.pdf", 1, 0.8),
            self._make("a.pdf", 2, 0.7),
        ]
        # top_k=5 但只有 3 个候选，max_per_doc=2 → 第一轮取2，回退补1，共3
        primary = rag._diversify_by_document(candidates, top_k=5, max_per_doc=2)
        assert len(primary) == 3  # 不超过候选总数

    def test_empty_candidates(self):
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        assert rag._diversify_by_document([], top_k=5, max_per_doc=2) == []

    def test_max_per_doc_zero_no_cap(self):
        """max_per_doc=0 → 不裁剪，按分数取 top_k"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        candidates = [self._make("a.pdf", i, 0.9 - i * 0.1) for i in range(5)]
        primary = rag._diversify_by_document(candidates, top_k=3, max_per_doc=0)
        assert len(primary) == 3
        assert primary[0]["score"] == 0.9

    def test_uses_rerank_score_when_present(self):
        """有 rerank_score 时按 rerank_score 排序选"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        candidates = [
            {"chunk_id": "a_0", "file_name": "a.pdf", "chunk_index": 0,
             "score": 0.1, "rerank_score": 0.95},
            {"chunk_id": "b_0", "file_name": "b.pdf", "chunk_index": 0,
             "score": 0.9, "rerank_score": 0.30},
        ]
        primary = rag._diversify_by_document(candidates, top_k=2, max_per_doc=1)
        # rerank 高的 a_0 应排第一
        assert primary[0]["chunk_id"] == "a_0"

    def test_entity_match_docs_prioritized_for_coverage(self):
        """实体命中文档优先覆盖：即使 RRF 分数低，entity_match 文档先占席位"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        # 候选：5个不同文档，其中涉警/计算机/简历标记 entity_match
        candidates = [
            {"chunk_id": "面试_0", "file_name": "面试题.pdf", "chunk_index": 0, "score": 0.05},  # RRF最高
            {"chunk_id": "涉警_0", "file_name": "涉警.pdf", "chunk_index": 0, "score": 0.03, "entity_match": True},
            {"chunk_id": "计算机_0", "file_name": "计算机.pdf", "chunk_index": 0, "score": 0.02, "entity_match": True},
            {"chunk_id": "简历_0", "file_name": "简历.pdf", "chunk_index": 0, "score": 0.04, "entity_match": True},
            {"chunk_id": "amap_0", "file_name": "amap.pdf", "chunk_index": 0, "score": 0.01},
        ]
        primary = rag._diversify_by_document(candidates, top_k=4, max_per_doc=2)
        # entity_match 文档应优先占席位（涉警、计算机、简历 + 面试题1个）
        primary_docs = {c["file_name"] for c in primary}
        assert "涉警.pdf" in primary_docs
        assert "计算机.pdf" in primary_docs
        assert "简历.pdf" in primary_docs

    def test_entity_chunk_preferred_over_higher_score(self):
        """entity_match 文档优先选 entity_chunk（含实体词），而非更高分的非实体chunk"""
        from internal.service.rag_service import RAGService
        rag = RAGService.__new__(RAGService)
        # 涉警论文有两个 chunk：#20 案例分数高但不含实体，#0 作者页含实体
        candidates = [
            {"chunk_id": "涉警_20", "file_name": "涉警.pdf", "chunk_index": 20,
             "score": 0.398, "entity_match": True},  # 案例页，不含张成都
            {"chunk_id": "涉警_0", "file_name": "涉警.pdf", "chunk_index": 0,
             "score": 0.341, "entity_match": True, "entity_chunk": True},  # 作者页，含张成都
            {"chunk_id": "简历_0", "file_name": "简历.pdf", "chunk_index": 0,
             "score": 0.04, "entity_match": True, "entity_chunk": True},
            {"chunk_id": "计算机_0", "file_name": "计算机.pdf", "chunk_index": 0,
             "score": 0.04, "entity_match": True, "entity_chunk": True},
        ]
        primary = rag._diversify_by_document(candidates, top_k=4, max_per_doc=2)
        涉警 = [c for c in primary if c["file_name"] == "涉警.pdf"]
        assert len(涉警) >= 1
        # 应选 #0（entity_chunk）而非 #20（案例）
        assert 涉警[0]["chunk_index"] == 0


# ==================== 内容预算 ====================
class TestContextCharBudget:
    """上下文按文档分配字符预算，回答「该用多少内容」"""

    def test_per_doc_budget_truncation(self):
        """单文档内容超预算 → 截断并提示（budget 有 1000 下限，用 2000 字内容触发）"""
        from internal.service.rag_service import RAGService
        from unittest.mock import patch, MagicMock
        rag = RAGService.__new__(RAGService)

        fake_cfg = MagicMock()
        fake_cfg.retrieval.context_char_budget = 200  # 被下限抬到 1000
        with patch('internal.service.rag_service.get_config', return_value=fake_cfg):
            candidates = [
                {"chunk_id": "a_0", "content": "X" * 2000, "file_name": "a.pdf",
                 "chunk_index": 0, "score": 0.9, "rerank_score": 0.9},
            ]
            ctx = rag._build_context(candidates)
        assert "已按文档预算截断" in ctx
        # 内容被截断到 1000，不再含 2000 个 X
        assert ctx.count("X") < 2000

    def test_multi_doc_budget_split(self):
        """多文档预算均分，各文档都保留内容"""
        from internal.service.rag_service import RAGService
        from unittest.mock import patch, MagicMock
        rag = RAGService.__new__(RAGService)

        fake_cfg = MagicMock()
        fake_cfg.retrieval.context_char_budget = 2000
        with patch('internal.service.rag_service.get_config', return_value=fake_cfg):
            candidates = [
                {"chunk_id": "a_0", "content": "甲" * 100, "file_name": "a.pdf",
                 "chunk_index": 0, "score": 0.9, "rerank_score": 0.9},
                {"chunk_id": "b_0", "content": "乙" * 100, "file_name": "b.pdf",
                 "chunk_index": 0, "score": 0.8, "rerank_score": 0.8},
            ]
            ctx = rag._build_context(candidates)
        # 两个文档内容都应在上下文里
        assert "甲" in ctx
        assert "乙" in ctx

    def test_full_doc_not_truncated(self):
        """完整文档（用户手动召回）不参与预算截断"""
        from internal.service.rag_service import RAGService
        from unittest.mock import patch, MagicMock
        rag = RAGService.__new__(RAGService)

        fake_cfg = MagicMock()
        fake_cfg.retrieval.context_char_budget = 100  # 极小
        with patch('internal.service.rag_service.get_config', return_value=fake_cfg):
            candidates = [
                {"chunk_id": "a__full", "content": "全文内容" * 200, "file_name": "a.pdf",
                 "chunk_index": -1, "score": 0.0, "is_full_doc": True}
            ]
            ctx = rag._build_context(candidates)
        assert "已按文档预算截断" not in ctx
        assert "全文内容" in ctx


# ==================== 邻居进入上下文（修复回归）====================
class TestNeighborsReachContext:
    """关键回归：邻居扩展后不应被 [:top_k] 切掉，必须进入上下文"""

    @patch('internal.service.rag_service.EmbeddingService')
    @patch('internal.service.rag_service.ESService')
    @patch('internal.service.rag_service.LLMService')
    def test_neighbors_in_context_not_cut(self, mock_llm, mock_es, mock_emb):
        """top_k=1，命中1块 + 召回2邻居 → 上下文含邻居，sources 只含 primary"""
        import numpy as np
        from internal.service.rag_service import RAGService

        mock_emb_instance = MagicMock()
        mock_emb_instance.encode_query.return_value = np.random.randn(512).astype(np.float32)
        mock_emb.return_value = mock_emb_instance

        mock_es_instance = MagicMock()
        # search 返回 2 个候选（同文档），让 diversity cap=2 都进 primary
        mock_es_instance.search_hybrid.return_value = [
            {"chunk_id": "a_2", "content": "主命中C2", "file_name": "a.pdf",
             "chunk_index": 2, "score": 0.9},
        ]
        # 邻居扩展返回 主命中 + 邻居1,3
        def expand(cands, window=1, **kw):
            return cands + [
                {"chunk_id": "a_1", "content": "邻居C1", "file_name": "a.pdf",
                 "chunk_index": 1, "section_title": "", "score": 0.0},
                {"chunk_id": "a_3", "content": "邻居C3", "file_name": "a.pdf",
                 "chunk_index": 3, "section_title": "", "score": 0.0},
            ]
        mock_es_instance.expand_neighbors.side_effect = expand
        mock_es_instance.list_file_names.return_value = []
        mock_es.return_value = mock_es_instance

        captured_context = {"ctx": None}
        def fake_generate(question, context, **kw):
            captured_context["ctx"] = context
            return "答案"
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = fake_generate
        mock_llm.return_value = mock_llm_instance

        rag = RAGService(embedding_service=mock_emb_instance,
                         es_service=mock_es_instance, llm_service=mock_llm_instance)
        result = rag.query(question="测试", api_key="sk-test", top_k=1)

        ctx = captured_context["ctx"]
        # 邻居内容必须出现在上下文里（旧 bug 会被 [:top_k] 切掉）
        assert "邻居C1" in ctx
        assert "邻居C3" in ctx
        # sources 只含 primary（1 个），不含邻居
        assert len(result["sources"]) == 1
        assert result["sources"][0]["file_name"] == "a.pdf"
        # trace 记录邻居数
        assert result["timing"].get("neighbor_expanded") == 2


# ==================== 端到端：多文档一次引用 ====================
class TestEndToEndMultiDocDiversity:
    """端到端：检索到多文档候选时，primary 覆盖多文档"""

    @patch('internal.service.rag_service.EmbeddingService')
    @patch('internal.service.rag_service.ESService')
    @patch('internal.service.rag_service.LLMService')
    def test_primary_covers_multiple_docs(self, mock_llm, mock_es, mock_emb):
        import numpy as np
        from internal.service.rag_service import RAGService
        from unittest.mock import patch as _patch, MagicMock

        mock_emb_instance = MagicMock()
        mock_emb_instance.encode_query.return_value = np.random.randn(512).astype(np.float32)
        mock_emb.return_value = mock_emb_instance

        mock_es_instance = MagicMock()
        # 检索返回 6 候选，来自 3 个文档（每文档2个）
        mock_es_instance.search_hybrid.return_value = [
            {"chunk_id": f"{d}_{i}", "content": f"{d}内容{i}",
             "file_name": f"{d}.pdf", "chunk_index": i, "score": 0.9 - i * 0.1 - n * 0.05}
            for n, d in enumerate(["a", "b", "c"]) for i in range(2)
        ]
        mock_es_instance.expand_neighbors.side_effect = lambda c, **kw: c
        mock_es_instance.list_file_names.return_value = []
        mock_es.return_value = mock_es_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "答案"
        mock_llm.return_value = mock_llm_instance

        # max_per_doc=1 强制每文档至多 1 个，top_k=3 → 3 个不同文档各 1
        fake_cfg = MagicMock()
        fake_cfg.retrieval.max_chunks_per_doc = 1
        fake_cfg.retrieval.enable_doc_diversity = True
        fake_cfg.retrieval.enable_neighbor_expansion = True
        fake_cfg.retrieval.neighbor_window = 1
        fake_cfg.retrieval.enable_full_document = True
        fake_cfg.retrieval.min_score = 0.0
        fake_cfg.retrieval.context_char_budget = 6000
        fake_cfg.conversation = MagicMock(source_boost=0.0, summary_threshold=3,
                                           follow_up_vector_use_expanded=True,
                                           enable_query_vector_cache=False,
                                           query_vector_cache_size=128,
                                           enable_query_rewrite=True, enable_intent_llm=False,
                                           intent_rule_confidence_threshold=0.85,
                                           enable_full_document=True, max_history=5,
                                           context_window=8000, source_boost_decay=5,
                                           intent_model="gpt-4o-mini")

        with _patch('internal.service.rag_service.get_config', return_value=fake_cfg):
            rag = RAGService(embedding_service=mock_emb_instance,
                             es_service=mock_es_instance, llm_service=mock_llm_instance)
            result = rag.query(question="有哪些相关内容", api_key="sk-test", top_k=3)

        # sources 应覆盖 3 个不同文档（max_per_doc=1, top_k=3）
        source_docs = {s["file_name"] for s in result["sources"]}
        assert len(source_docs) == 3
        assert result["trace"]["primary_count"] == 3


# ==================== 实体文档扩展 ====================
class TestEntityDocumentExpansion:
    """单实体多文档召回：解决「张成都是谁」只召回一篇论文"""

    def test_extract_entity_terms_filters_generic_words(self):
        """普通名词（项目/总结）不当作实体"""
        from internal.service.rag_service import RAGService
        assert RAGService._extract_entity_terms("他的项目用了什么") == []
        assert RAGService._extract_entity_terms("总结一下") == []
        # 人名/技术名保留
        assert "张成都" in RAGService._extract_entity_terms("张成都是谁")
        assert "FastAPI" in RAGService._extract_entity_terms("FastAPI和Flask区别")

    def test_expand_injects_new_docs(self):
        """实体 BM25 命中候选池里没有的文档 → 注入代表 chunk"""
        from internal.service.rag_service import RAGService
        from unittest.mock import MagicMock
        rag = RAGService.__new__(RAGService)
        rag.logger = MagicMock()
        rag.es = MagicMock()
        # BM25「张成都」命中 3 个文档，文件名含「张成都」→ name_match 判定为实体文档
        rag.es.search_keyword_only.return_value = [
            {"chunk_id": "涉警_0", "content": "张成都", "file_name": "涉警-张成都.pdf", "score": 3.0},
            {"chunk_id": "计算机_0", "content": "作者张成都", "file_name": "4.张成都-计算机.pdf", "score": 2.8},
            {"chunk_id": "简历_0", "content": "张成都简历", "file_name": "张成都-简历.pdf", "score": 2.5},
        ]
        existing = [{"chunk_id": "涉警_0", "file_name": "涉警-张成都.pdf", "score": 0.4}]
        reps = rag._expand_entity_documents(["张成都"], existing, top_k=3)
        # 应注入计算机和简历（涉警已在候选）
        new_docs = {r["file_name"] for r in reps}
        assert "4.张成都-计算机.pdf" in new_docs
        assert "张成都-简历.pdf" in new_docs
        assert "涉警-张成都.pdf" not in new_docs
        # 注入的代表标记 entity_match 且分数归一化到 RRF 量级
        for r in reps:
            assert r.get("entity_match") is True
            assert r["score"] == 0.04

    def test_expand_picks_highest_score_rep_per_doc(self):
        """同一文档多个命中 → 取分数最高的代表"""
        from internal.service.rag_service import RAGService
        from unittest.mock import MagicMock
        rag = RAGService.__new__(RAGService)
        rag.logger = MagicMock()
        rag.es = MagicMock()
        rag.es.search_keyword_only.return_value = [
            {"chunk_id": "a_0", "content": "x", "file_name": "a.pdf", "score": 1.0},
            {"chunk_id": "a_1", "content": "y", "file_name": "a.pdf", "score": 5.0},
        ]
        reps = rag._expand_entity_documents(["张三"], [], top_k=3)
        assert len(reps) == 1
        assert reps[0]["chunk_id"] == "a_1"  # 分数高的

    def test_expand_no_entities_returns_empty(self):
        """无实体 → 不扩展"""
        from internal.service.rag_service import RAGService
        from unittest.mock import MagicMock
        rag = RAGService.__new__(RAGService)
        rag.logger = MagicMock()
        rag.es = MagicMock()
        assert rag._expand_entity_documents([], [{"file_name": "a.pdf"}], 3) == []

    @patch('internal.service.rag_service.EmbeddingService')
    @patch('internal.service.rag_service.ESService')
    @patch('internal.service.rag_service.LLMService')
    def test_single_entity_query_recalls_multiple_docs(self, mock_llm, mock_es, mock_emb):
        """端到端：「张成都是谁」应让两篇论文都进入 sources"""
        import numpy as np
        from internal.service.rag_service import RAGService

        mock_emb_instance = MagicMock()
        mock_emb_instance.encode_query.return_value = np.random.randn(512).astype(np.float32)
        mock_emb.return_value = mock_emb_instance

        mock_es_instance = MagicMock()
        # 向量检索只偏向涉警那篇
        mock_es_instance.search_hybrid.return_value = [
            {"chunk_id": "涉警_0", "content": "张成都涉警论文", "file_name": "涉警.pdf",
             "chunk_index": 0, "score": 0.4},
        ]
        # BM25「张成都」命中两篇论文
        def kw_only(query, size=10, **kw):
            if "张成都" in query:
                return [
                    {"chunk_id": "涉警_0", "content": "张成都", "file_name": "涉警.pdf", "score": 3.0},
                    {"chunk_id": "计算机_0", "content": "作者张成都", "file_name": "计算机.pdf", "score": 2.8},
                ]
            return []
        mock_es_instance.search_keyword_only.side_effect = kw_only
        mock_es_instance.expand_neighbors.side_effect = lambda c, **kw: c
        mock_es_instance.list_file_names.return_value = []
        mock_es.return_value = mock_es_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "答案"
        mock_llm.return_value = mock_llm_instance

        rag = RAGService(embedding_service=mock_emb_instance,
                         es_service=mock_es_instance, llm_service=mock_llm_instance)
        result = rag.query(question="张成都是谁", api_key="sk-test", top_k=3)

        # 两篇论文都应在 sources 里（实体扩展注入了第二篇）
        source_docs = {s["file_name"] for s in result["sources"]}
        assert "涉警.pdf" in source_docs
        assert "计算机.pdf" in source_docs
        # trace 记录实体扩展
        assert result["timing"].get("entity_doc_expanded") == 1
        assert result["trace"].get("entity_terms") == ["张成都"]


