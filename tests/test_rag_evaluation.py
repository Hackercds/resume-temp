"""
RAG 效果评估 - 面试点：如何评估 RAG 系统效果？
指标：Recall@K, MRR, 答案准确率
"""
import pytest
import numpy as np


class TestRAGEvaluation:
    """RAG 效果评估测试"""

    def test_recall_at_k(self):
        """Recall@K 计算"""
        # 模拟检索结果
        retrieved_chunks = [
            "张成都熟悉Python、Go语言开发",
            "使用Python + Redis实现DCS缓存",
            "了解Kubernetes和Docker技术"
        ]
        expected_keywords = ["Python", "Redis", "FastAPI"]

        # 计算召回率
        retrieved_text = " ".join(retrieved_chunks)
        recall = sum(1 for kw in expected_keywords if kw in retrieved_text) / len(expected_keywords)

        # Python 和 Redis 应该能匹配到，FastAPI 不在结果中
        assert recall == 2 / 3
        assert recall > 0.5

    def test_mrr_basic(self):
        """MRR (Mean Reciprocal Rank) 计算"""
        # 正确答案在排名 2
        results = [
            {"content": "无关内容A", "score": 0.5},
            {"content": "包含Python开发经验", "score": 0.3},
            {"content": "无关内容B", "score": 0.1},
        ]

        mrr = 0
        expected_keywords = ["Python"]
        for rank, result in enumerate(results, 1):
            if any(kw in result["content"] for kw in expected_keywords):
                mrr = 1 / rank
                break

        assert mrr == 1 / 2  # 排名第二
        assert mrr > 0

    def test_mrr_first_rank(self):
        """MRR: 正确答案在第一位"""
        results = [
            {"content": "包含Python和Go开发经验"},
            {"content": "无关内容"},
        ]

        mrr = 0
        expected_keywords = ["Python"]
        for rank, result in enumerate(results, 1):
            if any(kw in result["content"] for kw in expected_keywords):
                mrr = 1 / rank
                break

        assert mrr == 1.0  # 排名第一

    def test_mrr_not_found(self):
        """MRR: 正确答案不在结果中"""
        results = [
            {"content": "无关内容A"},
            {"content": "无关内容B"},
        ]

        mrr = 0
        expected_keywords = ["Python"]
        for rank, result in enumerate(results, 1):
            if any(kw in result["content"] for kw in expected_keywords):
                mrr = 1 / rank
                break

        assert mrr == 0  # 没找到

    def test_evaluation_metrics(self):
        """综合评估指标计算"""
        test_cases = [
            {
                "question": "张成都有哪些Python经验？",
                "retrieved": ["熟悉Python开发", "使用FastAPI"],
                "expected_keywords": ["Python", "FastAPI"],
                "expected_rank": 1  # 期望排名
            },
            {
                "question": "张成都用过哪些技术？",
                "retrieved": ["了解Docker技术", "熟悉Python开发", "使用Redis"],
                "expected_keywords": ["Docker", "Redis"],
                "expected_rank": 2
            },
        ]

        recalls = []
        mrrs = []

        for case in test_cases:
            retrieved_text = " ".join(case["retrieved"])
            recall = sum(
                1 for kw in case["expected_keywords"] if kw in retrieved_text
            ) / len(case["expected_keywords"])
            recalls.append(recall)

            mrr = 0
            for rank, content in enumerate(case["retrieved"], 1):
                if any(kw in content for kw in case["expected_keywords"]):
                    mrr = 1 / rank
                    break
            mrrs.append(mrr)

        avg_recall = np.mean(recalls)
        avg_mrr = np.mean(mrrs)

        assert avg_recall > 0
        assert avg_mrr > 0
        assert avg_recall <= 1.0
        assert avg_mrr <= 1.0


class TestDTOValidation:
    """DTO 请求校验测试"""

    def test_query_request_valid(self):
        """正常查询请求"""
        from internal.model.dto import QueryRequest
        req = QueryRequest(question="测试问题", api_key="sk-test", top_k=5)
        assert req.question == "测试问题"
        assert req.provider == "openai"
        assert req.top_k == 5

    def test_query_request_empty_question(self):
        """空问题应校验失败"""
        from internal.model.dto import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(question="", api_key="sk-test")

    def test_api_response_format(self):
        """统一响应格式"""
        from internal.model.dto import APIResponse
        resp = APIResponse.success(data={"answer": "test"})
        d = resp.model_dump()
        assert d["code"] == 0
        assert d["message"] == "success"
        assert "requestId" in d
        assert "timestamp" in d
