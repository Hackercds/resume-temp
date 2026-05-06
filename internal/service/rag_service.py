"""
RAG 核心服务 - 面试核心问题：
1. 完整 RAG 链路是什么？query → embedding → hybrid search → rerank → prompt → LLM
2. 全链路耗时分布？embedding:50-200ms, ES检索:20-50ms, LLM生成:1-3s
3. 有哪些兜底策略？BM25降级、空检索兜底、LLM失败友好提示
"""
import time
import uuid
from typing import List, Dict, Optional

import numpy as np

from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.pkg.errors import (
    ModelLoadError, ESConnectionError, EmptyRetrievalError, LLMAPIError
)
from internal.service.embedding_service import EmbeddingService
from internal.service.es_service import ESService
from internal.service.llm_service import LLMService
from internal.service.rerank_service import RerankService


class RAGService:
    """RAG 核心编排服务 - 面试点：全链路如何串联？"""

    def __init__(self,
                 embedding_service: EmbeddingService = None,
                 es_service: ESService = None,
                 llm_service: LLMService = None,
                 rerank_service: RerankService = None):
        self.embedding = embedding_service or EmbeddingService()
        self.es = es_service or ESService()
        self.llm = llm_service or LLMService()
        self.rerank = rerank_service  # 可选
        self.logger = get_logger()

    def query(self, question: str, api_key: str,
              provider: str = "openai", model: str = None,
              top_k: int = 5) -> Dict:
        """
        RAG 查询主流程 - 面试点：全链路耗时分布
        1. embedding 推理: ~50-200ms (CPU)
        2. ES 混合检索: ~20-50ms
        3. 可选重排: ~50ms/candidate
        4. LLM 生成: ~1-3s

        返回结构：
        {answer, sources, trace_id, timing}
        """
        trace_id = str(uuid.uuid4())[:8]
        timing = {}

        # Step 1: 查询向量化
        embed_start = time.time()
        try:
            query_vector = self.embedding.encode_query(question)
        except ModelLoadError:
            # 降级到纯 BM25
            self.logger.warn(trace_id, "rag_service",
                             "Embedding加载失败，降级到 BM25 检索")
            return self._query_bm25_fallback(question, api_key, provider, model, top_k, trace_id)
        embed_time = (time.time() - embed_start) * 1000
        timing["embedding_ms"] = round(embed_time, 1)
        self.logger.info(trace_id, "rag_service",
                         "Step1 Embedding 完成",
                         duration_ms=round(embed_time, 1))

        # Step 2: ES 混合检索
        search_start = time.time()
        try:
            candidates = self.es.search_hybrid(
                query_vector, query_text=question,
                top_k=top_k * 2,  # 检索 2 倍给重排留空间
                min_score=get_config().retrieval.min_score
            )
        except ESConnectionError as e:
            self.logger.error(trace_id, "rag_service", "ES 连接失败", error=str(e))
            return {
                "answer": "检索服务暂时不可用，请稍后重试。",
                "sources": [],
                "trace_id": trace_id,
                "timing": timing,
                "error": "search_unavailable"
            }
        search_time = (time.time() - search_start) * 1000
        timing["search_ms"] = round(search_time, 1)
        self.logger.info(trace_id, "rag_service",
                         "Step2 ES混合检索完成",
                         duration_ms=round(search_time, 1),
                         candidates=len(candidates))

        # Step 3: 可选重排
        if self.rerank and len(candidates) > top_k:
            candidates = self.rerank.rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # Step 4: 空检索兜底
        if not candidates:
            return {
                "answer": "抱歉，知识库中未找到与您问题相关的信息。\n\n建议：\n1. 尝试用不同的关键词提问\n2. 检查知识库中是否已上传相关文档",
                "sources": [],
                "trace_id": trace_id,
                "timing": timing
            }

        # Step 5: 拼接上下文
        context = self._build_context(candidates)

        # Step 6: 调用 LLM 生成
        llm_start = time.time()
        try:
            answer = self.llm.generate(
                question=question,
                context=context,
                api_key=api_key,
                provider=provider,
                model=model
            )
        except LLMAPIError as e:
            # LLM 失败时返回检索到的原始内容
            self.logger.error(trace_id, "rag_service", "LLM调用失败", error=str(e))
            answer = f"AI 生成服务暂时不可用。以下是知识库中相关的内容（降级模式）：\n\n{context}"
        llm_time = time.time() - llm_start
        timing["llm_s"] = round(llm_time, 1)
        self.logger.info(trace_id, "rag_service",
                         "Step6 LLM 生成完成",
                         duration_s=round(llm_time, 1))

        # 构造返回
        total_time_ms = sum(v for k, v in timing.items())
        self.logger.info(trace_id, "rag_service",
                         "RAG 查询完成",
                         total_ms=round(total_time_ms, 1),
                         sources=len(candidates))

        return {
            "answer": answer,
            "sources": [
                {
                    "content": c.get("content", ""),
                    "file_name": c.get("file_name", ""),
                    "score": round(c.get("rerank_score", c.get("score", 0)), 3)
                }
                for c in candidates
            ],
            "trace_id": trace_id,
            "timing": timing
        }

    def _query_bm25_fallback(self, question: str, api_key: str,
                             provider: str, model: str,
                             top_k: int, trace_id: str) -> Dict:
        """
        BM25 降级方案 - 面试点：即使 embedding 不可用，基本搜索仍可用
        """
        try:
            results = self.es.search_keyword_only(question, top_k)
        except ESConnectionError:
            return {
                "answer": "检索服务暂时不可用，请稍后重试。",
                "sources": [],
                "trace_id": trace_id,
                "fallback": True,
                "error": "search_unavailable"
            }

        if not results:
            return {
                "answer": "抱歉，知识库中未找到相关内容。",
                "sources": [],
                "trace_id": trace_id,
                "fallback": True
            }

        # 降级模式下也尝试调用 LLM（如果有 API Key）
        context = self._build_context(results)
        try:
            answer = self.llm.generate(question, context, api_key, provider, model)
        except Exception:
            # LLM 也不可用时直接返回检索内容
            answer = f"[降级模式-关键词匹配]\n\n{context}"

        return {
            "answer": answer,
            "sources": [
                {"content": r.get("content", ""), "file_name": r.get("file_name", ""), "score": 0}
                for r in results
            ],
            "trace_id": trace_id,
            "fallback": True,
            "timing": {}
        }

    def _build_context(self, candidates: List[Dict]) -> str:
        """
        上下文拼接 - 面试点：Prompt 构造原则
        1. 明确告诉 LLM 信息源
        2. 标注来源，方便验证
        3. 控制长度，不超过 LLM 上下文窗口的 50%
        """
        context_parts = []
        for i, c in enumerate(candidates, 1):
            score = c.get("rerank_score", c.get("score", 0))
            context_parts.append(
                f"【来源{i}】{c['file_name']} (相关度: {score:.2f})\n"
                f"内容: {c['content']}"
            )
        return "\n\n".join(context_parts)
