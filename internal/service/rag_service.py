"""
RAG 核心服务 - 面试核心问题：
1. 完整 RAG 链路是什么？query → embedding → hybrid search → rerank → prompt → LLM
2. 多实体关系查询（张三和李四的合作）怎么查？
   答：查询分解 → 多路独立检索 → 去重融合
3. 有哪些兜底策略？BM25降级、空检索兜底、LLM失败友好提示
"""
import re
import time
import uuid
from typing import List, Dict, Optional, Set

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
              base_url: str = None, top_k: int = 5) -> Dict:
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

        # Step 4: 空检索 / 低质量结果 → 尝试多实体分解检索
        # ⚠️ 触发条件严格：1) 结果<3条  2) 问题含≥2个人名
        # 正常查询完全跳过此步骤，时延影响 = 0
        # 面试点：为什么不每次都做？时延 × N，99% 的查询不需要
        if not candidates or len(candidates) < 3:
            entities = self._extract_entities(question)
            if len(entities) >= 2:
                self.logger.info(trace_id, "rag_service",
                                 "向量检索结果不足，启用多实体分解检索",
                                 entities=entities, original_count=len(candidates))
                decomposed = self._multi_entity_retrieve(question, entities, top_k)
                if decomposed:
                    # 合并原始结果和分解检索结果
                    seen = {c["chunk_id"] for c in candidates}
                    for c in decomposed:
                        if c["chunk_id"] not in seen:
                            candidates.append(c)
                            seen.add(c["chunk_id"])
                    timing["decomposed"] = True
                    timing["decomposed_count"] = len(decomposed)

        # Step 5: 检索仍为空时的兜底
        if not candidates:
            return {
                "answer": "抱歉，知识库中未找到与您问题相关的信息。\n\n"
                          "建议：\n"
                          "1. 尝试用不同的关键词提问\n"
                          "2. 检查知识库中是否已上传相关文档\n"
                          "3. 如果是多个人名的关系查询，可分别搜索每个人的信息",
                "sources": [],
                "trace_id": trace_id,
                "timing": timing
            }

        # Step 6: 拼接上下文
        context = self._build_context(candidates)

        # Step 7: 调用 LLM 生成
        llm_start = time.time()
        try:
            answer = self.llm.generate(
                question=question,
                context=context,
                api_key=api_key,
                provider=provider,
                model=model,
                base_url=base_url
            )
        except LLMAPIError as e:
            # LLM 失败时返回检索到的原始内容
            self.logger.error(trace_id, "rag_service", "LLM调用失败", error=str(e))
            answer = f"AI 生成服务暂时不可用。以下是知识库中相关的内容（降级模式）：\n\n{context}"
        llm_time = time.time() - llm_start
        timing["llm_s"] = round(llm_time, 1)
        self.logger.info(trace_id, "rag_service",
                         "Step7 LLM 生成完成",
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
        上下文拼接
        """
        context_parts = []
        for i, c in enumerate(candidates, 1):
            score = c.get("rerank_score", c.get("score", 0))
            context_parts.append(
                f"【来源{i}】{c['file_name']} (相关度: {score:.2f})\n"
                f"内容: {c['content']}"
            )
        return "\n\n".join(context_parts)

    # ==================== 查询分解检索 ====================

    # 从问题中排除的停用词（不参与关键词拆分）
    _STOP_WORDS = {
        '的', '了', '在', '是', '我', '有', '和', '与', '或', '及',
        '对', '等', '从', '到', '中', '上', '下', '为', '把', '被',
        '什么', '哪些', '哪个', '怎么', '如何', '为什么', '多少',
        '吗', '呢', '吧', '啊', '呀', '请', '可以', '能', '会',
        '一个', '这个', '那个', '这些', '那些', '每个', '所有',
        '之间', '关系', '区别', '对比', '比较', '不同', '相同',
        '有没有', '是不是', '查询', '检索', '搜索', '告诉', '帮我',
        '根据', '按照', '基于', '关于', '问题', '请问', '帮忙',
        '一下', '一点', '一些', '出来', '起来', '过来',
    }

    @staticmethod
    def _extract_entities(question: str) -> List[str]:
        """
        从问题中提取关键词做分解检索 — 不限实体类型

        面试点：为什么不用 NER？
        答：企业场景跨实体查询不止人名 —
           "FastAPI 和 Flask" = 技术对比
           "Redis 和 Memcached" = 中间件选型
           "张成都和李某某" = 人名关系
           停用词分割方案覆盖所有类型，零额外依赖。
        """
        import re

        # 1. 把连接词、疑问词、常见动词替换为空格，充当分词边界
        splits = [
            '和', '与', '或', '及', '以及', '还有', '还是',
            '对比', '比较', '区别', '不同', '相同', '相似',
            '什么', '哪些', '哪个', '怎么', '如何', '为什么', '多少',
            '之间', '关系', '联系', '关联',
            '有没有', '是不是', '能不能', '会不会', '可不可以',
            '能', '会', '可以', '有', '是', '做', '过', '用',
            '用过', '做过', '合作过', '工作过',
            '请', '请问', '帮忙', '帮我', '告诉',
            '查询', '检索', '搜索', '查找', '找',
            '根据', '按照', '基于', '关于', '的',
            '吗', '呢', '吧', '啊', '呀',
        ]

        text = question
        for w in sorted(splits, key=len, reverse=True):  # 长词优先替换
            text = text.replace(w, ' ')

        # 2. 去标点
        # 保留中英文+数字，其余变空格
        text = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', text)

        # 3. 拆分
        raw = [t.strip() for t in text.split() if t.strip()]

        # 4. 过滤
        stop = RAGService._STOP_WORDS
        terms = []
        for t in raw:
            if len(t) < 2:
                continue
            if t in stop:
                continue
            if re.match(r'^[\d\.\-+#_@]+$', t):
                continue
            terms.append(t)

        # 5. 去重保序
        seen = set()
        result = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result[:5]

    def _multi_entity_retrieve(self, question: str, terms: List[str],
                                top_k: int) -> List[Dict]:
        """
        关键词分解检索 — 通用的检索增强

        面试话术：
        「向量检索做语义粗排，关键词分解做精确补充。
          检索不足时，把问题中的关键词拆出来各自用 BM25 搜一遍。
          BM25 对专有名词/技术术语是 100% 精确匹配，不会语义漂移。
          比如'张成都' embedding 可能当普通文本编码遗漏掉，
          但 BM25 能命中文档中每个出现位置。」

        适用场景：
        - 人名关系查询："张三和李四"
        - 技术对比："FastAPI vs Flask"
        - 中间件选型："Redis 和 Kafka"
        - 项目关联："项目A 项目B 进度"
        """
        all_results: List[Dict] = []
        seen_ids: Set[str] = set()

        self.logger.info("decompose", "rag_service",
                         "关键词分解检索", terms=terms)

        for term in terms:
            try:
                results = self.es.search_keyword_only(term, top_k)
                for r in results:
                    cid = r["chunk_id"]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_results.append(r)
            except Exception as e:
                self.logger.warn("decompose", "rag_service",
                                 f"子检索失败", term=term, error=str(e))

        # 多关键词组合（同时出现更相关）
        if len(terms) >= 2:
            combo = " ".join(terms[:3])
            try:
                results = self.es.search_keyword_only(combo, top_k * 2)
                for r in results:
                    cid = r["chunk_id"]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_results.append(r)
            except Exception:
                pass

        self.logger.info("decompose", "rag_service",
                         "关键词分解完成", terms=len(terms),
                         retrieved=len(all_results))
        return all_results
