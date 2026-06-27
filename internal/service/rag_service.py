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
from internal.service.intent_service import IntentService


class RAGService:
    """RAG 核心编排服务 - 面试点：全链路如何串联？"""

    # LLM 请求查看完整文档的标记
    _RETRIEVE_MARK = re.compile(r"\{\{retrieve_full_doc:([^}]+)\}\}")

    def __init__(self,
                 embedding_service: EmbeddingService = None,
                 es_service: ESService = None,
                 llm_service: LLMService = None,
                 rerank_service: RerankService = None):
        self.embedding = embedding_service or EmbeddingService()
        self.es = es_service or ESService()
        self.llm = llm_service or LLMService()
        self.rerank = rerank_service  # 可选
        self.intent = IntentService(llm_service=self.llm)
        self.logger = get_logger()

    def _classify_intent(self, question: str, history: List[Dict]) -> str:
        """调用 IntentService 进行意图识别"""
        intent, _ = self.intent.classify(question, history)
        return intent

    def _rewrite_query(self, question: str, intent: str, history: List[Dict]) -> Dict:
        """
        根据意图重写检索查询。
        返回 dict：{
            "original": question,
            "expanded": 指代消解后的独立问句,
            "context": 带上下文的查询,
            "intent": intent
        }
        """
        cfg = get_config().conversation
        result = {
            "original": question,
            "expanded": question,
            "context": question,
            "intent": intent,
        }

        if not cfg.enable_query_rewrite or intent == "new_topic" or not history:
            return result

        # 提取最近 assistant 答案中的实体，做指代消解
        entities = IntentService.extract_entities(question, history)
        expanded = IntentService.resolve_references(question, entities)
        result["expanded"] = expanded

        # 取最近 2 条消息作为上下文
        recent = history[-2:] if len(history) >= 2 else history
        context_text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')[:240]}"
            for m in recent
        )

        if intent == "follow_up":
            prev_user = next((m for m in reversed(recent) if m.get('role') == 'user'), None)
            prev_topic = prev_user.get('content', '') if prev_user else ''
            result["context"] = (
                f"基于之前关于「{prev_topic}」的讨论，当前追问：{expanded}\n\n"
                f"前文摘要：{context_text[:200]}"
            )
        elif intent == "summarize":
            result["context"] = f"请基于以下内容总结或对比：\n{context_text}\n\n用户要求：{question}"
        elif intent == "clarify":
            result["context"] = f"{question}（上下文：{context_text[:300]}）"

        return result

    def _build_historical_file_weights(self, history: List[Dict]) -> Dict[str, float]:
        """
        从历史 assistant 消息中构建 file_name → weight 映射。
        权重 = 引用次数 + 时间衰减（越近越高）。
        """
        file_stats: Dict[str, Dict] = {}
        if not history:
            return file_stats

        assistant_msgs = [m for m in history if m.get("role") == "assistant"]
        total = len(assistant_msgs)

        for idx, m in enumerate(assistant_msgs):
            sources = m.get("sources") or []
            rounds_ago = total - idx  # 1 = 最近一轮
            for s in sources:
                fn = s.get("file_name")
                if not fn:
                    continue
                stat = file_stats.setdefault(fn, {"count": 0, "recency": rounds_ago})
                stat["count"] += 1
                stat["recency"] = min(stat["recency"], rounds_ago)

        return file_stats

    def _boost_historical_sources(self, candidates: List[Dict],
                                  history: List[Dict],
                                  boost: float = 0.15) -> List[Dict]:
        """
        对历史来源文档的 chunks 进行加权。
        加权 = source_boost * (1 + log(引用次数)) * exp(-轮数/衰减窗口)
        """
        if not history or boost <= 0:
            return candidates

        cfg = get_config().conversation
        decay_window = max(1, cfg.source_boost_decay)
        file_weights = self._build_historical_file_weights(history)
        if not file_weights:
            return candidates

        import math
        for c in candidates:
            fn = c.get("file_name")
            if fn in file_weights:
                stat = file_weights[fn]
                count_factor = 1 + math.log1p(stat["count"])
                time_factor = math.exp(-stat["recency"] / decay_window)
                actual_boost = boost * count_factor * time_factor
                c["score"] = c.get("score", 0) + actual_boost
                c["source_boost"] = round(actual_boost, 4)

        # 重新按分数排序
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return candidates

    def _compress_history(self, history: List[Dict], threshold: int = 3) -> List[Dict]:
        """
        压缩历史：保留最近 2 轮完整内容，更早的历史生成摘要。
        当前使用规则摘要（取 user 消息拼接），后续可升级为 LLM 摘要。
        """
        if not history or len(history) <= threshold * 2:
            return history

        # 最近 2 轮完整保留
        recent = history[-4:]
        older = history[:-4]

        # 规则摘要：提取更早历史中的 user 问题
        older_questions = [m.get("content", "")[:80]
                           for m in older if m.get("role") == "user"]
        summary = "；".join(older_questions)
        if len(summary) > 400:
            summary = summary[:400] + "..."

        compressed = [{"role": "system", "content": f"更早的对话摘要：{summary}"}]
        compressed.extend(recent)
        return compressed

    def query(self, question: str, api_key: str,
              provider: str = "openai", model: str = None,
              base_url: str = None, top_k: int = 5,
              history: List[Dict] = None,
              retrieve_full_doc: bool = False) -> Dict:
        """
        RAG 查询主流程 - 面试点：全链路耗时分布
        """
        trace_id = str(uuid.uuid4())[:8]
        timing = {}
        cfg = get_config().conversation
        trace = {
            "trace_id": trace_id,
            "intent": None,
            "original_question": question,
            "expanded_question": question,
            "context_question": question,
            "rewrite_method": "none",
            "candidates_before_boost": 0,
            "candidates_after_boost": 0,
            "source_boosts": {},
            "full_doc_requested": False,
            "error": None,
        }

        try:
            # 意图识别 + 查询重写
            intent = self._classify_intent(question, history)
            rewritten = self._rewrite_query(question, intent, history)
            trace["intent"] = intent
            trace["expanded_question"] = rewritten.get("expanded", question)
            trace["context_question"] = rewritten.get("context", question)
            trace["rewrite_method"] = "intent+entity" if rewritten.get("expanded") != question else "none"

            # 同时用 expanded 和 context 做向量检索，然后融合
            rewritten_question = rewritten.get("context", question)
            expanded_question = rewritten.get("expanded", question)
            compressed_history = self._compress_history(history, threshold=cfg.summary_threshold)

            self.logger.info(trace_id, "rag_service",
                             "意图识别完成",
                             intent=intent,
                             original=question[:80],
                             expanded=expanded_question[:120],
                             context=rewritten_question[:120])

            # Step 1: 查询向量化（使用重写后的问题）
            embed_start = time.time()
            try:
                query_vector = self.embedding.encode_query(rewritten_question)
            except ModelLoadError:
                self.logger.warn(trace_id, "rag_service",
                                 "Embedding加载失败，降级到 BM25 检索")
                return self._query_bm25_fallback(question, api_key, provider, model,
                                                 top_k, trace_id, history=history,
                                                 retrieve_full_doc=retrieve_full_doc)
            embed_time = (time.time() - embed_start) * 1000
            timing["embedding_ms"] = round(embed_time, 1)

            # Step 2: ES 混合检索（主查询：context 版本）
            search_start = time.time()
            try:
                candidates = self.es.search_hybrid(
                    query_vector, query_text=rewritten_question,
                    top_k=top_k * 2,
                    min_score=get_config().retrieval.min_score
                )

                # 若 expanded 与 context 不同，再做一次检索并融合
                if expanded_question != rewritten_question:
                    vec2 = self.embedding.encode_query(expanded_question)
                    expanded_candidates = self.es.search_hybrid(
                        vec2, query_text=expanded_question,
                        top_k=top_k * 2,
                        min_score=get_config().retrieval.min_score
                    )
                    candidates = self._merge_candidates(candidates, expanded_candidates)

            except ESConnectionError as e:
                self.logger.error(trace_id, "rag_service", "ES 连接失败", error=str(e))
                trace["error"] = "search_unavailable"
                return self._build_error_response(
                    trace_id, timing, trace,
                    "检索服务暂时不可用，请稍后重试。",
                    suggestion="请检查 Elasticsearch 连接（ES_HOST 环境变量或 config.yaml）是否正确。"
                )
            search_time = (time.time() - search_start) * 1000
            timing["search_ms"] = round(search_time, 1)

            # Step 3: 可选重排
            if self.rerank and len(candidates) > top_k:
                candidates = self.rerank.rerank(expanded_question, candidates, top_k=top_k)
            else:
                candidates = candidates[:top_k]

            # Step 4: 空检索 / 低质量结果 → 尝试多实体分解检索
            if not candidates or len(candidates) < 3:
                entities = self._extract_entities(expanded_question)
                if len(entities) >= 2:
                    self.logger.info(trace_id, "rag_service",
                                     "向量检索结果不足，启用多实体分解检索",
                                     entities=entities, original_count=len(candidates))
                    decomposed = self._multi_entity_retrieve(expanded_question, entities, top_k)
                    if decomposed:
                        seen = {c["chunk_id"] for c in candidates}
                        for c in decomposed:
                            if c["chunk_id"] not in seen:
                                candidates.append(c)
                                seen.add(c["chunk_id"])
                        timing["decomposed"] = True
                        timing["decomposed_count"] = len(decomposed)

            # Step 5: 检索仍为空时的兜底 + 引导
            if not candidates:
                trace["error"] = "empty_retrieval"
                return self._build_error_response(
                    trace_id, timing, trace,
                    "抱歉，知识库中未找到与您问题相关的信息。",
                    suggestion="建议：1) 尝试用不同关键词提问；2) 上传包含相关内容的文档；3) 如果是人名/项目关系查询，可分别搜索每个实体。",
                    empty_retrieval=True
                )

            # Step 6: 用户手动触发整篇文档召回
            if retrieve_full_doc and get_config().retrieval.enable_full_document:
                full_docs = self._retrieve_full_documents(
                    list({c["file_name"] for c in candidates})
                )
                if full_docs:
                    candidates = full_docs

            # Step 7: 来源复用加权（放在重排后，避免干扰重排）
            trace["candidates_before_boost"] = len(candidates)
            candidates = self._boost_historical_sources(
                candidates, history, boost=cfg.source_boost
            )
            trace["candidates_after_boost"] = len(candidates)
            trace["source_boosts"] = {
                c.get("file_name"): c.get("source_boost", 0)
                for c in candidates if c.get("source_boost")
            }

            # summarize 意图扩大最终返回数量
            if intent == "summarize":
                top_k = max(top_k, 10)

            candidates = candidates[:top_k]

            # Step 8: 拼接上下文
            context = self._build_context(candidates)

            # Step 9: 调用 LLM 生成
            llm_start = time.time()
            try:
                answer = self.llm.generate(
                    question=question,
                    context=context,
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    base_url=base_url,
                    history=compressed_history,
                    allow_full_doc_retrieval=(not retrieve_full_doc)
                )
            except LLMAPIError as e:
                self.logger.error(trace_id, "rag_service", "LLM调用失败", error=str(e))
                trace["error"] = "llm_error"
                return self._build_error_response(
                    trace_id, timing, trace,
                    f"AI 生成服务暂时不可用：{str(e)}",
                    suggestion="请检查 API Key 是否有效、模型名称是否正确，或稍后重试。",
                    fallback_context=context
                )
            llm_time = time.time() - llm_start
            timing["llm_s"] = round(llm_time, 1)

            # Step 10: 检测 LLM 是否请求查看完整文档
            requested_file = self._need_full_document(answer)
            if requested_file and not retrieve_full_doc and get_config().retrieval.enable_full_document:
                self.logger.info(trace_id, "rag_service",
                                 "LLM 请求召回整篇文档", file_name=requested_file)
                full_docs = self._retrieve_full_documents([requested_file])
                if full_docs:
                    full_context = self._build_context(full_docs)
                    try:
                        answer = self.llm.generate(
                            question=question,
                            context=full_context,
                            api_key=api_key,
                            provider=provider,
                            model=model,
                            base_url=base_url,
                            history=compressed_history,
                            allow_full_doc_retrieval=False
                        )
                        candidates = full_docs
                        timing["full_doc_retrieval"] = True
                        trace["full_doc_requested"] = True
                    except LLMAPIError as e:
                        self.logger.error(trace_id, "rag_service",
                                          "整篇文档重新生成失败", error=str(e))

            # 构造返回
            return self._build_success_response(
                question, answer, candidates, trace_id, timing, trace
            )

        except Exception as e:
            self.logger.error(trace_id, "rag_service", "查询未捕获异常", error=str(e))
            trace["error"] = "internal_error"
            return self._build_error_response(
                trace_id, timing, trace,
                f"系统异常：{str(e)}",
                suggestion="请稍后重试，或查看后端日志排查问题。"
            )

    def _build_success_response(self, question: str, answer: str,
                                candidates: List[Dict], trace_id: str,
                                timing: Dict, trace: Dict) -> Dict:
        """构造成功响应"""
        total_time_ms = sum(v for k, v in timing.items() if isinstance(v, (int, float)))
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
                    "score": round(c.get("rerank_score", c.get("score", 0)), 3),
                    "section_title": c.get("section_title", ""),
                    "chunk_index": c.get("chunk_index", 0),
                    "is_full_doc": c.get("is_full_doc", False),
                }
                for c in candidates
            ],
            "trace_id": trace_id,
            "timing": timing,
            "trace": trace,
        }

    def _build_error_response(self, trace_id: str, timing: Dict, trace: Dict,
                              answer: str, suggestion: str = "",
                              empty_retrieval: bool = False,
                              fallback_context: str = "") -> Dict:
        """构造带建议的错误/兜底响应"""
        result = {
            "answer": answer,
            "sources": [],
            "trace_id": trace_id,
            "timing": timing,
            "trace": trace,
            "suggestion": suggestion,
            "empty_retrieval": empty_retrieval,
        }
        if fallback_context:
            result["fallback_context"] = fallback_context
        return result

    def _merge_candidates(self, c1: List[Dict], c2: List[Dict]) -> List[Dict]:
        """合并两路候选结果，去重并按分数排序"""
        seen = {}
        for c in c1 + c2:
            cid = c.get("chunk_id")
            if not cid:
                continue
            if cid in seen:
                seen[cid]["score"] = max(seen[cid].get("score", 0), c.get("score", 0))
            else:
                seen[cid] = dict(c)
        return sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)

    async def query_stream(self, question: str, api_key: str,
                           provider: str = "openai", model: str = None,
                           base_url: str = None, top_k: int = 5,
                           history: List[Dict] = None,
                           retrieve_full_doc: bool = False):
        """
        流式 RAG 查询 - 前 7 步同步执行，第 8 步流式生成 LLM token。
        yield SSE 事件字典：{"type": "token"/"done"/"error", ...}
        """
        trace_id = str(uuid.uuid4())[:8]
        timing = {}
        cfg = get_config().conversation
        trace = {
            "trace_id": trace_id,
            "intent": None,
            "original_question": question,
            "expanded_question": question,
            "context_question": question,
            "rewrite_method": "none",
            "candidates_before_boost": 0,
            "candidates_after_boost": 0,
            "source_boosts": {},
            "full_doc_requested": False,
            "error": None,
        }

        try:
            # 意图识别 + 查询重写 + 历史压缩
            intent = self._classify_intent(question, history)
            rewritten = self._rewrite_query(question, intent, history)
            trace["intent"] = intent
            trace["expanded_question"] = rewritten.get("expanded", question)
            trace["context_question"] = rewritten.get("context", question)
            trace["rewrite_method"] = "intent+entity" if rewritten.get("expanded") != question else "none"

            rewritten_question = rewritten.get("context", question)
            expanded_question = rewritten.get("expanded", question)
            compressed_history = self._compress_history(history, threshold=cfg.summary_threshold)

            self.logger.info(trace_id, "rag_service",
                             "流式查询意图识别完成",
                             intent=intent,
                             original=question[:80],
                             expanded=expanded_question[:120],
                             context=rewritten_question[:120])

            # Step 1: 查询向量化（使用重写后的问题）
            embed_start = time.time()
            try:
                query_vector = self.embedding.encode_query(rewritten_question)
            except ModelLoadError:
                self.logger.warn(trace_id, "rag_service",
                                 "Embedding加载失败，流式查询终止")
                yield {"type": "error", "message": "Embedding 加载失败，无法流式生成", "trace_id": trace_id}
                return
            embed_time = (time.time() - embed_start) * 1000
            timing["embedding_ms"] = round(embed_time, 1)

            # Step 2: ES 混合检索（主查询：context 版本）
            search_start = time.time()
            try:
                candidates = self.es.search_hybrid(
                    query_vector, query_text=rewritten_question,
                    top_k=top_k * 2,
                    min_score=get_config().retrieval.min_score
                )

                # 若 expanded 与 context 不同，再做一次检索并融合
                if expanded_question != rewritten_question:
                    vec2 = self.embedding.encode_query(expanded_question)
                    expanded_candidates = self.es.search_hybrid(
                        vec2, query_text=expanded_question,
                        top_k=top_k * 2,
                        min_score=get_config().retrieval.min_score
                    )
                    candidates = self._merge_candidates(candidates, expanded_candidates)
            except ESConnectionError as e:
                self.logger.error(trace_id, "rag_service", "ES 连接失败", error=str(e))
                trace["error"] = "search_unavailable"
                yield {"type": "error", "message": "检索服务暂时不可用，请稍后重试", "suggestion": "请检查 Elasticsearch 连接（ES_HOST）是否正确。", "trace_id": trace_id}
                return
            search_time = (time.time() - search_start) * 1000
            timing["search_ms"] = round(search_time, 1)

            # Step 3: 可选重排
            if self.rerank and len(candidates) > top_k:
                candidates = self.rerank.rerank(expanded_question, candidates, top_k=top_k)
            else:
                candidates = candidates[:top_k]

            # Step 4: 多实体分解检索
            if not candidates or len(candidates) < 3:
                entities = self._extract_entities(expanded_question)
                if len(entities) >= 2:
                    decomposed = self._multi_entity_retrieve(expanded_question, entities, top_k)
                    if decomposed:
                        seen = {c["chunk_id"] for c in candidates}
                        for c in decomposed:
                            if c["chunk_id"] not in seen:
                                candidates.append(c)
                                seen.add(c["chunk_id"])
                        timing["decomposed"] = True
                        timing["decomposed_count"] = len(decomposed)

            # Step 5: 空检索兜底 + 引导
            if not candidates:
                trace["error"] = "empty_retrieval"
                yield {
                    "type": "error",
                    "message": "知识库中未找到与您问题相关的信息",
                    "suggestion": "建议：1) 尝试用不同关键词提问；2) 上传包含相关内容的文档；3) 如果是人名/项目关系查询，可分别搜索每个实体。",
                    "empty_retrieval": True,
                    "trace_id": trace_id
                }
                return

            # Step 6: 用户手动触发整篇文档召回
            if retrieve_full_doc and get_config().retrieval.enable_full_document:
                full_docs = self._retrieve_full_documents(
                    list({c["file_name"] for c in candidates})
                )
                if full_docs:
                    candidates = full_docs

            # Step 7: 来源复用加权
            trace["candidates_before_boost"] = len(candidates)
            candidates = self._boost_historical_sources(
                candidates, history, boost=cfg.source_boost
            )
            trace["candidates_after_boost"] = len(candidates)
            trace["source_boosts"] = {
                c.get("file_name"): c.get("source_boost", 0)
                for c in candidates if c.get("source_boost")
            }

            # summarize 意图扩大最终返回数量
            if intent == "summarize":
                top_k = max(top_k, 10)

            candidates = candidates[:top_k]

            # Step 8: 拼接上下文
            context = self._build_context(candidates)

            # Step 9: 流式生成
            llm_start = time.time()
            answer_parts = []
            try:
                async for token in self.llm.generate_stream(
                    question=question,
                    context=context,
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    base_url=base_url,
                    history=compressed_history,
                    allow_full_doc_retrieval=(not retrieve_full_doc)
                ):
                    answer_parts.append(token)
                    yield {"type": "token", "content": token}
            except LLMAPIError as e:
                self.logger.error(trace_id, "rag_service", "LLM流式调用失败", error=str(e))
                yield {"type": "error", "message": f"LLM 调用失败: {str(e)}", "suggestion": "请检查 API Key 是否有效、模型名称是否正确，或稍后重试。", "trace_id": trace_id}
                return
            llm_time = time.time() - llm_start
            timing["llm_s"] = round(llm_time, 1)

            answer = "".join(answer_parts)

            # Step 10: 检测 LLM 是否请求查看完整文档
            requested_file = self._need_full_document(answer)
            if requested_file and not retrieve_full_doc and get_config().retrieval.enable_full_document:
                yield {"type": "token", "content": "\n\n[正在召回完整文档...]\n"}
                full_docs = self._retrieve_full_documents([requested_file])
                if full_docs:
                    full_context = self._build_context(full_docs)
                    answer_parts.clear()
                    try:
                        async for token in self.llm.generate_stream(
                            question=question,
                            context=full_context,
                            api_key=api_key,
                            provider=provider,
                            model=model,
                            base_url=base_url,
                            history=compressed_history,
                            allow_full_doc_retrieval=False
                        ):
                            answer_parts.append(token)
                            yield {"type": "token", "content": token}
                        answer = "".join(answer_parts)
                        candidates = full_docs
                        timing["full_doc_retrieval"] = True
                        trace["full_doc_requested"] = True
                    except LLMAPIError as e:
                        self.logger.error(trace_id, "rag_service",
                                          "整篇文档流式重新生成失败", error=str(e))
                        yield {"type": "error", "message": f"完整文档重新生成失败: {str(e)}", "trace_id": trace_id}

            yield {
                "type": "done",
                "answer": answer,
                "sources": [
                    {
                        "content": c.get("content", ""),
                        "file_name": c.get("file_name", ""),
                        "score": round(c.get("rerank_score", c.get("score", 0)), 3),
                        "section_title": c.get("section_title", ""),
                        "chunk_index": c.get("chunk_index", 0),
                        "is_full_doc": c.get("is_full_doc", False),
                    }
                    for c in candidates
                ],
                "trace_id": trace_id,
                "timing": timing,
                "trace": trace,
            }

        except Exception as e:
            self.logger.error(trace_id, "rag_service", "流式查询未捕获异常", error=str(e))
            trace["error"] = "internal_error"
            yield {"type": "error", "message": f"系统异常：{str(e)}", "suggestion": "请稍后重试，或查看后端日志排查问题。", "trace_id": trace_id}

    def _query_bm25_fallback(self, question: str, api_key: str,
                             provider: str, model: str,
                             top_k: int, trace_id: str,
                             history: List[Dict] = None,
                             retrieve_full_doc: bool = False) -> Dict:
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
                "error": "search_unavailable",
                "suggestion": "请检查 Elasticsearch 连接（ES_HOST）是否正确。"
            }

        if not results:
            return {
                "answer": "抱歉，知识库中未找到相关内容。",
                "sources": [],
                "trace_id": trace_id,
                "fallback": True,
                "empty_retrieval": True,
                "suggestion": "建议：1) 尝试用不同关键词提问；2) 上传包含相关内容的文档。"
            }

        if retrieve_full_doc:
            full_docs = self._retrieve_full_documents(
                list({r["file_name"] for r in results})
            )
            if full_docs:
                results = full_docs

        # 降级模式下也尝试调用 LLM（如果有 API Key）
        context = self._build_context(results)
        try:
            answer = self.llm.generate(question, context, api_key, provider, model,
                                       history=history)
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
            label = "【完整文档】" if c.get("is_full_doc") else f"【来源{i}】"
            section = c.get("section_title", "")
            section_hint = f" (章节: {section})" if section else ""
            context_parts.append(
                f"{label}{c['file_name']}{section_hint} (相关度: {score:.2f})\n"
                f"内容: {c['content']}"
            )
        return "\n\n".join(context_parts)

    def _need_full_document(self, answer: str) -> Optional[str]:
        """检测 LLM 是否请求查看完整文档"""
        m = self._RETRIEVE_MARK.search(answer)
        if m:
            return m.group(1).strip()
        return None

    def _retrieve_full_documents(self, file_names: List[str]) -> List[Dict]:
        """按 file_name 召回整篇文档"""
        results = []
        seen = set()
        for fn in file_names:
            if fn in seen:
                continue
            doc = self.es.retrieve_full_document(fn)
            if doc:
                seen.add(fn)
                results.append(doc)
        return results

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
