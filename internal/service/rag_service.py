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

    def _classify_intent(self, question: str, history: List[Dict],
                         llm_kwargs: Dict = None) -> str:
        """调用 IntentService 进行意图识别（规则优先，可选 LLM 兜底）"""
        intent, _ = self.intent.classify(question, history, llm_kwargs=llm_kwargs)
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
        面试点：旧版只保留 user 问句，丢失 assistant 答案导致长对话失忆；
            改进后把更早的「问→答」配对压缩成「用户问了X，助手答了Y要点」，
            保留关键结论，让 LLM 在长对话中仍有记忆。
        """
        if not history or len(history) <= threshold * 2:
            return history

        # 最近 2 轮完整保留（一轮 = user + assistant）
        recent = history[-4:]
        older = history[:-4]

        # 把 older 按 user/assistant 配对，压缩成摘要要点
        summary_parts = []
        pending_q = None
        for m in older:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role == "user" and content:
                pending_q = content[:80]
            elif role == "assistant" and content:
                ans_brief = content[:120].replace("\n", " ")
                if pending_q:
                    summary_parts.append(f"用户问「{pending_q}」，助手答「{ans_brief}…」")
                else:
                    summary_parts.append(f"助手答「{ans_brief}…」")
                pending_q = None
        # 残留未配对的 user 问题也保留
        if pending_q:
            summary_parts.append(f"用户问「{pending_q}」")

        summary = "；".join(summary_parts)
        if len(summary) > 600:
            summary = summary[:600] + "..."

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
            intent = self._classify_intent(question, history, llm_kwargs={
                "api_key": api_key, "provider": provider, "model": model, "base_url": base_url
            })
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
                # 面试点：追问向量去污染
                # follow_up 时主向量用 expanded（纯实体消解，如「张成都的项目」），
                # 而非 context（「基于之前关于…的讨论」）——后者混入对话套话会
                # 把语义空间拉向"讨论/上下文"这类词，降低对原文档的召回精度。
                # context 仅用于 BM25（关键词检索对套话不敏感，反而能借上下文多命中实体）。
                use_expanded_for_vector = (
                    cfg.follow_up_vector_use_expanded
                    and intent in ("follow_up", "clarify")
                    and expanded_question != rewritten_question
                )
                vector_question = expanded_question if use_expanded_for_vector else rewritten_question
                query_vector = self.embedding.encode_query(vector_question)
            except ModelLoadError:
                self.logger.warn(trace_id, "rag_service",
                                 "Embedding加载失败，降级到 BM25 检索")
                return self._query_bm25_fallback(question, api_key, provider, model,
                                                 top_k, trace_id, history=history,
                                                 retrieve_full_doc=retrieve_full_doc)
            embed_time = (time.time() - embed_start) * 1000
            timing["embedding_ms"] = round(embed_time, 1)

            # Step 2: ES 混合检索
            search_start = time.time()
            try:
                # 主检索：向量用去污染的 vector_question；BM25 用 context（带对话上下文，多命中实体）
                bm25_question = rewritten_question if use_expanded_for_vector else vector_question
                candidates = self.es.search_hybrid(
                    query_vector, query_text=bm25_question,
                    top_k=top_k * 2,
                    min_score=get_config().retrieval.min_score
                )

                # expanded 与 context 不同时，补一次 expanded 的 BM25 + 向量融合
                # （多路召回：context 借上下文，expanded 借精确实体，RRF 融合取并集）
                if expanded_question != rewritten_question and not use_expanded_for_vector:
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

            # summarize 意图扩大最终返回数量（影响 diversity 目标）
            if intent == "summarize":
                top_k = max(top_k, 10)

            # Step 3: 可选重排 —— 保留 top_k*3 候选供多样性选择，不一刀切到 top_k
            rerank_query = expanded_question if expanded_question != question else question
            rerank_keep = max(top_k * 3, top_k + 5)
            if self.rerank and len(candidates) > top_k:
                candidates = self.rerank.rerank(rerank_query, candidates, top_k=rerank_keep)
            else:
                candidates = self._sort_candidates_by_score(candidates)[:rerank_keep]

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

            # Step 4.5: 实体文档扩展（单实体也触发）
            # 解决「张成都是谁」只召回一篇论文：对核心实体做 BM25，
            # 把包含实体但未进候选池的文档代表 chunk 注入，并标记实体命中文档
            entity_terms = self._extract_entity_terms(expanded_question)
            if entity_terms:
                entity_reps = self._expand_entity_documents(
                    entity_terms, candidates, top_k)
                if entity_reps:
                    for c in entity_reps:
                        candidates.append(c)
                    timing["entity_doc_expanded"] = len(entity_reps)
                    trace["entity_terms"] = entity_terms
                # 标记候选池里实体命中的文档，多样性选择时优先覆盖
                marked = self._mark_entity_matched_chunks(candidates, entity_terms)
                if marked:
                    trace["entity_matched_chunks"] = marked

            # Step 5: 文档多样性选择 → primary（回答「其他文档为什么不能一次引用」）
            # 单文档 cap + MMR 贪心，确保 top_k 覆盖多文档
            if get_config().retrieval.enable_doc_diversity:
                primary = self._diversify_by_document(
                    candidates, top_k, get_config().retrieval.max_chunks_per_doc)
            else:
                primary = self._sort_candidates_by_score(candidates)[:top_k]
            trace["primary_count"] = len(primary)
            trace["primary_docs"] = list({c.get("file_name") for c in primary if c.get("file_name")})

            # Step 6: 检索仍为空时的兜底 + 引导
            if not primary:
                trace["error"] = "empty_retrieval"
                return self._build_error_response(
                    trace_id, timing, trace,
                    "抱歉，知识库中未找到与您问题相关的信息。",
                    suggestion=self._build_empty_retrieval_suggestion(),
                    empty_retrieval=True
                )

            # Step 7: 来源复用加权（在 primary 上，避免邻居分数干扰）
            trace["candidates_before_boost"] = len(primary)
            primary = self._boost_historical_sources(
                primary, history, boost=cfg.source_boost
            )
            trace["candidates_after_boost"] = len(primary)
            trace["source_boosts"] = {
                c.get("file_name"): c.get("source_boost", 0)
                for c in primary if c.get("source_boost")
            }
            primary = primary[:top_k]

            # Step 8: 用户手动触发整篇文档召回（替换 primary 为完整文档）
            if retrieve_full_doc and get_config().retrieval.enable_full_document:
                full_docs = self._retrieve_full_documents(
                    list({c["file_name"] for c in primary})
                )
                if full_docs:
                    primary = full_docs

            # Step 9: 邻域上下文扩展 → context_candidates（primary + 邻居）
            # 回答「同一文档该用多少内容」：primary 命中 + 相邻块补充，不切 top_k，邻居进入上下文
            context_candidates = list(primary)
            neighbor_count = 0
            if primary and get_config().retrieval.enable_neighbor_expansion:
                # 全文召回模式下不再扩展邻居（已是完整文档）
                is_full_doc_mode = retrieve_full_doc and any(c.get("is_full_doc") for c in primary)
                if not is_full_doc_mode:
                    window = get_config().retrieval.neighbor_window
                    expanded = self.es.expand_neighbors(primary, window=window)
                    primary_ids = {c.get("chunk_id") for c in primary}
                    neighbors = [c for c in expanded
                                 if c.get("chunk_id") not in primary_ids]
                    if neighbors:
                        context_candidates = primary + neighbors
                        neighbor_count = len(neighbors)
                        timing["neighbor_expanded"] = neighbor_count

            # Step 10: 拼接上下文（带文档预算，回答「该用多少内容」）
            context = self._build_context(context_candidates)

            # Step 11: 调用 LLM 生成
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

            # 防御性清理：移除答案中可能残留的 retrieve_full_doc 标记（LLM 不再被指示输出，
            # 但作为兜底）。已移除自动重新生成逻辑——旧设计让 LLM 输出标记触发全文召回并
            # 重新生成整个答案，多文档问题会级联产生多个答案拼接，且每次耗时10s+。
            # 用户需要全文可手动点「召回完整文档」按钮（retrieve_full_doc=true 走短路返回原文）。
            answer = self._strip_retrieve_marks(answer)

            # 构造返回（sources 用 primary 多样化命中；context 已含邻居）
            return self._build_success_response(
                question, answer, primary, trace_id, timing, trace
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

    def _build_empty_retrieval_suggestion(self) -> str:
        """
        动态空检索建议：列出知识库中实际的文档名，帮用户「知道能问什么」。
        面试点：静态建议（"换个问法"）信息量低，用户不知道库里有什么；
            动态列出文档名，用户立刻能据此组织问题，体验提升明显。
        """
        base = "建议：1) 尝试用不同关键词提问；2) 上传包含相关内容的文档；3) 多实体/项目关系查询可分别搜索每个实体。"
        try:
            docs = self.es.list_file_names()
            names = [d.get("file_name", "") for d in docs if d.get("file_name")]
            if names:
                shown = "、".join(names[:5])
                more = f"等共 {len(names)} 份" if len(names) > 5 else ""
                return f"当前知识库包含：{shown}{more}。\n{base}"
        except Exception:
            pass
        return base

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

    def _diversify_by_document(self, candidates: List[Dict],
                               top_k: int, max_per_doc: int = 2) -> List[Dict]:
        """
        文档多样性选择（MMR 简化版）—— 回答「其他文档为什么不能一次引用」。
        确保 primary 命中覆盖多个文档，单一文档不超过 max_per_doc 个 chunk，
        避免一份文档（如某份简历）的多个片段垄断 top_k、挤掉其他文档。

        策略（贪心 MMR）：
        1. 候选先按分数降序（重排分数优先，否则 RRF 分数）
        2. 贪心选入：每文档计数，达 max_per_doc 则跳过该文档后续块，留位置给其他文档
        3. 第一轮若不足 top_k（候选文档太少），回退忽略 cap 补齐，保证信息量

        面试点：为什么不直接用纯 MMR（相似度惩罚）？
        答：纯 MMR 需要 chunk 间相似度矩阵，计算开销随候选数平方增长；
            per-doc cap 是 O(n) 的简化版，效果近似且对海量数据友好。
            文档级多样性比 chunk 级去重更符合 RAG「多来源佐证」的诉求。

        海量文章可扩展性：本策略 O(n)，与候选规模线性，百倍数据量也无需改算法；
            若需更强去重，可叠加 chunk 级 MMR（仅对同一文档内的 chunk 做相似度去重）。
        """
        if not candidates:
            return []
        # 按分数降序（rerank_score 优先，回退 score），entity_match/entity_chunk 优先
        ranked = self._sort_candidates_by_score(candidates)
        # entity_match 文档排前；同文档内 entity_chunk（含实体词）排前
        ranked.sort(key=lambda c: (
            not c.get("entity_match", False),
            not c.get("entity_chunk", False),
        ))

        if max_per_doc <= 0:
            return ranked[:top_k]

        # 候选数 ≤ top_k：仍按 per-doc cap 取，不足时回退补齐（保证信息量）
        # 同文档内 entity_chunk 优先（已由排序保证）
        if len(ranked) <= top_k:
            selected = []
            doc_count: Dict[str, int] = {}
            # 第一轮：per-doc cap 内取
            for c in ranked:
                if len(selected) >= top_k:
                    break
                fn = c.get("file_name", "")
                if doc_count.get(fn, 0) >= max_per_doc:
                    continue
                selected.append(c)
                doc_count[fn] = doc_count.get(fn, 0) + 1
            # 不足 top_k 回退：忽略 cap 补齐
            if len(selected) < top_k and len(selected) < len(ranked):
                sel_ids = {id(c) for c in selected}
                for c in ranked:
                    if len(selected) >= top_k:
                        break
                    if id(c) in sel_ids:
                        continue
                    selected.append(c)
            return selected

        # 2. 第一轮：文档覆盖优先 —— 每个不同文档先各取1个代表
        #    对 entity_match 文档：优先选 entity_chunk（含实体词的 chunk，如作者页 #0
        #    而非案例页 #20），确保选中的 chunk 含实体信息供 LLM 提取
        selected: List[Dict] = []
        doc_count: Dict[str, int] = {}
        # 2. 第一轮：文档覆盖优先 —— 每个不同文档先各取1个代表
        #    ranked 已按 entity_match → entity_chunk → 分数 排序，故同文档内
        #    entity_chunk（含实体词，如作者页 #0）先于非 entity_chunk（案例页 #20）被选
        selected: List[Dict] = []
        doc_count: Dict[str, int] = {}
        doc_seen: set = set()
        for c in ranked:
            if len(selected) >= top_k:
                break
            fn = c.get("file_name", "")
            if fn in doc_seen:
                continue
            selected.append(c)
            doc_count[fn] = 1
            doc_seen.add(fn)

        # 3. 第二轮：在 max_per_doc 内按分数补齐（同一文档可再取，直到 cap）
        if len(selected) < top_k:
            for c in ranked:
                if len(selected) >= top_k:
                    break
                fn = c.get("file_name", "")
                if doc_count.get(fn, 0) >= max_per_doc:
                    continue
                if id(c) in {id(s) for s in selected}:
                    continue
                selected.append(c)
                doc_count[fn] = doc_count.get(fn, 0) + 1

        # 4. 仍不足 top_k：忽略 cap 补齐（保证信息量，宁可单文档多取）
        if len(selected) < top_k:
            sel_ids = {id(c) for c in selected}
            for c in ranked:
                if len(selected) >= top_k:
                    break
                if id(c) in sel_ids:
                    continue
                selected.append(c)

        return selected

    @staticmethod
    def _sort_candidates_by_score(candidates: List[Dict]) -> List[Dict]:
        """按分数降序排序：优先 rerank_score，回退 score"""
        return sorted(candidates, key=lambda c: c.get("rerank_score", c.get("score", 0)),
                      reverse=True)

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
            intent = self._classify_intent(question, history, llm_kwargs={
                "api_key": api_key, "provider": provider, "model": model, "base_url": base_url
            })
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
                # 追问向量去污染：见 query() 同名注释
                use_expanded_for_vector = (
                    cfg.follow_up_vector_use_expanded
                    and intent in ("follow_up", "clarify")
                    and expanded_question != rewritten_question
                )
                vector_question = expanded_question if use_expanded_for_vector else rewritten_question
                query_vector = self.embedding.encode_query(vector_question)
            except ModelLoadError:
                self.logger.warn(trace_id, "rag_service",
                                 "Embedding加载失败，流式查询终止")
                yield {"type": "error", "message": "Embedding 加载失败，无法流式生成", "trace_id": trace_id}
                return
            embed_time = (time.time() - embed_start) * 1000
            timing["embedding_ms"] = round(embed_time, 1)

            # Step 2: ES 混合检索
            search_start = time.time()
            try:
                bm25_question = rewritten_question if use_expanded_for_vector else vector_question
                candidates = self.es.search_hybrid(
                    query_vector, query_text=bm25_question,
                    top_k=top_k * 2,
                    min_score=get_config().retrieval.min_score
                )

                if expanded_question != rewritten_question and not use_expanded_for_vector:
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

            # summarize 意图扩大最终返回数量（影响 diversity 目标）
            if intent == "summarize":
                top_k = max(top_k, 10)

            # Step 3: 可选重排 —— 保留 top_k*3 候选供多样性选择
            rerank_query = expanded_question if expanded_question != question else question
            rerank_keep = max(top_k * 3, top_k + 5)
            if self.rerank and len(candidates) > top_k:
                candidates = self.rerank.rerank(rerank_query, candidates, top_k=rerank_keep)
            else:
                candidates = self._sort_candidates_by_score(candidates)[:rerank_keep]

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

            # Step 4.5: 实体文档扩展（单实体也触发）
            entity_terms = self._extract_entity_terms(expanded_question)
            if entity_terms:
                entity_reps = self._expand_entity_documents(
                    entity_terms, candidates, top_k)
                if entity_reps:
                    for c in entity_reps:
                        candidates.append(c)
                    timing["entity_doc_expanded"] = len(entity_reps)
                    trace["entity_terms"] = entity_terms
                marked = self._mark_entity_matched_chunks(candidates, entity_terms)
                if marked:
                    trace["entity_matched_chunks"] = marked

            # Step 5: 文档多样性选择 → primary
            if get_config().retrieval.enable_doc_diversity:
                primary = self._diversify_by_document(
                    candidates, top_k, get_config().retrieval.max_chunks_per_doc)
            else:
                primary = self._sort_candidates_by_score(candidates)[:top_k]
            trace["primary_count"] = len(primary)
            trace["primary_docs"] = list({c.get("file_name") for c in primary if c.get("file_name")})

            # Step 6: 空检索兜底 + 引导
            if not primary:
                trace["error"] = "empty_retrieval"
                yield {
                    "type": "error",
                    "message": "知识库中未找到与您问题相关的信息",
                    "suggestion": self._build_empty_retrieval_suggestion(),
                    "empty_retrieval": True,
                    "trace_id": trace_id
                }
                return

            # Step 7: 来源复用加权（在 primary 上）
            trace["candidates_before_boost"] = len(primary)
            primary = self._boost_historical_sources(
                primary, history, boost=cfg.source_boost
            )
            trace["candidates_after_boost"] = len(primary)
            trace["source_boosts"] = {
                c.get("file_name"): c.get("source_boost", 0)
                for c in primary if c.get("source_boost")
            }
            primary = primary[:top_k]

            # Step 8: 用户手动触发整篇文档召回
            if retrieve_full_doc and get_config().retrieval.enable_full_document:
                full_docs = self._retrieve_full_documents(
                    list({c["file_name"] for c in primary})
                )
                if full_docs:
                    # 短路：直接流式返回文档原文，不调 LLM
                    # 面试点：为什么完整文档召回不调 LLM？
                    # 答：完整文档 context 可能几万字，LLM 处理超长 context 极慢（10-30s），
                    # 且用户要的是文档原文而非总结。直接分块 yield 原文，零 LLM 延迟，
                    # 前端即时显示，彻底解决「召回完整文档耗时长、前端卡死」。
                    answer_parts = []
                    for doc in full_docs:
                        fn = doc.get("file_name", "")
                        content = doc.get("content", "") or ""
                        if not content:
                            continue
                        header = f"📄 {fn}\n\n"
                        answer_parts.append(header)
                        yield {"type": "token", "content": header}
                        # 按段落分块流式输出，每块 ~200 字，即时反馈
                        chunks = self._split_for_stream(content, 200)
                        for piece in chunks:
                            answer_parts.append(piece)
                            yield {"type": "token", "content": piece}
                    answer = "".join(answer_parts)
                    timing["llm_s"] = 0.0
                    timing["full_doc_retrieval"] = True
                    trace["full_doc_requested"] = True
                    trace["full_doc_files"] = [d.get("file_name") for d in full_docs]
                    yield {
                        "type": "done",
                        "answer": answer,
                        "sources": [
                            {
                                "content": d.get("content", "")[:500],
                                "file_name": d.get("file_name", ""),
                                "score": 0,
                                "section_title": "",
                                "chunk_index": -1,
                                "is_full_doc": True,
                            }
                            for d in full_docs
                        ],
                        "trace_id": trace_id,
                        "timing": timing,
                        "trace": trace,
                    }
                    return

            # Step 9: 邻域上下文扩展 → context_candidates
            context_candidates = list(primary)
            if primary and get_config().retrieval.enable_neighbor_expansion:
                is_full_doc_mode = retrieve_full_doc and any(c.get("is_full_doc") for c in primary)
                if not is_full_doc_mode:
                    window = get_config().retrieval.neighbor_window
                    expanded = self.es.expand_neighbors(primary, window=window)
                    primary_ids = {c.get("chunk_id") for c in primary}
                    neighbors = [c for c in expanded
                                 if c.get("chunk_id") not in primary_ids]
                    if neighbors:
                        context_candidates = primary + neighbors
                        timing["neighbor_expanded"] = len(neighbors)

            # Step 10: 拼接上下文（带文档预算）
            context = self._build_context(context_candidates)

            # Step 11: 流式生成（同时过滤 retrieve_full_doc 标记，避免泄露给用户）
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
                    # 过滤掉 retrieve_full_doc 标记：把标记拆开，按字符逐个输出
                    # 如果 token 包含标记，把标记部分替换为空再输出
                    cleaned_token = self._filter_stream_marks(token)
                    if cleaned_token:  # 跳过空 token（标记整段被吃掉）
                        answer_parts.append(cleaned_token)
                        yield {"type": "token", "content": cleaned_token}
            except LLMAPIError as e:
                self.logger.error(trace_id, "rag_service", "LLM流式调用失败", error=str(e))
                yield {"type": "error", "message": f"LLM 调用失败: {str(e)}", "suggestion": "请检查 API Key 是否有效、模型名称是否正确，或稍后重试。", "trace_id": trace_id}
                return
            llm_time = time.time() - llm_start
            timing["llm_s"] = round(llm_time, 1)

            # 释放流式标记缓存残留（未闭合 buffer，已确认非合法标记），并入答案
            leftover = self._flush_stream_mark_buf()
            if leftover:
                answer_parts.append(leftover)
                yield {"type": "token", "content": leftover}

            # 兜底清理（filter 已处理，防御性双重保险）
            answer = "".join(answer_parts)
            answer = self._strip_retrieve_marks(answer)

            # 已移除自动全文召回重新生成（旧设计让 LLM 输出标记触发重新生成整个答案，
            # 多文档问题级联产生多个答案拼接 + 每次耗时10s+）。用户需要全文可手动点
            # 「召回完整文档」按钮（retrieve_full_doc=true 走短路返回文档原文，不调 LLM）。

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
                    for c in primary
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
        上下文拼接 - 面试点：
        - 邻域扩展块（score≈0 且非 full_doc）标注「上下文补充」，提示 LLM 这是相邻片段
        - 同文件的邻域块紧跟主命中块后排序，便于 LLM 衔接理解跨块语义
        - 内容预算（context_char_budget）：按文档分配字符额度，单文档不垄断上下文窗口
          回答「同一文档多次出现该用多少内容」——每文档给到额度上限，超出截断并提示
        """
        if not candidates:
            return ""

        budget = max(1000, get_config().retrieval.context_char_budget)
        # 排序：主命中（有分）优先，邻域块（零分）按 file_name + chunk_index 紧随其后
        def sort_key(c):
            fn = c.get("file_name", "")
            ci = c.get("chunk_index", 0) or 0
            is_neighbor = (c.get("score", 0) == 0.0 and not c.get("is_full_doc"))
            return (is_neighbor, fn, ci)

        ordered = sorted(candidates, key=sort_key)

        # 按文档聚合，分配预算：primary 文档数 + 每文档额度
        doc_names = []
        for c in ordered:
            fn = c.get("file_name", "")
            if fn and fn not in doc_names:
                doc_names.append(fn)
        per_doc_budget = budget // max(1, len(doc_names))

        context_parts = []
        doc_used: Dict[str, int] = {}
        global_used = 0
        src_idx = 0
        nbr_idx = 0
        truncated_docs: List[str] = []

        for c in ordered:
            fn = c.get("file_name", "")
            content = c.get("content", "") or ""
            score = c.get("rerank_score", c.get("score", 0))
            is_neighbor = (c.get("score", 0) == 0.0 and not c.get("is_full_doc")
                           and "rerank_score" not in c)
            if c.get("is_full_doc"):
                label = "【完整文档】"
                idx = ""
            elif is_neighbor:
                nbr_idx += 1
                label = f"【上下文补充{nbr_idx}】"
                idx = ""
            else:
                src_idx += 1
                label = f"【来源{src_idx}】"
                idx = ""
            section = c.get("section_title", "")
            section_hint = f" (章节: {section})" if section else ""
            ci = c.get("chunk_index", 0)
            ci_hint = f" · 片段#{ci}" if ci is not None and not c.get("is_full_doc") else ""
            header = f"{label}{fn}{section_hint}{ci_hint} (相关度: {score:.2f})\n内容: "

            # 完整文档不参与 per-doc 截断（用户主动召回全文）
            if c.get("is_full_doc"):
                context_parts.append(header + content)
                global_used += len(content)
                continue

            # 按文档预算截断
            used = doc_used.get(fn, 0)
            remaining_doc = per_doc_budget - used
            if remaining_doc <= 0:
                if fn not in truncated_docs:
                    truncated_docs.append(fn)
                continue
            # 头部开销 + 内容
            piece = content[:remaining_doc]
            if len(content) > remaining_doc:
                piece = piece + "…[内容已按文档预算截断]"
                if fn not in truncated_docs:
                    truncated_docs.append(fn)
            context_parts.append(header + piece)
            doc_used[fn] = used + len(piece) + len(header)
            global_used += len(piece) + len(header)

        result = "\n\n".join(context_parts)
        if truncated_docs:
            result += f"\n\n[注：以下文档内容超出预算已截断：{', '.join(truncated_docs)}]"
        return result

    def _need_full_document(self, answer: str) -> Optional[str]:
        """检测 LLM 是否请求查看完整文档（取第一个匹配）"""
        m = self._RETRIEVE_MARK.search(answer)
        if m:
            return m.group(1).strip()
        return None

    def _need_full_documents(self, answer: str) -> List[str]:
        """检测 LLM 请求的所有完整文档（去重保序）"""
        seen = set()
        result = []
        for m in self._RETRIEVE_MARK.finditer(answer):
            fn = m.group(1).strip()
            if fn and fn not in seen:
                seen.add(fn)
                result.append(fn)
        return result

    def _strip_retrieve_marks(self, text: str) -> str:
        """从答案中移除所有 retrieve_full_doc 标记（防止泄露给用户）"""
        if not text:
            return text
        return self._RETRIEVE_MARK.sub("", text).strip()

    def _filter_stream_marks(self, token: str) -> str:
        """
        流式标记过滤：处理跨 token 被拆开的 {{retrieve_full_doc:xxx}} 标记。
        状态机：遇到未闭合的 {{ 开头时缓存到 buffer，等后续 token 拼齐判断。
        - 拼出完整标记 → 丢弃（不泄露给用户）
        - 拼到一定长度仍无 }} 且不像标记 → 释放 buffer（避免无限缓存正常文本）
        面试点：为什么流式要单独处理？
        答：LLM 的 {{retrieve_full_doc:a.pdf}} 可能被拆成多个 token 到达
           （如 "{{retrieve" + "_full_doc:a.pdf}}"），逐 token 用完整正则无法匹配，
           导致标记片段泄露给用户。状态机缓存未闭合的 {{，拼齐后再判断。
        """
        if not hasattr(self, '_stream_mark_buf'):
            self._stream_mark_buf = ''

        buf = self._stream_mark_buf + token
        result_parts = []
        last_end = 0
        # 用完整正则找出所有已闭合的标记，丢弃它们，保留标记之间的文本
        for m in self._RETRIEVE_MARK.finditer(buf):
            result_parts.append(buf[last_end:m.start()])
            last_end = m.end()
        tail = buf[last_end:]  # 最后一个标记之后的剩余

        # 检查 tail 是否含未闭合的 {{（标记可能跨 token）
        mark_open = tail.rfind('{{')
        if mark_open >= 0:
            after_open = tail[mark_open:]
            # 若 {{ 后已有 }} 但正则没匹配（格式不合法）→ 不是标记，释放全部
            if '}}' in after_open:
                result_parts.append(tail)
                self._stream_mark_buf = ''
            else:
                # 有 {{ 无 }}：判断 {{ 后是否像标记前缀
                after_brace = after_open[2:].lstrip()
                looks_like_mark = (after_brace.startswith('retrieve')
                                   or after_open == '{{')
                if looks_like_mark and len(after_open) <= 80:
                    # 缓存未闭合标记部分，等下一 token
                    result_parts.append(tail[:mark_open])
                    self._stream_mark_buf = after_open
                else:
                    # 不像标记前缀，或超长 → 普通 {{ 文本，释放
                    result_parts.append(tail)
                    self._stream_mark_buf = ''
        else:
            result_parts.append(tail)
            self._stream_mark_buf = ''

        return ''.join(result_parts)

    def _flush_stream_mark_buf(self) -> str:
        """
        流式结束时处理缓存中残留的未闭合 buffer。
        - 若 buffer 是未闭合的标记前缀（以 {{retrieve 开头）→ 丢弃（不泄露给用户）
        - 否则（普通 {{ 文本）→ 释放作为正常文本
        """
        buf = getattr(self, '_stream_mark_buf', '')
        self._stream_mark_buf = ''
        if not buf:
            return ''
        # 未闭合的标记前缀（如 "{{retrieve_full_doc"）→ 丢弃
        stripped = buf[2:].lstrip() if buf.startswith('{{') else ''
        if stripped.startswith('retrieve') or buf == '{{':
            return ''  # 丢弃，不泄露标记片段
        return buf  # 普通文本，释放

    def _split_for_stream(self, text: str, size: int = 200) -> List[str]:
        """
        把长文本按段落/指定长度切分，用于流式分块输出。
        优先在段落/句子边界切，避免切断词。
        """
        if not text:
            return []
        if len(text) <= size:
            return [text]
        parts = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            if end < n:
                # 在 [start, end] 找最近的换行/句号切分
                snippet = text[start:end]
                for br in ['\n\n', '\n', '。', '；', '！', '？', ' ']:
                    pos = snippet.rfind(br)
                    if pos > size // 2:
                        end = start + pos + len(br)
                        break
            parts.append(text[start:end])
            start = end
        return parts

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

    # 非实体词：实体文档扩展时排除普通名词/动词/泛称，避免对"项目""总结"做无意义 BM25
    # 仅保留专有名词（人名、技术名、产品名、机构名等）
    _NON_ENTITY_WORDS = {
        '项目', '经验', '能力', '技术', '内容', '信息', '问题', '情况',
        '方面', '方法', '方式', '结果', '结论', '建议', '方案', '策略',
        '特点', '特征', '优势', '劣势', '原理', '流程', '步骤', '阶段',
        '总结', '概括', '归纳', '分析', '介绍', '说明', '解释', '展开',
        '详情', '详细', '概况', '简介', '背景', '现状', '发展', '趋势',
        '区别', '不同', '相同', '相似', '关联', '联系', '影响', '作用',
        '价值', '意义', '目的', '目标', '需求', '场景', '应用', '实现',
        '设计', '架构', '结构', '组成', '模块', '功能', '性能', '指标',
        '评估', '评价', '测试', '验证', '检验', '检测', '预测', '评估者',
        '研究', '可行性', '论文', '文章', '文档', '资料', '数据', '代码',
        '系统', '平台', '工具', '框架', '模型', '算法', '接口', '服务',
        '谁', '哪', '什么', '怎么', '如何', '为何', '为什么', '多少',
        '人是', '写的', '作者', '名字', '身份', '简历',
        '总结一下', '总结', '概括', '归纳', '可行性研究', '可行性',
        '谁写', '谁写的', '是谁', '是什么', '怎么样', '怎样',
        '大型语言模型', '语言模型', '深度学习', '机器学习',
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

    @staticmethod
    def _extract_entity_terms(question: str) -> List[str]:
        """
        提取专有名词用于实体文档扩展（比 _extract_entities 更严格）。
        排除普通名词/动词/泛称（如「项目」「总结」「作者」），只保留人名、技术名、产品名等。
        面试点：为什么实体扩展要严格过滤？
        答：实体扩展会对每个实体做一次 BM25，若把「项目」「内容」当实体，
           会召回大量无关文档，反而稀释结果。只有专有名词才值得跨文档召回。
        规则：在 _extract_entities 基础上，再过滤 _NON_ENTITY_WORDS；
        另外专有名词通常含大写英文或长度≥3 的中文专名，用启发式筛选。
        """
        raw = RAGService._extract_entities(question)
        non_entity = RAGService._NON_ENTITY_WORDS
        import re
        result = []
        for t in raw:
            if t in non_entity:
                continue
            # 纯中文专名：长度≥2 且非泛称；含英文/数字的专有名词（FastAPI、Redis）直接保留
            has_cjk = bool(re.search(r'[一-龥]', t))
            has_ascii = bool(re.search(r'[A-Za-z]', t))
            has_digit = bool(re.search(r'[0-9]', t))
            if has_ascii or has_digit:
                result.append(t)  # 技术名/版本号直接保留
            elif has_cjk and len(t) >= 2:
                # 中文：排除"是""谁"等已被 _STOP_WORDS 过滤的，这里二次保险
                # 人名通常2-4字，技术中文名也常2-4字
                result.append(t)
        return result[:3]

    def _identify_entity_documents(self, entities: List[str]) -> set:
        """
        识别「实体命中文档」：name_match（文件名含实体）∪ BM25 top-3。
        面试题等噪声文档即使 BM25 命中也不在此集合，避免挤掉真实文档。
        """
        entity_docs = set()
        for entity in entities[:3]:
            try:
                results = self.es.search_keyword_only(entity, 15)
                name_matched = {r.get("file_name") for r in results
                                if entity in (r.get("file_name") or "")}
                top3 = {r.get("file_name") for r in results[:3] if r.get("file_name")}
                entity_docs |= name_matched | top3
            except Exception:
                continue
        return entity_docs

    def _expand_entity_documents(self, entities: List[str],
                                  existing_candidates: List[Dict],
                                  top_k: int) -> List[Dict]:
        """
        实体文档扩展：对每个实体做 BM25，把「实体命中文档中未进入候选池的」
        各取一个代表 chunk 注入候选。

        解决场景：库里有两篇张成都的论文，问「张成都是谁」时向量检索只偏向
        语义相近的一篇，另一篇没进候选。对「张成都」做 BM25 能命中两篇论文，
        把第二篇的代表 chunk 注入候选，后续多样性选择自然让它进入 primary，
        LLM 即可合并两篇信息回答。

        面试点：为什么只注入实体命中文档，而非所有 BM25 命中？
        答：面试题文档常把「张成都」当示例提到，BM25 也会命中。若全注入，
           面试题代表会挤掉真实论文。用 _identify_entity_documents 精准判定
           （文件名含实体 或 BM25 top-3），只注入真实文档代表。
        """
        if not entities:
            return []

        entity_docs = self._identify_entity_documents(entities)
        if not entity_docs:
            return []

        existing_docs = {c.get("file_name") for c in existing_candidates
                         if c.get("file_name")}
        existing_ids = {c.get("chunk_id") for c in existing_candidates}
        # 每文档保留分数最高的代表
        reps_by_doc: Dict[str, Dict] = {}

        for entity in entities[:3]:
            try:
                results = self.es.search_keyword_only(entity, max(top_k * 3, 15))
            except Exception as e:
                self.logger.warn("entity_expand", "rag_service",
                                 f"实体 BM25 失败", entity=entity, error=str(e)[:100])
                continue
            for r in results:
                fn = r.get("file_name")
                cid = r.get("chunk_id")
                if not fn or fn not in entity_docs or cid in existing_ids:
                    continue
                cur = reps_by_doc.get(fn)
                if cur is None or r.get("score", 0) > cur.get("score", 0):
                    reps_by_doc[fn] = r

        # 注入「候选池里没有的实体文档」的代表，并标记 entity_match
        # 分数归一化到 RRF 量级（~0.04），避免 BM25 裸分压垮原有 RRF 排序
        reps = []
        for fn, rep in reps_by_doc.items():
            if fn not in existing_docs:
                rep = dict(rep)
                rep["entity_match"] = True
                rep["score"] = 0.04
                reps.append(rep)
                existing_ids.add(rep.get("chunk_id"))

        if reps:
            self.logger.info("entity_expand", "rag_service",
                             "实体文档扩展注入",
                             entities=entities, new_docs=[r.get("file_name") for r in reps])
        return reps

    def _mark_entity_matched_chunks(self, candidates: List[Dict],
                                     entities: List[str]) -> int:
        """
        扫描候选池：
        - 对「实体命中文档」的 chunk 标记 entity_match=True（文档级优先覆盖）
        - 对「BM25 命中实体的 chunk」标记 entity_chunk=True（chunk 级含实体词）
        面试点：为什么区分 entity_match 与 entity_chunk？
        答：entity_match 标记文档（涉警论文是张成都的论文），entity_chunk 标记
            该文档中真正含「张成都」的 chunk（如作者页 #0，而非案例页 #20）。
            多样性选择第一轮选文档代表时，对 entity_match 文档优先选 entity_chunk，
            确保选中的 chunk 含实体信息，LLM 能提取身份/作者等关键事实。
        返回标记的 chunk 数。
        """
        if not entities or not candidates:
            return 0
        entity_docs = self._identify_entity_documents(entities)
        if not entity_docs:
            return 0
        # 收集 BM25 命中的 chunk_id（这些 chunk 内容一定含实体词）
        entity_chunk_ids: set = set()
        for entity in entities[:3]:
            try:
                results = self.es.search_keyword_only(entity, 15)
                entity_chunk_ids.update(r.get("chunk_id") for r in results if r.get("chunk_id"))
            except Exception:
                continue
        marked = 0
        for c in candidates:
            if c.get("file_name") in entity_docs:
                c["entity_match"] = True
                marked += 1
            if c.get("chunk_id") in entity_chunk_ids:
                c["entity_chunk"] = True
        return marked


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
