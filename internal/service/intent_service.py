"""
意图识别服务 - 支持规则零成本识别 + 可选 LLM 二分类增强

面试点：为什么先做规则？
- 规则零延迟、零成本，覆盖 80% 常见追问场景；
- LLM 作为高置信度兜底，解决隐式指代、长问句歧义。
"""
import re
from typing import List, Dict, Optional, Tuple

from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.service.llm_service import LLMService


class IntentService:
    """对话意图识别服务"""

    # 强指代词：几乎一定是追问
    _STRONG_FOLLOW_UP = {
        '他', '她', '它', '这个', '那个', '这篇', '这份', '这个文件',
        '此人', '这位', '该', '上述', '前者', '后者',
    }

    # 弱指代词：结合历史更可能是追问
    _WEAK_FOLLOW_UP = {
        '这', '那', '上文', '前面', '刚才', '接着', '还有', '另外',
        '除此之外', '那么', '接下来',
    }

    # 总结/对比意图
    _SUMMARIZE_KEYWORDS = {
        '总结', '概括', '归纳', '对比', '比较', '区别', '不同', '相同',
        '优劣', '整体', '全文', '主要', '主旨', '核心', '概要',
    }

    # 澄清/解释意图
    _CLARIFY_KEYWORDS = {
        '解释', '说明', '举例', '举个例子', '详细', '展开', '为什么',
        '怎么回事', '什么意思', '如何理解',
    }

    # 对比/比较关键词：当问题包含两个实体+对比词时，更可能是 follow_up 而不是 summarize
    _COMPARISON_KEYWORDS = {'和', '与', '及', '以及', '还有', '还是'}

    def __init__(self, llm_service: LLMService = None):
        self.llm = llm_service
        self.logger = get_logger()

    def classify(self, question: str, history: List[Dict],
                 llm_kwargs: Optional[Dict] = None) -> Tuple[str, float]:
        """
        返回 (intent, confidence)
        intent: new_topic | follow_up | summarize | clarify

        llm_kwargs: 启用 LLM 识别时透传给 LLMService 的参数
            {api_key, provider, model, base_url}，无则用配置默认。
        """
        cfg = get_config().conversation

        # 1. 无历史 → 新话题
        if not history:
            return "new_topic", 1.0

        # 2. 规则识别
        intent, confidence = self._rule_classify(question)
        if confidence >= cfg.intent_rule_confidence_threshold:
            return intent, confidence

        # 3. 规则置信度不足，且启用 LLM 识别 → 调用 LLM
        if cfg.enable_intent_llm and self.llm:
            try:
                return self._llm_classify(question, history, llm_kwargs or {})
            except Exception as e:
                self.logger.warn("intent", "intent_service",
                                 "LLM 意图识别失败，回退到规则", error=str(e))
                return intent, confidence

        return intent, confidence

    def _rule_classify(self, question: str) -> Tuple[str, float]:
        """规则版意图识别，返回意图 + 置信度"""
        q = question.strip().lower()
        stripped = question.strip()

        # 强指代词 → 追问（最高优先级）
        if any(k in q for k in self._STRONG_FOLLOW_UP):
            return "follow_up", 0.95

        # 总结意图（显式总结词，且不包含强对比实体结构）
        has_summarize_word = any(k in q for k in self._SUMMARIZE_KEYWORDS)
        has_comparison_structure = (
            len(stripped) >= 12
            and any(k in q for k in self._COMPARISON_KEYWORDS)
        )
        if has_summarize_word and not has_comparison_structure:
            return "summarize", 0.95

        # 澄清意图
        if any(k in q for k in self._CLARIFY_KEYWORDS):
            return "clarify", 0.85

        # 弱指代词 → 追问（置信度稍低）
        if any(k in q for k in self._WEAK_FOLLOW_UP):
            return "follow_up", 0.80

        # 极短问题（<10 字）且有历史 → 大概率追问
        if len(stripped) < 10:
            return "follow_up", 0.75

        # 问题较短（10-22 字）且以疑问词开头 → 可能是追问
        if len(stripped) < 22 and re.match(r'^(为什么|怎么|多少|多久|哪些|什么样|哪|什么)', stripped):
            return "follow_up", 0.65

        # 长问题但包含强对比结构（A 和 B 有什么区别）→ 结合历史判断
        if len(stripped) >= 15 and any(k in q for k in self._COMPARISON_KEYWORDS):
            return "follow_up", 0.70

        # 长问题（>=25字）且包含总结/澄清词，但与上文主题高度相关时，仍可能是追问；
        # 如果既不包含总结/澄清词，也无指代，且长度>=25 → 新话题
        if len(stripped) >= 25 and not has_summarize_word and not any(k in q for k in self._CLARIFY_KEYWORDS):
            return "new_topic", 0.80

        # 中等长度问题（12-24字），包含“介绍一下/什么是/如何/为什么”等开头，且无指代 → 新话题
        new_topic_markers = {'介绍一下', '什么是', '如何', '为什么', '怎么', '有哪些'}
        if len(stripped) >= 12 and any(stripped.startswith(k) for k in new_topic_markers) and not any(k in q for k in self._STRONG_FOLLOW_UP | self._WEAK_FOLLOW_UP):
            return "new_topic", 0.75

        # 默认追问（有历史但无强新话题信号）
        return "follow_up", 0.60

    def _llm_classify(self, question: str, history: List[Dict],
                      llm_kwargs: Dict) -> Tuple[str, float]:
        """
        调用 LLM 做意图四分类。
        面试点：为什么规则优先、LLM 兜底？
        答：规则零延迟零成本覆盖 80% 明确指代；剩余 20% 隐式指代/长句歧义
            才花一次 LLM 调用判定，整体成本与延迟最优。
        用 JSON 输出约束 LLM 返回结构化结果，解析失败回退规则。
        """
        import json

        full_cfg = get_config()
        cfg = full_cfg.conversation
        # 取最近 2 轮作为上下文
        recent = history[-4:] if len(history) >= 4 else history
        context_text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')[:200]}"
            for m in recent
        )

        prompt = f"""你是对话意图识别专家。请判断当前用户问题与前文的关系。

前文：
{context_text}

当前问题：{question}

请从以下分类中选择最符合的一项，并给出 0-1 的置信度：
- new_topic：新话题，与上文无关
- follow_up：追问/指代上文提到的实体或内容
- summarize：总结/对比/概括前文
- clarify：要求解释/举例/详细说明前文

必须严格按 JSON 格式输出，不要解释：
{{"intent": "follow_up", "confidence": 0.92}}"""

        api_key = llm_kwargs.get("api_key") or getattr(full_cfg.app, "default_api_key", "") or ""
        if not api_key:
            raise ValueError("LLM 意图识别需要 api_key，但未提供")

        # 复用 LLMService 的生成能力（用一个极简 context 占位，不触发检索标记）
        text = self.llm.generate(
            question=prompt,
            context="(无需检索，直接判断意图)",
            api_key=api_key,
            provider=llm_kwargs.get("provider", "openai"),
            model=cfg.intent_model,
            base_url=llm_kwargs.get("base_url"),
            history=[],
            allow_full_doc_retrieval=False,
        )

        # 解析 JSON（容错：提取首个含 intent 的 {...}）
        m = re.search(r'\{[^{}]*"intent"[^{}]*\}', text, re.DOTALL)
        if not m:
            raise ValueError(f"LLM 意图识别返回无法解析: {text[:120]}")
        data = json.loads(m.group(0))
        intent = data.get("intent", "follow_up")
        confidence = float(data.get("confidence", 0.6))
        if intent not in ("new_topic", "follow_up", "summarize", "clarify"):
            intent = "follow_up"
        return intent, max(0.0, min(1.0, confidence))

    @staticmethod
    def extract_entities(question: str, history: List[Dict]) -> List[str]:
        """
        从历史 assistant 答案中提取关键实体（名词短语），用于追问继承。
        当前使用简单规则：提取被引号、书名号包裹的内容，以及连续中文字段。
        """
        entities = []

        # 从最近 assistant 答案中提取引号/书名号内容
        recent_assistant = next(
            (m for m in reversed(history) if m.get("role") == "assistant"),
            None
        )
        if recent_assistant:
            content = recent_assistant.get("content", "")
            # 匹配 “xxx”、「xxx」、"xxx"、《xxx》
            quoted = re.findall(r'[“""`「《]([^”""`」》]{2,20})[”""`」》]', content)
            entities.extend(quoted)

            # 匹配可能的人名（句首/主语位置 + 身份词）
            name_patterns = [
                r'^([一-龥]{2,4})(?:是|为|，)',  # 张成都是... / 张成都，...
                r'([一-龥]{2,4})(?:先生|女士|同学|工程师|简历|项目|经验|熟悉|掌握|使用)',
            ]
            for pattern in name_patterns:
                for m in re.finditer(pattern, content):
                    name = m.group(1)
                    # 过滤常见误匹配：单个字或“测试/开发”等泛称
                    if name and len(name) >= 2 and name not in {'测试', '开发', '工程', '技术', '项目', '经验'}:
                        entities.append(name)

        # 去重保序
        seen = set()
        result = []
        for e in entities:
            e = e.strip()
            if e and e not in seen and len(e) >= 2:
                seen.add(e)
                result.append(e)
        return result[:5]

    @staticmethod
    def resolve_references(question: str, entities: List[str]) -> str:
        """
        把追问中的指代词替换为实体。
        例如："他的项目和他的经验" → "张成都的项目和张成都的经验"

        增强：支持多代词/多次出现替换（旧版只替换首个）。
        策略：
        1. 「这个/那个 + 量词/名词」「该/此 + 名词」→ 主实体 + 后缀
        2. 独立的「他/她/它」→ 主实体（替换所有出现，因 RAG 上下文里代词多指同一主体）
        多实体场景（实体≥2）保守起见只替换句首代词，避免错配。
        """
        if not entities:
            return question

        q = question
        primary = entities[0]
        multi_entity = len(entities) >= 2

        # 1. 「这个/那个 + 量词或名词」→ 主实体 + 后缀
        suffix_patterns = [
            (r'这个(人|文件|文档|项目|简历|人|岗位)', f'{primary}\\1'),
            (r'那个(人|文件|文档|项目|简历|岗位)', f'{primary}\\1'),
            (r'这(篇|份|个)(文档|文件|简历|项目)', f'{primary}的\\2'),
            (r'那(篇|份|个)(文档|文件|简历|项目)', f'{primary}的\\2'),
            (r'^(该|此)(人|项目|文档|文件|简历)', f'{primary}\\2'),
        ]
        for pattern, repl in suffix_patterns:
            q = re.sub(pattern, repl, q, count=1)

        # 2. 独立代词替换
        if multi_entity:
            # 多实体：只替换句首一个代词，避免把第二个实体的「他」也替换错
            # 句首「他」无论后接什么都替换为主实体（如「他和她」→「张三和她」）
            q = re.sub(r'^(他|她|它)', primary, q, count=1)
        else:
            # 单实体：替换所有「他/她/它」及其带「的」的形式
            # 先保护「其他」这个词，避免误替换
            q = re.sub(r'(?<!其)(他|她|它)(?=的|之|会|是|用|有|做)', primary, q)
            # 句首裸代词
            q = re.sub(r'^(他|她|它)(?=[一-龥])', primary, q, count=1)

        # 3. 「这个/那个」单独出现（后接中文词）→ 主实体
        q = re.sub(r'这个(?=[一-龥]{2,})', primary, q, count=1)
        q = re.sub(r'那个(?=[一-龥]{2,})', primary, q, count=1)

        return q
