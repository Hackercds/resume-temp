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

    def classify(self, question: str, history: List[Dict]) -> Tuple[str, float]:
        """
        返回 (intent, confidence)
        intent: new_topic | follow_up | summarize | clarify
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
                return self._llm_classify(question, history)
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

    def _llm_classify(self, question: str, history: List[Dict]) -> Tuple[str, float]:
        """调用轻量 LLM 做意图二分类"""
        # 取最近 1 轮 user + assistant 作为上下文
        recent = history[-2:] if len(history) >= 2 else history
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

        # 使用 LLMService 的非流式方法生成
        # 这里用默认 provider/model，无需 api_key（LLMService 当前设计需要 api_key）
        # 因此 IntentService 依赖外部传入 llm_service，实际不在这里直接调用 LLM
        # 改为抛出异常，由 RAGService 统一调用
        raise NotImplementedError("LLM 意图识别由 RAGService 统一调度，避免重复初始化")

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
        例如："他的项目" → "张成都的项目"
        """
        if not entities:
            return question

        q = question
        primary = entities[0]

        # 指代词 → 实体
        replacements = [
            (r'^(他|她|它|该|此)(?=[一-龥])', primary),
            (r'这个(人|文件|文档|项目|简历)', f'{primary}\\1'),
            (r'那个(人|文件|文档|项目|简历)', f'{primary}\\1'),
            (r'这个(?=[一-龥]{2,})', primary),
            (r'那个(?=[一-龥]{2,})', primary),
        ]

        for pattern, repl in replacements:
            new_q = re.sub(pattern, repl, q, count=1)
            if new_q != q:
                return new_q

        return q
