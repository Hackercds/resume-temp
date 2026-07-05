"""
LLM API 调用服务 - 面试核心问题：
1. 支持多 Provider (OpenAI/Anthropic) 的设计模式？策略模式，统一接口
2. 为什么 API Key 透传而不是存储？用户自己的 Key，保护隐私，降低后端安全风险
3. Prompt 设计要点？角色设定 + 已知信息 + 回答约束
"""
import json
from typing import Dict, List

import httpx

from internal.model.config import get_config
from internal.pkg.logger import get_logger
from internal.pkg.errors import LLMAPIError


class LLMService:
    """
    LLM API 调用服务 - 面试点：
    1. 为什么 temperature=0.3？RAG 需要准确引用知识库，低温度减少幻觉
    2. 为什么 max_tokens=1000？知识库问答通常 500-1000 字够用，避免超长
    3. Prompt 中为何强调"不要编造"？这是 RAG 的核心：基于检索事实，不是凭空生成
    """

    def __init__(self):
        self.logger = get_logger()
        cfg = get_config()
        llm_cfg = cfg.llm
        self.timeout = llm_cfg.timeout
        self.default_temperature = llm_cfg.temperature
        self.default_max_tokens = llm_cfg.max_tokens

    def generate(self, question: str, context: str,
                 api_key: str, provider: str = "openai",
                 model: str = None, base_url: str = None,
                 history: List[Dict] = None,
                 allow_full_doc_retrieval: bool = True) -> str:
        """
        统一生成接口 - 面试点：Provider 切换只需传参
        base_url: 自定义 API 地址（兼容 DeepSeek / 豆包 / 本地模型等）
        """
        cfg = get_config()
        llm_cfg = cfg.llm
        provider = provider or llm_cfg.default_provider
        model = model or llm_cfg.default_model

        if provider == "openai" or provider == "custom":
            return self._call_openai(api_key, model, question, context,
                                     base_url, history, allow_full_doc_retrieval)
        elif provider == "anthropic":
            return self._call_anthropic(api_key, model, question, context,
                                        base_url, history, allow_full_doc_retrieval)
        else:
            raise LLMAPIError(f"不支持的 Provider: {provider}，可选 openai / anthropic / custom")

    def _call_openai(self, api_key: str, model: str,
                     question: str, context: str,
                     base_url: str = None,
                     history: List[Dict] = None,
                     allow_full_doc_retrieval: bool = True) -> str:
        """
        OpenAI 兼容 API 调用
        base_url: 自定义 API 地址，如 https://api.deepseek.com/v1
                  不传则用 OpenAI 官方地址
        """
        if base_url:
            url = base_url.rstrip("/") + "/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = [
            {"role": "system",
             "content": "你是一个专业的知识库问答助手。请严格基于知识库内容回答，不要编造信息。"}
        ]
        if history:
            quoted_files = []
            for m in history:
                if m.get("role") != "assistant":
                    continue
                for s in (m.get("sources") or []):
                    fn = s.get("file_name")
                    if fn and fn not in quoted_files:
                        quoted_files.append(fn)
            if quoted_files:
                ref_list = "\n".join(f"- {f}" for f in quoted_files)
                messages.append({
                    "role": "system",
                    "content": (
                        "【历次对话引用源（仅作背景；本次回答以最新检索结果为准）】\n"
                        f"{ref_list}\n\n"
                        "注意：用户可能追问上一轮你提到的细节（如日期、数字、术语）。"
                        "本次回答请基于最新检索结果，并可参考以上引用源文件名辅助定位。"
                        "如果你发现上一轮你说过某个具体日期/数字在本次检索结果里"
                        "找不到依据，请明确告知用户并修正，不要硬撑。"
                    )
                })
            for m in history:
                if m.get("role") in ("user", "assistant") and m.get("content"):
                    messages.append({"role": m["role"], "content": m["content"]})

        user_prompt = self._build_prompt(question, context,
                                         allow_full_doc_retrieval=allow_full_doc_retrieval)
        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": model,
            "messages": messages,
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail = e.response.text
            self.logger.error("llm_call", "llm_service",
                              "OpenAI API 调用失败",
                              status=e.response.status_code if e.response else 0,
                              error=str(error_detail)[:200])
            if e.response is not None and e.response.status_code == 401:
                raise LLMAPIError("API Key 无效，请检查后重试")
            raise LLMAPIError(f"OpenAI API 调用失败: {str(error_detail)[:100]}")
        except httpx.TimeoutException:
            raise LLMAPIError(f"API 调用超时 ({self.timeout}s)，请稍后重试")
        except Exception as e:
            raise LLMAPIError(f"API 调用异常: {str(e)}")

    def _call_anthropic(self, api_key: str, model: str,
                        question: str, context: str,
                        base_url: str = None,
                        history: List[Dict] = None,
                        allow_full_doc_retrieval: bool = True) -> str:
        """Anthropic API 调用"""
        if base_url:
            url = base_url.rstrip("/") + "/messages"
        else:
            url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        messages = []
        if history:
            quoted_files = []
            for m in history:
                if m.get("role") != "assistant":
                    continue
                for s in (m.get("sources") or []):
                    fn = s.get("file_name")
                    if fn and fn not in quoted_files:
                        quoted_files.append(fn)
            if quoted_files:
                ref_list = "\n".join(f"- {f}" for f in quoted_files)
                messages.append({
                    "role": "system",
                    "content": (
                        "【历次对话引用源（仅作背景；本次回答以最新检索结果为准）】\n"
                        f"{ref_list}\n\n"
                        "注意：用户可能追问上一轮你提到的细节（如日期、数字、术语）。"
                        "本次回答请基于最新检索结果，并可参考以上引用源文件名辅助定位。"
                        "如果你发现上一轮你说过某个具体日期/数字在本次检索结果里"
                        "找不到依据，请明确告知用户并修正，不要硬撑。"
                    )
                })
            for m in history:
                if m.get("role") in ("user", "assistant") and m.get("content"):
                    messages.append({"role": m["role"], "content": m["content"]})

        user_prompt = self._build_prompt(question, context,
                                         allow_full_doc_retrieval=allow_full_doc_retrieval)
        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": model,
            "max_tokens": self.default_max_tokens,
            "temperature": self.default_temperature,
            "system": "你是一个专业的知识库问答助手。请严格基于知识库内容回答，不要编造信息。",
            "messages": messages
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
            # Anthropic 返回 content 是数组
            content = data.get("content", [])
            if isinstance(content, list):
                return "".join(
                    block.get("text", "") for block in content
                    if block.get("type") == "text"
                )
            return str(content)
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 401:
                raise LLMAPIError("API Key 无效，请检查后重试")
            raise LLMAPIError(f"Anthropic API 调用失败: HTTP {e.response.status_code if e.response else 0}")
        except httpx.TimeoutException:
            raise LLMAPIError(f"API 调用超时 ({self.timeout}s)，请稍后重试")
        except Exception as e:
            raise LLMAPIError(f"API 调用异常: {str(e)}")

    async def generate_stream(self, question: str, context: str,
                              api_key: str, provider: str = "openai",
                              model: str = None, base_url: str = None,
                              history: List[Dict] = None,
                              allow_full_doc_retrieval: bool = True):
        """
        流式生成接口 - yield 每个内容 token（不含思考过程）
        allow_full_doc_retrieval: 是否允许 LLM 请求召回完整文档
        """
        cfg = get_config()
        llm_cfg = cfg.llm
        provider = provider or llm_cfg.default_provider
        model = model or llm_cfg.default_model

        if provider in ("openai", "custom"):
            async for token in self._call_openai_stream(
                api_key, model, question, context, base_url, history,
                allow_full_doc_retrieval
            ):
                yield token
        elif provider == "anthropic":
            async for token in self._call_anthropic_stream(
                api_key, model, question, context, base_url, history,
                allow_full_doc_retrieval
            ):
                yield token
        else:
            raise LLMAPIError(f"不支持的 Provider: {provider}，可选 openai / anthropic / custom")

    async def _call_openai_stream(self, api_key: str, model: str,
                                  question: str, context: str,
                                  base_url: str = None,
                                  history: List[Dict] = None,
                                  allow_full_doc_retrieval: bool = True):
        """
        OpenAI 兼容流式调用 - 只返回 content，过滤 reasoning_content
        """
        if base_url:
            url = base_url.rstrip("/") + "/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        messages = [
            {"role": "system",
             "content": "你是一个专业的知识库问答助手。请严格基于知识库内容回答，不要编造信息。"}
        ]
        if history:
            quoted_files = []
            for m in history:
                if m.get("role") != "assistant":
                    continue
                for s in (m.get("sources") or []):
                    fn = s.get("file_name")
                    if fn and fn not in quoted_files:
                        quoted_files.append(fn)
            if quoted_files:
                ref_list = "\n".join(f"- {f}" for f in quoted_files)
                messages.append({
                    "role": "system",
                    "content": (
                        "【历次对话引用源（仅作背景；本次回答以最新检索结果为准）】\n"
                        f"{ref_list}\n\n"
                        "注意：用户可能追问上一轮你提到的细节（如日期、数字、术语）。"
                        "本次回答请基于最新检索结果，并可参考以上引用源文件名辅助定位。"
                        "如果你发现上一轮你说过某个具体日期/数字在本次检索结果里"
                        "找不到依据，请明确告知用户并修正，不要硬撑。"
                    )
                })
            for m in history:
                if m.get("role") in ("user", "assistant") and m.get("content"):
                    messages.append({"role": m["role"], "content": m["content"]})

        user_prompt = self._build_prompt(question, context,
                                         allow_full_doc_retrieval=allow_full_doc_retrieval)
        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": model,
            "messages": messages,
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens,
            "stream": True,
            # 让 OpenAI 在最后一个 chunk 里返回 usage（prompt/completion/total tokens）
            # 启用后客户端可以知道每次调用花了多少 token，做成本核算
            "stream_options": {"include_usage": True}
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        # OpenAI 流最后一个 chunk：choices=[], usage={prompt_tokens,completion_tokens,total_tokens}
                        # 不再有文本 token，单独 yield 一个 usage marker 给 caller
                        usage = data.get("usage")
                        if usage:
                            yield {"__usage__": usage}
                            continue
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        # 只输出 content，忽略 reasoning_content / thinking
                        content = delta.get("content")
                        if content:
                            yield content
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = await e.response.aread()
                error_detail = error_detail.decode('utf-8', errors='ignore')
            except Exception:
                error_detail = str(e)
            self.logger.error("llm_stream", "llm_service",
                              "OpenAI 流式调用失败",
                              status=e.response.status_code if e.response else 0,
                              error=str(error_detail)[:200])
            if e.response is not None and e.response.status_code == 401:
                raise LLMAPIError("API Key 无效，请检查后重试")
            raise LLMAPIError(f"OpenAI API 流式调用失败: {str(error_detail)[:100]}")
        except httpx.TimeoutException:
            raise LLMAPIError(f"API 流式调用超时 ({self.timeout}s)，请稍后重试")
        except Exception as e:
            raise LLMAPIError(f"API 流式调用异常: {str(e)}")

    async def _call_anthropic_stream(self, api_key: str, model: str,
                                     question: str, context: str,
                                     base_url: str = None,
                                     history: List[Dict] = None,
                                     allow_full_doc_retrieval: bool = True):
        """
        Anthropic 流式调用 - 忽略 thinking，只返回 text_delta
        """
        if base_url:
            url = base_url.rstrip("/") + "/messages"
        else:
            url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        messages = []
        if history:
            quoted_files = []
            for m in history:
                if m.get("role") != "assistant":
                    continue
                for s in (m.get("sources") or []):
                    fn = s.get("file_name")
                    if fn and fn not in quoted_files:
                        quoted_files.append(fn)
            if quoted_files:
                ref_list = "\n".join(f"- {f}" for f in quoted_files)
                messages.append({
                    "role": "system",
                    "content": (
                        "【历次对话引用源（仅作背景；本次回答以最新检索结果为准）】\n"
                        f"{ref_list}\n\n"
                        "注意：用户可能追问上一轮你提到的细节（如日期、数字、术语）。"
                        "本次回答请基于最新检索结果，并可参考以上引用源文件名辅助定位。"
                        "如果你发现上一轮你说过某个具体日期/数字在本次检索结果里"
                        "找不到依据，请明确告知用户并修正，不要硬撑。"
                    )
                })
            for m in history:
                if m.get("role") in ("user", "assistant") and m.get("content"):
                    messages.append({"role": m["role"], "content": m["content"]})

        user_prompt = self._build_prompt(question, context,
                                         allow_full_doc_retrieval=allow_full_doc_retrieval)
        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": model,
            "max_tokens": self.default_max_tokens,
            "temperature": self.default_temperature,
            "system": "你是一个专业的知识库问答助手。请严格基于知识库内容回答，不要编造信息。",
            "messages": messages,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        event_type = data.get("type")
                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield text
                        # Anthropic 流末尾的 message_delta 带 usage（input_tokens/output_tokens）
                        elif event_type == "message_delta":
                            usage = data.get("usage")
                            if usage:
                                # 转换为 OpenAI 同构字段，方便前端统一处理
                                yield {
                                    "__usage__": {
                                        "prompt_tokens": usage.get("input_tokens", 0),
                                        "completion_tokens": usage.get("output_tokens", 0),
                                        "total_tokens": (usage.get("input_tokens", 0)
                                                         + usage.get("output_tokens", 0)),
                                    }
                                }
                        # 忽略 thinking、content_block_start、content_block_stop 等
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 401:
                raise LLMAPIError("API Key 无效，请检查后重试")
            raise LLMAPIError(f"Anthropic API 流式调用失败: HTTP {e.response.status_code if e.response else 0}")
        except httpx.TimeoutException:
            raise LLMAPIError(f"API 流式调用超时 ({self.timeout}s)，请稍后重试")
        except Exception as e:
            raise LLMAPIError(f"API 流式调用异常: {str(e)}")

    def _build_prompt(self, question: str, context: str,
                      allow_full_doc_retrieval: bool = True) -> str:
        """
        Prompt 工程 - 面试点：RAG prompt 设计要点？
        1. 给 LLM 明确角色：知识库问答助手
        2. 提供检索到的上下文：已知信息
        3. 明确约束：
           - 有答案就引用来源编号
           - 不知道就说不知道（防幻觉核心）
           - 不要编造信息
        4. 控制 prompt 长度：避免超过模型上下文窗口
        5. 邻域上下文说明：检索可能包含相邻片段，需自行衔接，不相关则忽略

        面试点：为什么不再让 LLM 输出 {{retrieve_full_doc}} 标记自动触发全文召回？
        答：旧设计让 LLM 输出标记，后端召回完整文档后重新生成整个答案。
           多文档/多部分问题会让 LLM 多次输出标记，级联重新生成，产生多个答案
           拼接（用户看到「5个回答」），且每次重新生成耗时10s+，体验差。
           初始检索（向量+BM25+邻域扩展+实体文档扩展）已提供足够上下文；
           用户需要全文时可手动点「召回完整文档」按钮（retrieve_full_doc=true，
           走短路直接返回文档原文，不调 LLM）。故移除自动标记特性。
        """
        return f"""你是知识库问答助手。请严格基于下面的"已知信息"回答用户问题，不要编造信息。

已知信息（来自知识库检索，可能包含为补充上下文而召回的相邻片段，请自行衔接理解；与问题无关的片段请忽略。本次检索结果是回答的主要依据）：

{context}

用户问题：{question}

回答要求（严格遵守，逐条执行）：
1. 优先基于上面的"已知信息"回答。引用事实时**必须**标注来源编号（如「据【来源1】」），不要给出无法映射到已知信息具体来源的陈述。
2. 如果"已知信息"中同时出现同一来源的相邻片段，应合并理解，不要把它们当成矛盾信息重复列举。
3. 引用编号与内容必须一一对应——不要捏造"【来源N】"编号去支撑你编造的事实，也不要把【来源1】的内容标成【来源2】。
4. **如果"已知信息"完全不相关或不足以回答，请如实回答"知识库中未找到相关内容"**，并提示用户换个问法或上传相关文档，不要用外部知识补全；这是最重要的防幻觉约束。
5. 涉及具体数字、日期、名称时尤其谨慎：必须能在已知信息中找到对应原文才能写，不能凭"看起来合理"就推断。
6. 回答要简洁准确、条理清晰；涉及列举时用要点或表格，避免冗长铺陈。
7. 回答使用中文。"""
