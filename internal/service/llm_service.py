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
            "stream_options": {"include_usage": False}
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
        """
        full_doc_hint = ""
        if allow_full_doc_retrieval:
            full_doc_hint = """
6. 【整篇文档召回标记】如果你判断当前"已知信息"碎片化、缺少全局背景、或无法覆盖用户问题的全貌，请输出一个或多个标记请求完整文档：
   {{retrieve_full_doc:文件名}}
   - 每个文件只能标记一次，标记之间用换行分隔
   - 标记必须独立成行（前后无其他字符）
   - 同时给出你认为需要的补充检索方向（先用一行简要说明）
   - 系统会召回完整文档并自动重新生成答案；你不需要解释标记的含义
7. 【关键禁止】你的最终答案中**绝对不能**包含任何 `{{retrieve_full_doc:...}}` 标记 —— 标记仅用于触发检索，不可直接出现在面向用户的回答中。如果你判断不需要额外文档，则**不要**输出任何标记。"""

        return f"""你是知识库问答助手。请严格基于下面的"已知信息"回答用户问题，不要编造信息。

已知信息（来自知识库检索，可能包含为补充上下文而召回的相邻片段，请自行衔接理解；与问题无关的片段请忽略。本次检索结果是回答的主要依据）：

{context}

用户问题：{question}

回答要求：
1. 优先基于上面的"已知信息"回答，并在引用事实时标注来源编号（如「据【来源1】」）。
2. 如果"已知信息"中同时出现同一来源的相邻片段，应合并理解，不要把它们当成矛盾信息重复列举。
3. 如果"已知信息"完全不相关或不足以回答，请如实说明"知识库中未找到相关内容"，并提示用户换个问法或上传相关文档，不要用外部知识补全。
4. 回答要简洁准确、条理清晰；涉及列举时用要点或表格，避免冗长铺陈。
5. 回答使用中文。{full_doc_hint}"""
