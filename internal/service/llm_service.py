"""
LLM API 调用服务 - 面试核心问题：
1. 支持多 Provider (OpenAI/Anthropic) 的设计模式？策略模式，统一接口
2. 为什么 API Key 透传而不是存储？用户自己的 Key，保护隐私，降低后端安全风险
3. Prompt 设计要点？角色设定 + 已知信息 + 回答约束
"""
import json
from typing import Dict

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
                 model: str = None) -> str:
        """
        统一生成接口 - 面试点：Provider 切换只需传参
        response 格式：
        - OpenAI:      {"choices": [{"message": {"content": "..."}}]}
        - Anthropic:   {"content": [{"text": "..."}]}
        本方法统一提取 answer text
        """
        cfg = get_config()
        llm_cfg = cfg.llm
        provider = provider or llm_cfg.default_provider
        model = model or llm_cfg.default_model
        prompt = self._build_prompt(question, context)

        if provider == "openai":
            return self._call_openai(api_key, model, prompt)
        elif provider == "anthropic":
            return self._call_anthropic(api_key, model, prompt)
        else:
            raise LLMAPIError(f"不支持的 Provider: {provider}")

    def _call_openai(self, api_key: str, model: str, prompt: str) -> str:
        """OpenAI API 调用"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system",
                 "content": "你是一个专业的知识库问答助手。请严格基于知识库内容回答，不要编造信息。"},
                {"role": "user", "content": prompt}
            ],
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

    def _call_anthropic(self, api_key: str, model: str, prompt: str) -> str:
        """Anthropic API 调用"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": self.default_max_tokens,
            "temperature": self.default_temperature,
            "system": "你是一个专业的知识库问答助手。请严格基于知识库内容回答，不要编造信息。",
            "messages": [
                {"role": "user", "content": prompt}
            ]
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

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Prompt 工程 - 面试点：RAG prompt 设计要点？
        1. 给 LLM 明确角色：知识库问答助手
        2. 提供检索到的上下文：已知信息
        3. 明确约束：
           - 有答案就引用来源
           - 不知道就说不知道
           - 不要编造信息
        4. 控制 prompt 长度：避免超过模型上下文窗口
        """
        return f"""已知信息（来自知识库检索，可能有不准确之处，请结合上下文判断）：

{context}

问题：{question}

回答要求：
1. 如果知识库中有明确相关信息，请基于已知信息回答，并引用对应的来源编号
2. 如果知识库中完全没有相关信息，请如实说明"知识库中未找到相关内容"
3. 回答要简洁准确，不要编造知识库中没有的信息
4. 回答使用中文"""
