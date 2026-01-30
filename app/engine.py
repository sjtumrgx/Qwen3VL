"""
LMDeploy 推理引擎封装

通过 LMDeploy OpenAI API Server 推理 Qwen3-VL（FastAPI 作为轻量适配层）
"""

import json
import logging
from typing import Any, Dict, Generator, List, Optional

import requests

from .config import config

logger = logging.getLogger(__name__)


def _ensure_data_url(image_base64_or_data_url: str) -> str:
    value = (image_base64_or_data_url or "").strip()
    if value.startswith("data:"):
        return value
    return f"data:image/jpeg;base64,{value}"


def _normalize_messages(messages: List[dict]) -> List[dict]:
    normalized: List[dict] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str) or content is None:
            normalized.append({"role": role, "content": content or ""})
            continue

        if not isinstance(content, list):
            normalized.append({"role": role, "content": str(content)})
            continue

        new_content: List[dict] = []
        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type == "text":
                new_content.append({"type": "text", "text": item.get("text", "")})
                continue

            if item_type == "image_url":
                image_url = item.get("image_url") or {}
                url = image_url.get("url", "")
                new_content.append(
                    {"type": "image_url", "image_url": {"url": _ensure_data_url(url)}}
                )
                continue

            if item_type == "image":
                raw = item.get("data") or item.get("image") or ""
                if isinstance(raw, str) and raw:
                    new_content.append(
                        {"type": "image_url", "image_url": {"url": _ensure_data_url(raw)}}
                    )
                continue

        normalized.append({"role": role, "content": new_content})

    return normalized


class LMDeployEngine:
    """LMDeploy 推理引擎（HTTP 转发）"""

    _instance: Optional["LMDeployEngine"] = None

    def __init__(self) -> None:
        self.base_url = config.lmdeploy_base_url.rstrip("/")
        self.timeout_s = config.lmdeploy_timeout_s
        self.model_name = config.model_path.split("/")[-1]

        self._resolved_model: Optional[str] = None
        try:
            self._resolved_model = self._resolve_model_id()
        except Exception as e:
            logger.warning(f"LMDeploy 模型探测失败，将使用默认 model 字段: {e!r}")

    @classmethod
    def get_instance(cls) -> "LMDeployEngine":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _resolve_model_id(self) -> str:
        resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        models = data.get("data", []) or []
        if models and isinstance(models[0], dict) and models[0].get("id"):
            return str(models[0]["id"])
        return self.model_name

    def generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> dict:
        if images:
            raise ValueError("LMDeployEngine 暂不支持 images 参数（请使用 messages 传入多图）")

        content: Any
        if image:
            content = [
                {"type": "image_url", "image_url": {"url": _ensure_data_url(image)}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        messages = [{"role": "user", "content": content}]
        return self.generate_from_messages(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate_from_messages(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> dict:
        model = self._resolved_model or self.model_name

        payload: Dict[str, Any] = {
            "model": model,
            "messages": _normalize_messages(messages),
            "max_tokens": max_tokens or config.max_tokens,
            "temperature": temperature if temperature is not None else config.temperature,
            "top_p": top_p if top_p is not None else config.top_p,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json() or {}

        text = ""
        choices = data.get("choices", []) or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict):
                text = str(message.get("content") or "")

        usage = data.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def generate_stream(
        self,
        prompt: str,
        image: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成（单图或纯文本）

        Yields:
            SSE 格式的数据块
        """
        content: Any
        if image:
            content = [
                {"type": "image_url", "image_url": {"url": _ensure_data_url(image)}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        messages = [{"role": "user", "content": content}]
        yield from self.generate_from_messages_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate_from_messages_stream(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成（多轮对话/多图）

        Yields:
            SSE 格式的数据块
        """
        model = self._resolved_model or self.model_name

        payload: Dict[str, Any] = {
            "model": model,
            "messages": _normalize_messages(messages),
            "max_tokens": max_tokens or config.max_tokens,
            "temperature": temperature if temperature is not None else config.temperature,
            "top_p": top_p if top_p is not None else config.top_p,
            "stream": True,
        }

        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
            stream=True,
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue

                # SSE 格式: "data: {...}" 或 "data: [DONE]"
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        # 直接转发 LMDeploy 的 SSE 数据
                        yield f"data: {data_str}\n\n"
                    except Exception as e:
                        logger.warning(f"解析流式数据失败: {e}")
                        continue


def get_engine() -> LMDeployEngine:
    """获取推理引擎实例"""
    return LMDeployEngine.get_instance()
