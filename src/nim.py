"""NVIDIA NIM client — OpenAI-compatible chat completions."""

from __future__ import annotations

import os
import time
from typing import Any

NIM_BASE = "https://integrate.api.nvidia.com/v1"
LIGHTNING = "nvidia/nemotron-3.5-lightning-30b-a3b"
NANO = "nvidia/nemotron-3-nano-30b-a3b"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"


def api_key() -> str:
    return (
        os.getenv("NVIDIA_API_KEY", "").strip()
        or os.getenv("NGC_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def base_url() -> str:
    raw = os.getenv("NVIDIA_BASE_URL", "").strip() or NIM_BASE
    base = raw.rstrip("/")
    if base and not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def nim_model() -> str:
    return os.getenv("NIM_MODEL", LIGHTNING).strip() or LIGHTNING


def complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 256,
    enable_thinking: bool = False,
    timeout: float = 60.0,
) -> dict[str, Any]:
    import httpx

    key = api_key()
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    model = model or nim_model()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    t0 = time.perf_counter()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{base_url()}/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400 and "chat_template_kwargs" in payload:
            payload.pop("chat_template_kwargs", None)
            r = client.post(f"{base_url()}/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"NIM {r.status_code}: {r.text[:400]}")
        data = r.json()
    ms = int((time.perf_counter() - t0) * 1000)
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning_content") or "").strip(),
        "model": data.get("model") or model,
        "ms": ms,
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
    }
