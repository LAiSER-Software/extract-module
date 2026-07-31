import os
import re
from typing import Optional

import requests

from laiser.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P

_CODE_FENCE_RE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def _clean_llm_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = _CODE_FENCE_RE.sub("", text)
    return text.strip()


def anthropic_generate(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
    timeout: int = 60,
    max_tokens: int = 4096,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    """Generate text with the Anthropic Messages API.

    Decoding is deterministic by default: ``temperature`` defaults to
    ``config.DEFAULT_TEMPERATURE`` (0.0, i.e. greedy decoding). ``top_p`` is
    only transmitted when it departs from 1.0, because the Messages API
    discourages setting both ``temperature`` and ``top_p`` in one request.
    The API does not accept a sampling seed.
    """
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not provided")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if top_p is not None and top_p != 1.0:
        payload["top_p"] = top_p

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print("status:", resp.status_code)

        if resp.status_code != 200:
            print("body:", resp.text)
        resp.raise_for_status()

        data = resp.json()

        # Extract text from content blocks
        raw_text = ""
        for content_block in data.get("content", []):
            if content_block.get("type") == "text":
                raw_text += content_block.get("text", "")

        return _clean_llm_text(raw_text)

    except requests.RequestException as e:
        print(f"[Anthropic] Request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print("[Anthropic] status:", e.response.status_code)
            print("[Anthropic] body:", e.response.text)
        return ""
