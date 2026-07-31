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


def openai_generate(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4.1-mini",
    timeout: int = 60,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    """Generate text with the OpenAI Responses API.

    Decoding is deterministic by default: ``temperature`` defaults to
    ``config.DEFAULT_TEMPERATURE`` (0.0, i.e. greedy decoding) and ``top_p``
    defaults to ``config.DEFAULT_TOP_P`` (1.0, nucleus filtering inactive).
    The Responses API does not accept a sampling seed, so reproducibility on
    this backend rests on greedy decoding alone.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print("status:", resp.status_code)
        if resp.status_code != 200:
            print("body:", resp.text)
        resp.raise_for_status()

        data = resp.json()

        # ✅ MINIMAL FIX: extract text from output[].content[]
        raw_text = ""
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    raw_text += c["text"]

        # (optional) fallback if output_text exists
        if not raw_text:
            raw_text = data.get("output_text", "")

        return _clean_llm_text(raw_text)

    except requests.RequestException as e:
        print(f"[OpenAI] Request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print("[OpenAI] status:", e.response.status_code)
            print("[OpenAI] body:", e.response.text)
        return ""
