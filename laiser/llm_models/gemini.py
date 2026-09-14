"""Gemini generation helpers for LAiSER."""

import json
import os
import re
from typing import Any, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from laiser.config import DEFAULT_TEMPERATURE, GENERATION_SEED

DEFAULT_GEMINI_MODEL = os.getenv("LAISER_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GEMINI_TIMEOUT = float(os.getenv("LAISER_GEMINI_TIMEOUT", "60"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("LAISER_GEMINI_MAX_OUTPUT_TOKENS", "4096"))

# Older google-genai releases have no seed field on GenerateContentConfig, and
# passing one there fails validation. Check once so a seed is only sent when the
# installed release accepts it.
_SUPPORTS_SEED = "seed" in getattr(types.GenerateContentConfig, "model_fields", {})


class GeminiAPI:
    """Small wrapper to keep backward compatibility with older imports."""

    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        timeout: float = DEFAULT_GEMINI_TIMEOUT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: Optional[int] = GENERATION_SEED,
    ):
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.api_key = api_key
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.seed = seed
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        return self.generate_with_config(prompt)

    def generate_with_config(
        self,
        prompt: str,
        *,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Any] = None,
    ) -> str:
        try:
            config_kwargs = {"temperature": self.temperature, "max_output_tokens": self.max_output_tokens}
            if self.seed is not None and _SUPPORTS_SEED:
                config_kwargs["seed"] = self.seed
            if response_mime_type:
                config_kwargs["response_mime_type"] = response_mime_type
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.APIError as e:
            raise RuntimeError(f"Gemini API request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e

        if response_schema is not None and getattr(response, "parsed", None) is not None:
            return json.dumps(response.parsed, ensure_ascii=False)

        raw_text = getattr(response, "text", "") or ""
        raw_text = raw_text.strip()
        raw_text = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        return raw_text


def gemini_generate(
    prompt: str,
    api_key: str,
    model_name: Optional[str] = None,
    timeout: float = DEFAULT_GEMINI_TIMEOUT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    response_mime_type: Optional[str] = None,
    response_schema: Optional[Any] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = GENERATION_SEED,
) -> str:
    """Send `prompt` to Gemini and return generated text.

    Decoding is greedy by default (temperature 0.0), and ``seed`` is sent
    whenever the installed google-genai release accepts one.
    """
    client = GeminiAPI(
        api_key=api_key,
        model_name=model_name,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        seed=seed,
    )
    return client.generate_with_config(
        prompt,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
    )
