import gc
import inspect
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from laiser.config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, GENERATION_SEED

try:
    from llama_cpp import Llama  # type: ignore
except ImportError:  # pragma: no cover
    Llama = None


def _strip_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # mimic gemini.py behavior
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    return text


def _llama_accepts_seed(llama: Any) -> bool:
    """Report whether this llama-cpp-python build accepts a per-call seed.

    Older releases have no ``seed`` argument on ``create_chat_completion``;
    passing one there raises TypeError, so check the signature first.
    """
    try:
        return "seed" in inspect.signature(llama.create_chat_completion).parameters
    except (TypeError, ValueError):
        return False


class LlamaCppBackend:
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        temperature: float = DEFAULT_TEMPERATURE,
        chat_format: str = "chatml",
        seed: Optional[int] = GENERATION_SEED,
    ):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Install it to use the llama_cpp backend.")

        model_path = model_path or os.getenv("LAISER_LLAMA_CPP_MODEL_PATH")
        if not model_path:
            raise ValueError("Set LAISER_LLAMA_CPP_MODEL_PATH or pass model_path to LlamaCppBackend.")

        model_path = str(Path(model_path))
        # model_path = str(Path(model_path).expanduser().resolve())
        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        self.temperature = temperature
        self.seed = seed

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or None,
            n_gpu_layers=n_gpu_layers,
            # logits_all=False,
            # chat_format=chat_format,
        )

    def close(self) -> None:
        """Free llama.cpp resources early (avoid interpreter-shutdown __del__ errors)."""
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        try:
            llm.close()
        finally:
            self.llm = None
            gc.collect()

    def __enter__(self) -> "LlamaCppBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Avoid noisy exceptions during interpreter shutdown.
            pass

    def generate(
        self,
        prompt: str,
        system: str = "You are a helpful assistant that outputs in JSON.",
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        temperature = self.temperature if temperature is None else temperature
        seed = self.seed if seed is None else seed
        return llama_cpp_chat(
            prompt,
            self.llm,
            system=system,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            seed=seed,
        )


def llama_cpp_chat(
    prompt: str,
    llama: Any,
    *,
    system: str = "You are a helpful assistant that outputs in JSON.",
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    seed: Optional[int] = GENERATION_SEED,
) -> str:
    """Generate a chat completion with a local llama.cpp model.

    Decoding is greedy by default (temperature 0.0), and ``seed`` is sent
    whenever the installed llama-cpp-python accepts one.
    """
    if llama is None:
        raise ValueError("llama is None; expected an initialized llama_cpp.Llama instance.")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    kwargs = {"messages": messages}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop
    if temperature is not None:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = DEFAULT_TOP_P
    if seed is not None and _llama_accepts_seed(llama):
        kwargs["seed"] = seed

    resp = llama.create_chat_completion(**kwargs)
    return _strip_fences(resp["choices"][0]["message"]["content"])
