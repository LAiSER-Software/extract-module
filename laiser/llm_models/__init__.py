"""
LLM Models package for LAiSER

This package contains various LLM model implementations and utilities.
"""

import warnings

GeminiAPI = None
gemini_generate = None
llm_generate = None
llm_generate_vllm = None
LlamaCppBackend = None
LLMRouter = None
load_model_from_transformer = None
load_model_from_vllm = None

try:
    from .gemini import GeminiAPI, gemini_generate
except ImportError as e:
    warnings.warn(f"Gemini dependencies are not available: {e}")

try:
    from .hugging_face_llm import llm_generate, llm_generate_vllm
except ImportError as e:
    warnings.warn(f"HuggingFace LLM dependencies are not available: {e}")

try:
    from .llama_cpp_handler import LlamaCppBackend
except ImportError as e:
    warnings.warn(f"llama.cpp dependencies are not available: {e}")

try:
    from .llm_router import LLMRouter
    from .model_loader import load_model_from_transformer, load_model_from_vllm
except ImportError as e:
    warnings.warn(f"Core LLM router dependencies are not available: {e}")

__all__ = [
    "load_model_from_vllm",
    "load_model_from_transformer",
    "LLMRouter",
    "GeminiAPI",
    "gemini_generate",
    "llm_generate",
    "llm_generate_vllm",
    "LlamaCppBackend",
]
