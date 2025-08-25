"""
LLM Models package for LAiSER

This package contains various LLM model implementations and utilities.
"""

# Import key components for easier access
try:
    from .model_loader import load_model_from_vllm, load_model_from_transformer
    from .llm_router import llm_router
    from .gemini import GeminiAPI
    from .hugging_face_llm import HuggingFaceLLM
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some LLM model dependencies are not available: {e}")

__all__ = [
    'load_model_from_vllm',
    'load_model_from_transformer', 
    'llm_router',
    'GeminiAPI',
    'HuggingFaceLLM'
]