"""
LLM Models package for LAiSER

This package contains modules for loading and managing different LLM models.
"""

# Import main functions for easier access
try:
    from .model_loader import load_model_from_vllm, load_model_from_transformer
    from .llm_router import llm_router
except ImportError:
    # Handle cases where dependencies might not be available
    pass

__all__ = [
    'load_model_from_vllm',
    'load_model_from_transformer', 
    'llm_router'
]
