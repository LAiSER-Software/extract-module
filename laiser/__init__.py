"""
LAiSER - Leveraging Artificial Intelligence for Skills Extraction and Research

A Python package for extracting and aligning skills from text using AI models.
"""

__version__ = "0.2.32"

# Import main classes for easy access
try:
    from .skill_extractor import Skill_Extractor
    __all__ = ['Skill_Extractor']
except ImportError:
    # Handle cases where dependencies might not be available
    __all__ = []