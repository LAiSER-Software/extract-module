"""
LAiSER - Leveraging Artificial Intelligence for Skills Extraction and Research

A Python package for extracting and aligning skills from text using AI models.
"""

__version__ = "1.0"

# Import main classes for easy access
try:
    from .extractor import SkillExtractorRefactored

    # Make both available
    __all__ = ["SkillExtractor", "SkillExtractorRefactored"]
except ImportError as e:
    # Handle cases where dependencies might not be available
    import warnings

    warnings.warn(f"Could not import LAiSER components: {e}")
    __all__ = []

# For backward compatibility
try:
    # Make the refactored version available as an alias
    SkillExtractor = SkillExtractorRefactored
except NameError:
    pass
