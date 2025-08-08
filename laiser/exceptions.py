"""
Custom exceptions for LAiSER project
"""

class LAiSERError(Exception):
    """Base exception class for LAiSER project"""
    pass

class ModelLoadError(LAiSERError):
    """Raised when model loading fails"""
    pass

class VLLMNotAvailableError(LAiSERError):
    """Raised when vLLM is required but not available"""
    pass

class SkillExtractionError(LAiSERError):
    """Raised when skill extraction fails"""
    pass

class FAISSIndexError(LAiSERError):
    """Raised when FAISS index operations fail"""
    pass

class InvalidInputError(LAiSERError):
    """Raised when input validation fails"""
    pass
