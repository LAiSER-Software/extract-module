# LAiSER Architecture Documentation

## Overview

LAiSER (Leveraging Artificial Intelligence for Skill Extraction & Research) is a tool designed to help learners, educators, and employers share trusted and mutually intelligible information about skills. The tool extracts skills from text data (such as job descriptions or course syllabi) and aligns them with established skill taxonomies.

**Recent Major Update**: The LAiSER codebase has undergone comprehensive refactoring to implement a modular, service-oriented architecture with enhanced taxonomy-aware skill extraction capabilities. This update maintains backward compatibility while introducing improved functionality and maintainability.

## Repository Structure

```
extract-module/
├── .github/workflows/           # GitHub Actions workflows
├── dev_space/                   # Development notebooks and experimental files
├── laiser/                      # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Global Configuration management (NEW - Centralized config)
│   ├── data_access.py           # Data access layer (NEW - Modular data handling)
│   ├── exceptions.py            # Custom exceptions (NEW - Structured error handling)
│   ├── llm_methods.py           # LLM interaction methods
│   ├── params.py                # Global parameters (LEGACY - Use config.py instead)
│   ├── services.py              # Service layer for business logic (NEW - Core services)
│   ├── skill_extractor_refactored.py # Refactored skill extraction functionality (NEW)
│   ├── skill_extractor.py       # Core skill extraction functionality (UPDATED)
│   ├── utils.py                 # Utility functions
│   ├── llm_models/              # LLM model implementations (ENHANCED)
│   │   ├── __init__.py          # LLM model package initialization
│   │   ├── gemini.py            # Gemini API integration (ENHANCED)
│   │   ├── hugging_face_llm.py # Hugging Face model integration (ENHANCED)
│   │   ├── llm_router.py        # Router for different LLM implementations (ENHANCED)
│   │   └── model_loader.py      # Model loading utilities (ENHANCED)
│   └── public/                  # Public data files
│       ├── combined.csv         # Combined taxonomy data
│       └── esco_faiss_index.index # FAISS index for ESCO skills
├── .gitignore                   # Git ignore file
├── ARCHITECTURE.md              # Architecture documentation (this file)
├── CONTRIBUTING.md              # Guidelines for contributing to the project
├── LICENSE.md                   # Project license
├── main.py                      # Main script for running the tool
├── paper.md                     # Research paper describing the project
├── pyproject.toml               # Python project configuration
├── README.md                    # Project README
├── REFACTORING_GUIDE.md         # Comprehensive refactoring documentation (NEW)
├── requirements.txt             # Package dependencies
└── setup.py                     # Package setup script
```

## Core Components

### 1. Refactored Architecture Overview

The LAiSER codebase has been refactored into a modular, service-oriented architecture that separates concerns and improves maintainability. The system now supports both legacy and new interfaces while providing enhanced functionality.

#### New Modular Components:

- **Configuration Layer** (`config.py`): Centralized configuration management
- **Data Access Layer** (`data_access.py`): Handles data loading and FAISS operations  
- **Service Layer** (`services.py`): Core business logic and processing
- **Exception Layer** (`exceptions.py`): Structured error handling
- **Refactored Interface** (`skill_extractor_refactored.py`): New clean API

### 2. Skill_Extractor Class (Legacy Interface)

The `Skill_Extractor` class in `skill_extractor.py` maintains backward compatibility while integrating the new taxonomy-aware pipeline.

#### Key Methods:

- **__init__**: Initializes the extractor with model ID, token, and GPU settings. Loads necessary models and data.
- **extract_raw**: Extracts raw skills from input text using LLM or SkillNer.
- **align_skills**: Aligns extracted skills with the taxonomy using cosine similarity.
- **extractor**: Main method that orchestrates the extraction and alignment process (NOW RETURNS 7-COLUMN FORMAT).
- **get_top_esco_skills**: Retrieves top ESCO skills based on semantic similarity.
- **build_faiss_index_esco**: Builds a FAISS index for ESCO skills.
- **load_faiss_index_esco**: Loads a FAISS index for ESCO skills.

### 3. SkillExtractorRefactored Class (New Interface)

The new `SkillExtractorRefactored` class provides a cleaner, more modular interface:

#### Key Methods:

- **extract_skills**: Single skill/text extraction with method selection
- **extract_and_align**: Batch processing with taxonomy alignment
- **get_top_esco_skills**: Enhanced ESCO skill retrieval

### 4. Service Layer Components

#### 4.1 PromptBuilder
- Builds prompts for different extraction tasks (job descriptions, syllabi)
- Supports KSA (Knowledge, Skills, Abilities) extraction prompts

#### 4.2 ResponseParser  
- Robust JSON parsing with fallback mechanisms
- Handles malformed LLM responses gracefully

#### 4.3 SkillAlignmentService
- Manages ESCO taxonomy alignment using FAISS
- Provides semantic similarity-based skill matching

#### 4.4 SkillExtractionService
- Main orchestrator for the extraction pipeline
- Coordinates between different service components

### 5. Multi-Model LLM Integration

The package supports multiple LLM implementations through an enhanced router pattern:

- **llm_router.py**: Enhanced routing with better error handling and fallback mechanisms.
- **gemini.py**: Improved integration with Google's Gemini API.
- **hugging_face_llm.py**: Enhanced integration with Hugging Face models and vLLM.
- **model_loader.py**: Improved utilities for loading models with better resource management.

### 6. Data Access Layer

The new `data_access.py` provides:

- **DataAccessLayer**: Manages data loading from remote URLs and local files
- **FAISSIndexManager**: Handles FAISS index operations with caching and lazy loading

### 7. Configuration Management

The `config.py` module centralizes:

- Model identifiers and URLs
- Prompt templates for different extraction types
- SCQF level descriptors
- Batch sizes and thresholds
- Default parameters


## Usage Examples

### Legacy Interface (Backward Compatible)
```python
from laiser.skill_extractor import Skill_Extractor
import pandas as pd

# Initialize the extractor
se = Skill_Extractor(AI_MODEL_ID="your_model_id", HF_TOKEN="your_hf_token", use_gpu=True)

# Load data
data = pd.read_csv("job_descriptions.csv")

# Extract skills (now returns enhanced 7-column format)
results = se.extractor(data, id_column="job_id", text_columns=["description"])

# Save results
results.to_csv("extracted_skills.csv", index=False)
```

### New Refactored Interface
```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize with specific model
extractor = SkillExtractorRefactored(
    model_id="gemini",
    api_key="your-api-key",
    use_gpu=False
)

# Single skill extraction
text_data = {"description": "Software engineer with Python experience"}
skills = extractor.extract_skills(text_data, method="ksa", input_type='job_desc')

# Batch extraction with alignment
df = pd.read_csv("data.csv")
df_skills = extractor.extract_and_align(
    df, 
    id_column="job_id", 
    text_columns=["description"], 
    input_type='job_desc'
)
```

### Service Layer Usage
```python
from laiser.services import SkillExtractionService, SkillAlignmentService

# Direct service usage
alignment_service = SkillAlignmentService()
top_skills = alignment_service.get_top_esco_skills(
    text="Machine learning engineer", 
    top_k=10
)

# Full extraction service
extraction_service = SkillExtractionService()
results = extraction_service.extract_and_align_skills(
    text_data={"description": "Data scientist role"},
    input_type="job_desc"
)
```
