# LAiSER Documentation

## Leveraging Artificial Intelligence for Skill Extraction & Research

![LAiSER Logo](https://i.imgur.com/XznvjNi.png)

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API Reference](#api-reference)
7. [Core Components](#core-components)
8. [Workflow](#workflow)
9. [Advanced Features](#advanced-features)
10. [Development Guide](#development-guide)
11. [License](#license)

## Introduction

LAiSER (Leveraging Artificial Intelligence for Skill Extraction & Research) is a Python package designed to extract and analyze skills from textual data such as job descriptions and course syllabi. It uses advanced natural language processing and machine learning techniques to identify skills and align them with established taxonomies, providing a standardized way to understand and communicate skill requirements across different stakeholders.

### Purpose

LAiSER helps learners, educators, and employers share trusted and mutually intelligible information about skills by:

- Automatically extracting skills from job descriptions and educational content
- Aligning extracted skills with established taxonomies (e.g., ESCO)
- Providing detailed knowledge and task abilities associated with each skill
- Enabling standardized skill analysis across different sectors

### Target Audience

- **Researchers**: Analyzing skill trends and requirements across industries
- **Educators**: Aligning curriculum with industry needs
- **Employers**: Standardizing job descriptions and skill requirements
- **Workforce Developers**: Understanding skill gaps and training needs
- **Data Scientists**: Building applications on top of skill extraction capabilities

## Project Overview

LAiSER is developed by the George Washington University Institute of Public Policy through the Program on Skills, Credentials and Workforce Policy. The project aims to bridge the gap between education and employment by providing a common language for skills.

### Key Features

- **Skill Extraction**: Extract skills from job descriptions and syllabi using LLMs
- **Taxonomy Alignment**: Map extracted skills to standardized taxonomies
- **Knowledge & Task Analysis**: Identify knowledge requirements and task abilities for each skill
- **Multi-Model Support**: Use different LLM backends (Gemini, HuggingFace, vLLM)
- **GPU Acceleration**: Optimize performance with GPU support
- **Batch Processing**: Process large datasets efficiently

### Technical Highlights

- Modular architecture with clear separation of concerns
- FAISS-based semantic search for fast taxonomy matching
- Support for multiple LLM providers
- Comprehensive error handling
- Backward compatibility with previous versions

## Architecture

LAiSER follows a modular architecture with clear separation of concerns:

```
laiser/
├── __init__.py                  # Package initialization
├── config.py                    # Configuration constants & prompts
├── exceptions.py                # Custom exceptions
├── data_access.py               # Data access & FAISS management
├── services.py                  # Business logic services
├── skill_extractor.py           # Legacy skill extractor (backward compatibility)
├── skill_extractor_refactored.py # New refactored interface
├── llm_methods.py               # LLM utility methods
├── params.py                    # Legacy parameters (deprecated)
├── utils.py                     # Utility functions
├── llm_models/                  # LLM integration modules
│   ├── __init__.py
│   ├── model_loader.py          # Enhanced model loading
│   ├── llm_router.py            # Multi-LLM routing
│   ├── gemini.py                # Gemini API integration
│   └── hugging_face_llm.py      # HuggingFace integration
└── public/                      # Public assets
    ├── combined.csv             # Combined skills data
    ├── skills_df.json           # Skills data in JSON format
    ├── skills_v03.index         # FAISS index for skills
    └── esco_faiss_index.index   # FAISS index for ESCO skills
```

### Component Layers

1. **Interface Layer**: `skill_extractor.py` and `skill_extractor_refactored.py`
2. **Service Layer**: `services.py` (business logic)
3. **Data Access Layer**: `data_access.py` (data management)
4. **Model Layer**: `llm_models/` (LLM integration)
5. **Utility Layer**: `utils.py`, `config.py`, `exceptions.py`

## Installation

LAiSER can be installed using pip with different options depending on your needs:

### For GPU Support (Recommended)

```bash
pip install laiser[gpu]
```

### For CPU-Only Environments

```bash
pip install laiser[cpu]
```

### Development Installation

```bash
git clone https://github.com/LAiSER-Software/extract-module.git
cd extract-module
pip install -e ".[dev]"
```

### Requirements

- Python 3.9 or later (3.12 recommended)
- For GPU support: CUDA-capable GPU with at least 15GB video memory
- Dependencies: torch, transformers, pandas, numpy, sentence-transformers, faiss, spacy, and more

## Usage

### Basic Usage

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize the skill extractor
extractor = SkillExtractorRefactored(
    model_id="your_model_id",  # e.g., "microsoft/DialoGPT-medium"
    hf_token="your_hf_token",  # For accessing gated HuggingFace models
    use_gpu=True  # Set to False for CPU-only environments
)

# Load your data
data = pd.DataFrame([
    {
        "job_id": "job_001",
        "description": "Your job description text here..."
    }
])

# Extract and align skills
results = extractor.extract_and_align(
    data=data,
    id_column='job_id',
    text_columns=['description'],
    input_type='job_desc'
)

# View results
print(results)
```

### Using Gemini API

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize with Gemini
extractor = SkillExtractorRefactored(
    model_id="gemini",
    api_key="your_gemini_api_key",
    use_gpu=False  # Not needed for API-based models
)

# Process data
results = extractor.extract_and_align(
    data=your_dataframe,
    id_column='id',
    text_columns=['description']
)
```

### Processing Syllabi

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

extractor = SkillExtractorRefactored()

# Load syllabi data
syllabi_data = pd.DataFrame([
    {
        "id": "course_001",
        "description": "Course description text...",
        "learning_outcomes": "Learning outcomes text..."
    }
])

# Extract and align skills from syllabi
results = extractor.extract_and_align(
    data=syllabi_data,
    id_column='id',
    text_columns=['description', 'learning_outcomes'],
    input_type='syllabus'
)
```

## API Reference

### SkillExtractorRefactored

The main class for skill extraction and alignment.

#### Constructor

```python
SkillExtractorRefactored(
    model_id: Optional[str] = None, 
    hf_token: Optional[str] = None,
    api_key: Optional[str] = None, 
    use_gpu: Optional[bool] = None
)
```

- `model_id`: Model ID for the LLM (e.g., HuggingFace model ID or "gemini")
- `hf_token`: HuggingFace token for accessing gated repositories
- `api_key`: API key for external services (e.g., Gemini)
- `use_gpu`: Whether to use GPU for model inference (defaults to `torch.cuda.is_available()`)

#### Methods

##### extract_and_align

```python
extract_and_align(
    data: pd.DataFrame,
    id_column: str = 'Research ID',
    text_columns: List[str] = None,
    input_type: str = "job_desc",
    top_k: Optional[int] = None,
    levels: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warnings: bool = True
) -> pd.DataFrame
```

Extracts and aligns skills from a dataset.

- `data`: Input dataset
- `id_column`: Column name for document IDs
- `text_columns`: Column names containing text data (defaults to `["description"]`)
- `input_type`: Type of input data ("job_desc" or "syllabus")
- `top_k`: Number of top skills to return
- `levels`: Whether to extract skill levels
- `batch_size`: Batch size for processing
- `warnings`: Whether to show warnings

Returns a DataFrame with extracted and aligned skills.

##### align_skills

```python
align_skills(
    raw_skills: List[str], 
    document_id: str = '0', 
    description: str = ''
) -> pd.DataFrame
```

Aligns raw skills to taxonomy.

- `raw_skills`: List of raw extracted skills
- `document_id`: Document identifier
- `description`: Full description text for context

Returns a DataFrame with aligned skills and similarity scores.

### Legacy API (Skill_Extractor)

For backward compatibility, the original `Skill_Extractor` class is still available:

```python
from laiser.skill_extractor import Skill_Extractor

extractor = Skill_Extractor(
    AI_MODEL_ID="your_model_id",
    HF_TOKEN="your_hf_token",
    use_gpu=True
)

results = extractor.extractor(
    data=your_dataframe,
    id_column='id',
    text_columns=['description'],
    input_type='job_desc',
    batch_size=32
)
```

## Core Components

### 1. Skill Extraction Service

The `SkillExtractionService` class in `services.py` handles the core business logic for skill extraction:

- `extract_skills_basic`: Simple skill extraction from text
- `extract_skills_with_ksa`: Extract skills with Knowledge, Skills, Abilities details
- `get_skill_details`: Get detailed KSA information for a specific skill
- `align_extracted_skills`: Align extracted skills to taxonomy

### 2. FAISS Index Manager

The `FAISSIndexManager` class in `data_access.py` manages FAISS indices for efficient skill similarity search:

- `initialize_index`: Load or build FAISS index
- `search_similar_skills`: Find similar skills using vector similarity

### 3. LLM Router

The `llm_router` function in `llm_models/llm_router.py` routes requests to the appropriate LLM implementation:

- Supports Gemini API via `gemini_generate`
- Supports HuggingFace models via `llm_generate_vllm`

### 4. Prompt Builder

The `PromptBuilder` class in `services.py` constructs prompts for different extraction tasks:

- `build_skill_extraction_prompt`: For basic skill extraction
- `build_ksa_extraction_prompt`: For KSA (Knowledge, Skills, Abilities) extraction
- `build_ksa_details_prompt`: For detailed KSA information

## Workflow

The typical workflow for skill extraction and alignment follows these steps:

1. **Initialization**: Load models, taxonomies, and FAISS indices
2. **Text Processing**: Clean and prepare input text
3. **Skill Extraction**: Extract skills using LLM
4. **Taxonomy Alignment**: Align skills to taxonomy using FAISS
5. **KSA Analysis**: Extract knowledge and task abilities for each skill
6. **Result Compilation**: Format results into structured output

### Extraction Process

1. The input text is preprocessed to remove irrelevant information
2. A prompt is generated for the LLM to extract skills
3. The LLM response is parsed to extract skill names
4. Each skill is aligned with the taxonomy using FAISS similarity search
5. For each aligned skill, knowledge and task abilities are extracted
6. Results are compiled into a structured DataFrame

## Advanced Features

### 1. Multi-Model Support

LAiSER supports multiple LLM backends:

- **Gemini API**: Cloud-based inference using Google's Gemini models
- **HuggingFace Transformers**: Local inference using various transformer models
- **vLLM**: Optimized inference for faster processing

### 2. FAISS Semantic Search

LAiSER uses FAISS (Facebook AI Similarity Search) for efficient similarity search:

- Vector embeddings for skills using SentenceTransformers
- Indexed search for fast retrieval
- Cosine similarity for matching extracted skills to taxonomy

### 3. Batch Processing

For large datasets, LAiSER supports batch processing:

- Process multiple inputs in batches
- Configurable batch size
- Progress tracking with tqdm

### 4. Error Handling

Comprehensive error handling with custom exceptions:

- `LAiSERError`: Base exception class
- `ModelLoadError`: For model loading failures
- `VLLMNotAvailableError`: When vLLM is required but not available
- `SkillExtractionError`: For skill extraction failures
- `FAISSIndexError`: For FAISS index operations failures
- `InvalidInputError`: For input validation failures

## Development Guide

### Project Structure

The LAiSER codebase follows a modular architecture with clear separation of concerns:

- **Interface Layer**: Public API for users
- **Service Layer**: Business logic and orchestration
- **Data Access Layer**: Data loading and management
- **Model Layer**: LLM integration and routing
- **Utility Layer**: Helper functions and configuration

### Contributing

Contributions to LAiSER are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Commit Guidelines

LAiSER follows the Conventional Commits specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

Types include:
- `feat`: A new feature
- `fix`: A bug fix
- `chore`: Routine tasks or maintenance
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code changes that neither fix a bug nor add a feature
- `test`: Adding or correcting tests

### Refactoring

The LAiSER codebase has undergone significant refactoring to improve maintainability and scalability. Key improvements include:

- Complete modular architecture with separation of concerns
- Taxonomy-aware extraction with FAISS semantic search
- Enhanced output format with KSA details
- Multi-model support for different LLM backends
- Robust error handling throughout the pipeline

For more details, see [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md).

## License

LAiSER is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full license text.

Copyright (c) 2025, LAiSER.
All rights reserved.
