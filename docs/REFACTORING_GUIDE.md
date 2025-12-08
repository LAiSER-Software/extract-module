# LAiSER Codebase Refactoring Guide
A thorough guide to the refactoring process undertaken in the LAiSER codebase.

Ownership:
----------
Project: Leveraging Artificial intelligence for Skills Extraction and Research (LAiSER)
Owner:  George Washington University Insitute of Public Policy
        Program on Skills, Credentials and Workforce Policy
        Media and Public Affairs Building
        805 21st Street NW
        Washington, DC 20052
        PSCWP@gwu.edu
        https://gwipp.gwu.edu/program-skills-credentials-workforce-policy-pscwp

License:
--------
Copyright 2024 George Washington University Insitute of Public Policy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
(the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Revision History:
-----------------
| Rev No. |     Date    |        Author       |  Description |
|---------|-------------|---------------------|--------------|
| 1.0.0   | 08/13/2025  | Phanindra Kumar K.  | Initial Version
"""

## Overview

This document outlines the comprehensive refactoring completed for the LAiSER (Leveraging Artificial Intelligence for Skills Extraction and Research) codebase. The refactoring improves code maintainability, scalability, and separation of concerns while introducing a new taxonomy-aware skill extraction pipeline.

## Refactoring Achievements

### ✅ Phase 1: Foundation Architecture (COMPLETED)

#### 1.1 Configuration Management
- **File**: `laiser/config.py`
- **Implementation**: Centralized all configuration constants including model IDs, URLs, batch sizes, prompts, and SCQF level descriptors
- **Benefits**: Single source of truth for configuration, easy environment management

#### 1.2 Exception Handling
- **File**: `laiser/exceptions.py`
- **Implementation**: Custom exceptions for better error handling (`LAiSERError`, `InvalidInputError`, etc.)
- **Benefits**: More specific error messages and cleaner error handling

#### 1.3 Data Access Layer
- **File**: `laiser/data_access.py`
- **Implementation**: 
  - `DataAccessLayer`: Manages data loading from URLs
  - `FAISSIndexManager`: Handles FAISS index operations
- **Benefits**: Separation of data concerns, reusable methods

#### 1.4 Service Layer
- **File**: `laiser/services.py`
- **Implementation**:
  - `PromptBuilder`: Builds prompts for different extraction tasks
  - `ResponseParser`: Parses LLM responses with robust JSON handling
  - `SkillAlignmentService`: Manages ESCO taxonomy alignment
  - `SkillExtractionService`: Main orchestrator for the extraction pipeline
- **Benefits**: Clear separation of business logic, reusable components

### ✅ Phase 2: Enhanced Model Management (COMPLETED)

#### 2.1 Improved Model Architecture
- **Files**: `laiser/llm_models/` directory with modular design
- **Implementation**: 
  - Enhanced error handling with fallback mechanisms
  - Support for multiple LLM providers (HuggingFace, Gemini, vLLM)
  - Consistent return values across model types

#### 2.2 LLM Integration
- **File**: `laiser/llm_methods.py`
- **Implementation**: Added missing imports, proper error handling, and `get_ksa_details()` for Knowledge/Skills/Abilities extraction

### ✅ Phase 3: New Skill Extraction Pipeline (COMPLETED)

#### 3.1 Taxonomy-Aware Extraction
- **Implementation**: Complete rewrite of skill extraction to be taxonomy-aware
- **New Output Format**: 7-column DataFrame with `Research ID`, `Description`, `Raw Skill`, `Knowledge Required`, `Task Abilities`, `Skill Tag`, `Correlation Coefficient`
- **Process**:
  1. FAISS semantic search against ESCO taxonomy
  2. LLM-based KSA (Knowledge, Skills, Abilities) extraction for each skill
  3. Structured output with skill tags and correlation scores

#### 3.2 Refactored Interface
- **File**: `laiser/skill_extractor_refactored.py`
- **Features**:
  - Clean initialization with multiple model support
  - `extract_skills()`: Single skill/text extraction
  - `extract_and_align()`: Batch processing with taxonomy alignment
  - Backward compatibility maintained
  - Comprehensive error handling

### ✅ Phase 4: Enhanced Functionality (COMPLETED)

#### 4.1 FAISS Integration
- **Implementation**: Instance-based FAISS index management with caching
- **Features**: Automatic index building, lazy loading, similarity-based skill retrieval

#### 4.2 Multi-Model Support
- **Implementation**: Support for Gemini API, HuggingFace models, and vLLM
- **Routing**: Intelligent model selection based on availability and configuration

## Current Implementation Status

### ✅ Completed Features
- **Modular Architecture**: Full separation of concerns achieved
- **Taxonomy Integration**: ESCO skills taxonomy fully integrated with FAISS indexing
- **Multi-Model Support**: Support for HuggingFace, Gemini API, and vLLM models
- **Enhanced Output Format**: New 7-column structured output with KSA details
- **Backward Compatibility**: Original API preserved in legacy `skill_extractor.py`
- **Error Handling**: Comprehensive exception handling throughout the pipeline
- **Configuration Management**: Centralized configuration with environment support

### 🚧 In Progress
- **Performance Optimization**: Batch processing for LLM calls to reduce costs
- **Testing Suite**: Comprehensive unit and integration tests
- **Documentation**: Updated examples and API documentation

## Usage Examples

### Standard Usage (Unchanged)
```python
import torch
import pandas as pd
from laiser.skill_extractor import Skill_Extractor

use_gpu = torch.cuda.is_available()
se = Skill_Extractor(use_gpu=use_gpu)

# Load data
job_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/jobs-data/linkedin_jobs_sample_36rows.csv')
job_sample = job_sample[['description', 'job_id']].head(3)

# Extract skills (returns new 7-column format)
output = se.extractor(job_sample, 'job_id', text_columns=['description'])
output.to_csv('extracted_skills.csv', index=False)
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
skills = extractor.extract_skills(text_data, method="ksa", input_type='job_desc')

# Batch extraction with alignment
df_skills = extractor.extract_and_align(
    data_frame, 
    id_column="job_id", 
    text_columns=["description"], 
    input_type='job_desc'
)
```

### Service Layer Usage
```python
from laiser.services import SkillExtractionService

service = SkillExtractionService()
aligned_skills = service.alignment_service.get_top_esco_skills(text)
```

## Output Format

Both legacy and refactored system provides the same structured 7-column output format:

| Column | Description |
|--------|-------------|
| **Research ID** | Original identifier from input data |
| **Description** | Source text/description |
| **Raw Skill** | Extracted skill aligned with ESCO taxonomy |
| **Knowledge Required** | Knowledge areas needed for the skill |
| **Task Abilities** | Tasks/capabilities enabled by the skill |
| **Skill Tag** | ESCO taxonomy reference (e.g., `ESCO.1234`) |
| **Correlation Coefficient** | Similarity score to taxonomy skill |

## Architecture Benefits

### 1. **Maintainability**
- Modular design with clear separation of concerns
- Smaller, focused classes with single responsibilities
- Comprehensive error handling and logging

### 2. **Scalability**
- Service layer architecture supports new extraction methods
- Plugin architecture ready for different LLM providers
- Batch processing capabilities for large datasets

### 3. **Performance**
- FAISS-based semantic search for fast taxonomy matching
- Lazy loading of models and indices
- Efficient resource management

### 4. **Flexibility**
- Support for multiple input types (job descriptions, syllabi)
- Configurable extraction methods (basic, KSA)
- Multiple model backends (HuggingFace, Gemini, vLLM)

## File Structure

```
laiser/
├── __init__.py
├── config.py                    # Configuration constants & prompts
├── exceptions.py               # Custom exceptions
├── data_access.py             # Data access & FAISS management
├── services.py                # Business logic services
├── skill_extractor.py         # Updated with new pipeline
├── skill_extractor_refactored.py  # New refactored interface
├── llm_methods.py            # LLM utility methods
├── params.py                 # Legacy parameters (deprecated)
├── utils.py                  # Utility functions
├── llm_models/
│   ├── __init__.py
│   ├── model_loader.py       # Enhanced model loading
│   ├── llm_router.py        # Multi-LLM routing
│   ├── gemini.py            # Gemini API integration
│   └── hugging_face_llm.py  # HuggingFace integration
└── public/                   # Public assets
    ├── combined.csv          # Combined skills data
    └── esco_faiss_index.index # FAISS index file
```

## Key Improvements Summary

1. **✅ Complete Modular Architecture**: Full separation of concerns implemented
2. **✅ Taxonomy-Aware Extraction**: ESCO taxonomy integrated with semantic search
3. **✅ Enhanced Output Format**: Structured 7-column output with KSA details
4. **✅ Multi-Model Support**: HuggingFace, Gemini, and vLLM integration
5. **✅ Backward Compatibility**: Existing API preserved and enhanced
6. **✅ Robust Error Handling**: Comprehensive exception management
7. **✅ Performance Optimization**: FAISS indexing and efficient processing

## Future Enhancements

### Short Term
- **Batch LLM Processing**: Reduce API costs by batching multiple skill analyses
- **Comprehensive Testing**: Unit and integration test suite
- **Documentation Updates**: API docs and usage examples
- **Performance Monitoring**: Metrics and benchmarking

### Long Term  
- **Async Processing**: Support for large-scale dataset processing
- **Cloud Integration**: Distributed vector storage and processing
- **API Service**: REST API wrapper for service deployment
- **Additional Taxonomies**: Support for domain-specific skill taxonomies

## Conclusion

The LAiSER refactoring has successfully transformed the codebase from a monolithic structure to a modern, modular architecture. The new system provides:

- **Enhanced functionality** with taxonomy-aware skill extraction
- **Better maintainability** through clear separation of concerns  
- **Improved scalability** with service-oriented architecture
- **Comprehensive output** with structured KSA analysis
- **Multiple model support** for different use cases and environments

The refactored system maintains full backward compatibility while enhancing functionality, performance, and maintainability.
