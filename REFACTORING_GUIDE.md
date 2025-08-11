# LAiSER Codebase Refactoring Guide

## Overview

This document outlines the comprehensive refactoring plan for the LAiSER (Leveraging Artificial Intelligence for Skills Extraction and Research) codebase. The refactoring aims to improve code maintainability, scalability, and separation of concerns.

## Current Issues Identified

### 1. **Structural Issues**
- Large monolithic classes with multiple responsibilities
- Poor separation of concerns (business logic mixed with data access)
- Complex initialization in the main `Skill_Extractor` class
- Code duplication across modules

### 2. **Code Quality Issues**
- Missing imports and incomplete functions
- Inconsistent error handling
- Hard-coded values throughout the codebase
- Lack of proper validation

### 3. **Architectural Issues**
- No clear layered architecture
- Tight coupling between components
- Difficult to test individual components
- Hard to extend with new features

## Refactoring Strategy

### Phase 1: Create Foundation (COMPLETED)

#### 1.1 Configuration Management
- **File**: `laiser/config.py`
- **Purpose**: Centralize all configuration constants
- **Benefits**: 
  - Single source of truth for configuration
  - Easy to modify settings
  - Better environment-specific configurations

#### 1.2 Exception Handling
- **File**: `laiser/exceptions.py`
- **Purpose**: Define custom exceptions for better error handling
- **Benefits**:
  - More specific error messages
  - Better error recovery
  - Cleaner error handling code

#### 1.3 Data Access Layer
- **File**: `laiser/data_access.py`
- **Purpose**: Handle all data loading and external API calls
- **Components**:
  - `DataAccessLayer`: Manages data loading from URLs
  - `FAISSIndexManager`: Handles FAISS index operations
- **Benefits**:
  - Separation of data concerns
  - Reusable data access methods
  - Easier to mock for testing

#### 1.4 Service Layer
- **File**: `laiser/services.py`
- **Purpose**: Contains core business logic
- **Components**:
  - `PromptBuilder`: Builds prompts for different tasks
  - `ResponseParser`: Parses LLM responses
  - `SkillAlignmentService`: Handles skill alignment to taxonomies
  - `SkillExtractionService`: Main service orchestrator
- **Benefits**:
  - Clear separation of business logic
  - Reusable components
  - Easier to test and maintain

### Phase 2: Improve Model Management (COMPLETED)

#### 2.1 Enhanced Model Loading
- **File**: `laiser/llm_models/model_loader.py`
- **Improvements**:
  - Better error handling with custom exceptions
  - Fallback mechanisms for model loading
  - Consistent return values

#### 2.2 Fixed Missing Imports
- **File**: `laiser/llm_methods.py`
- **Improvements**:
  - Added missing imports
  - Proper error handling for optional dependencies

### Phase 3: Refactored Main Interface (COMPLETED)

#### 3.1 New Skill Extractor Class
- **File**: `laiser/skill_extractor_refactored.py`
- **Features**:
  - Clean initialization process
  - Separation of concerns
  - Backward compatibility with original API
  - Better error handling
  - More focused methods

## Implementation Steps

### Step 1: Gradual Migration
1. **Introduce new modules** alongside existing ones
2. **Update imports** to use new configuration and exceptions
3. **Test compatibility** with existing code
4. **Gradually migrate** functionality to new structure

### Step 2: Update Existing Code
1. **Modify `skill_extractor.py`** to use new components
2. **Update `llm_methods.py`** to use new configuration
3. **Refactor utility functions** in `utils.py`

### Step 3: Testing and Validation
1. **Unit tests** for individual components
2. **Integration tests** for the complete pipeline
3. **Performance tests** to ensure no regression
4. **Compatibility tests** with existing usage patterns

## Benefits of Refactoring

### 1. **Maintainability**
- Smaller, focused classes and methods
- Clear separation of responsibilities
- Easier to locate and fix bugs
- Better code documentation

### 2. **Scalability**
- Modular architecture allows adding new features easily
- Service layer can be extended for new extraction methods
- Data access layer can support new data sources

### 3. **Testability**
- Individual components can be tested in isolation
- Mock objects can be used for dependencies
- Better test coverage possible

### 4. **Reusability**
- Components can be reused across different parts of the application
- Service classes can be used in different contexts
- Data access methods are generic and reusable

### 5. **Performance**
- Better resource management
- Lazy loading of components
- More efficient error handling

## Migration Guide

### For Existing Users

#### Simple Usage (No changes required)
```python
import torch
import pandas as pd
import argparse
from laiser.skill_extractor import Skill_Extractor

use_gpu = True if torch.cuda.is_available() == True else False

se = Skill_Extractor(use_gpu=use_gpu)
print('The Skill Extractor has been initialized successfully!\n')

# To extract skills from a text
# Skill extraction from jobs data
print('\n\nLoading a sample dataset of 50 jobs...')
job_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/jobs-data/linkedin_jobs_sample_36rows.csv')
print('The sample jobs dataset has been loaded successfully!\n')

job_sample = job_sample[['description', 'job_id']]
job_sample = job_sample[1:3]
print('The sample dataset has been filtered successfully!\n')
print('Head of the sample:\n', job_sample.head())

output = se.extractor(job_sample, 'job_id', text_columns=['description'])
print('The skills have been extracted from jobs data successfully...\n')

# Save the extracted skills to a CSV file
print(output)
file_name = f'extracted_skills_for_{len(job_sample)}Jobs.csv'
output.to_csv(file_name, index=False)
print('The extracted skills have been saved to the file named:', file_name)
```

#### Advanced Usage (Optional improvements)
```python
# New way with better configuration
from laiser.skill_extractor_refactored import SkillExtractorRefactored

import torch
import pandas as pd
import argparse

use_gpu = True if torch.cuda.is_available() == True else False
extractor = SkillExtractorRefactored(
    model_id="your-model",
    use_gpu=use_gpu
)
print('The Skill Extractor has been initialized successfully!\n')

# To extract skills from a text
skills = extractor.extract_skills(text, method="ksa")
aligned = extractor.align_skills(skills)
```

### For Developers

#### Using Service Layer Directly
```python
from laiser.services import SkillExtractionService

service = SkillExtractionService()
esco_skills = service.alignment_service.get_top_esco_skills(text)
```

#### Using Data Access Layer
```python
from laiser.data_access import DataAccessLayer

dal = DataAccessLayer()
esco_df = dal.load_esco_skills()
```

## Next Steps

### Phase 4: Complete Migration (RECOMMENDED)
1. **Update main `skill_extractor.py`** to use new architecture
2. **Create comprehensive tests**
3. **Update documentation**
4. **Performance optimization**

### Phase 5: Advanced Features (FUTURE)
1. **Plugin architecture** for different extraction methods
2. **Caching layer** for better performance
3. **Async processing** for large datasets
4. **API wrapper** for service deployment

## File Structure After Refactoring

```
laiser/
├── __init__.py
├── config.py                    # Configuration constants
├── exceptions.py               # Custom exceptions
├── data_access.py             # Data access layer
├── services.py                # Business logic services
├── skill_extractor.py         # Original class (to be updated)
├── skill_extractor_refactored.py  # New refactored class
├── llm_methods.py            # LLM utility methods
├── params.py                 # Legacy parameters (to be deprecated)
├── utils.py                  # Utility functions
├── llm_models/
│   ├── __init__.py
│   ├── model_loader.py       # Improved model loading
│   ├── llm_router.py        # LLM routing logic
│   ├── gemini.py            # Gemini API integration
│   └── hugging_face_llm.py  # HuggingFace model integration
└── public/                   # Public assets
    └── esco_faiss_index.index
```

## Conclusion

This refactoring provides a solid foundation for the LAiSER project with improved maintainability, testability, and extensibility. The new architecture follows software engineering best practices while maintaining backward compatibility for existing users.

The modular design allows for easier debugging, testing, and feature additions. Each component has a clear responsibility, making the codebase more understandable and maintainable for future development.
