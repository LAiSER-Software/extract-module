# LAiSER Documentation

## Leveraging Artificial Intelligence for Skill Extraction & Research

![LAiSER Logo](https://i.imgur.com/XznvjNi.png)

Welcome to the LAiSER documentation. This repository contains comprehensive documentation for the LAiSER (Leveraging Artificial Intelligence for Skill Extraction & Research) project.

## Documentation Files

This documentation is organized into several files, each serving a different purpose:

1. [**Quick Start Guide**](LAiSER_QuickStart.md) - Get up and running with LAiSER quickly
2. [**User Documentation**](LAiSER_Documentation.md) - Comprehensive user guide and reference
3. [**Technical Reference**](LAiSER_Technical_Reference.md) - Detailed technical information for developers

## About LAiSER

LAiSER is a Python package designed to extract and analyze skills from textual data such as job descriptions and course syllabi. It uses advanced natural language processing and machine learning techniques to identify skills and align them with established taxonomies, providing a standardized way to understand and communicate skill requirements across different stakeholders.

### Key Features

- **Skill Extraction**: Extract skills from job descriptions and syllabi using LLMs
- **Taxonomy Alignment**: Map extracted skills to standardized taxonomies
- **Knowledge & Task Analysis**: Identify knowledge requirements and task abilities for each skill
- **Multi-Model Support**: Use different LLM backends (Gemini, HuggingFace, vLLM)
- **GPU Acceleration**: Optimize performance with GPU support
- **Batch Processing**: Process large datasets efficiently

## Installation

LAiSER can be installed using pip:

### For GPU Support (Recommended)

```bash
pip install laiser[gpu]
```

### For CPU-Only Environments

```bash
pip install laiser[cpu]
```

## Quick Example

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize the skill extractor
extractor = SkillExtractorRefactored()

# Create a simple dataset with job descriptions
data = pd.DataFrame([
    {
        "job_id": "job_001",
        "description": "Data Scientist Position: Requires Python, SQL, and machine learning experience."
    }
])

# Extract and align skills
results = extractor.extract_and_align(
    data=data,
    id_column='job_id',
    text_columns=['description']
)

# View results
print(results)
```

## Project Information

- **License**: BSD 3-Clause License
- **Version**: 0.3.0
- **Repository**: [GitHub](https://github.com/LAiSER-Software/extract-module)
- **Authors**: LAiSER Team (PSCWP@gwu.edu)
- **Organization**: George Washington University Institute of Public Policy

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/LAiSER-Software/extract-module) or contact the LAiSER team at PSCWP@gwu.edu.
