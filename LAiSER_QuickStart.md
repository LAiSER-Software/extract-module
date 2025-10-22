# LAiSER Quick Start Guide

## Leveraging Artificial Intelligence for Skill Extraction & Research

This quick start guide will help you get up and running with LAiSER for skill extraction and alignment.

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

## Basic Usage

### Extracting Skills from Job Descriptions

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize the skill extractor
extractor = SkillExtractorRefactored()

# Create a simple dataset with job descriptions
data = pd.DataFrame([
    {
        "job_id": "job_001",
        "description": """
        Data Scientist Position
        
        Requirements:
        - 3+ years of experience in data science
        - Proficiency in Python, SQL, and machine learning
        - Experience with data visualization tools
        - Strong communication skills
        """
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

### Using Gemini API

For better results, you can use the Gemini API:

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

# Initialize with Gemini
extractor = SkillExtractorRefactored(
    model_id="gemini",
    api_key="your_gemini_api_key"  # Replace with your actual API key
)

# Process data
results = extractor.extract_and_align(
    data=your_dataframe,
    id_column='id',
    text_columns=['description']
)
```

### Processing Syllabi

LAiSER can also extract skills from educational syllabi:

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

extractor = SkillExtractorRefactored()

# Create a simple dataset with course syllabi
syllabi_data = pd.DataFrame([
    {
        "id": "course_001",
        "description": "This course introduces students to the fundamentals of data science...",
        "learning_outcomes": "Upon completion, students will be able to: 1) Apply statistical methods to analyze data, 2) Create data visualizations, 3) Build predictive models..."
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

## Understanding the Output

LAiSER returns a DataFrame with the following columns:

- **Research ID**: Original identifier from input data
- **Description**: Source text/description
- **Raw LLM Skill**: Extracted skill from the LLM
- **Taxonomy Skill**: Skill aligned with the taxonomy
- **Correlation Coefficient**: Similarity score between extracted and taxonomy skill

Example output:

```
  Research ID                                        Description        Raw LLM Skill         Taxonomy Skill  Correlation Coefficient
0     job_001  Data Scientist Position\n\nRequirements:\n- 3+ ...       data science            data science                   0.92
1     job_001  Data Scientist Position\n\nRequirements:\n- 3+ ...              Python                  Python                   0.95
2     job_001  Data Scientist Position\n\nRequirements:\n- 3+ ...                 SQL                     SQL                   0.97
3     job_001  Data Scientist Position\n\nRequirements:\n- 3+ ...    machine learning        machine learning                   0.94
4     job_001  Data Scientist Position\n\nRequirements:\n- 3+ ...  data visualization    data visualization                   0.91
```

## Common Options

### Batch Size

For large datasets, you can adjust the batch size:

```python
results = extractor.extract_and_align(
    data=large_dataframe,
    id_column='id',
    text_columns=['description'],
    batch_size=64  # Process 64 items at a time
)
```

### Top K Skills

Limit the number of skills returned per document:

```python
results = extractor.extract_and_align(
    data=data,
    id_column='id',
    text_columns=['description'],
    top_k=10  # Return top 10 skills per document
)
```

### Skill Levels

Include skill levels in the output:

```python
results = extractor.extract_and_align(
    data=data,
    id_column='id',
    text_columns=['description'],
    levels=True  # Include skill levels
)
```

## Working with Large Files

For large datasets, you can process them in chunks:

```python
import pandas as pd
from laiser.skill_extractor_refactored import SkillExtractorRefactored

# Initialize extractor
extractor = SkillExtractorRefactored()

# Read large CSV in chunks
chunk_size = 100
results = []

for chunk in pd.read_csv("large_dataset.csv", chunksize=chunk_size):
    chunk_results = extractor.extract_and_align(
        data=chunk,
        id_column='id',
        text_columns=['description']
    )
    results.append(chunk_results)

# Combine results
final_results = pd.concat(results, ignore_index=True)

# Save to CSV
final_results.to_csv("extracted_skills.csv", index=False)
```

## Troubleshooting

### GPU Issues

If you encounter GPU memory issues:

1. Reduce batch size:
   ```python
   results = extractor.extract_and_align(data, batch_size=16)
   ```

2. Use CPU mode:
   ```python
   extractor = SkillExtractorRefactored(use_gpu=False)
   ```

3. Use Gemini API:
   ```python
   extractor = SkillExtractorRefactored(model_id="gemini", api_key="your_api_key")
   ```

### Missing Dependencies

If you encounter missing dependencies:

```bash
# For GPU support
pip install vllm torch

# For CPU support
pip install torch transformers

# For FAISS
pip install faiss-cpu  # or faiss-gpu
```

### FAISS Index Issues

If FAISS index fails to load:

```python
# Force rebuild the index
extractor.skill_service.faiss_manager.initialize_index(force_rebuild=True)
```

## Next Steps

- Check the full [LAiSER Documentation](LAiSER_Documentation.md) for detailed information
- Explore the [Technical Reference](LAiSER_Technical_Reference.md) for advanced usage
- Visit the [GitHub repository](https://github.com/LAiSER-Software/extract-module) for updates
