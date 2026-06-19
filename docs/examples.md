# LAiSER Examples

This file contains copy-paste examples for the current refactored API.

## 1. Install From PyPI

```bash
pip install laiser
```

GPU-enabled install:

```bash
pip install "laiser[gpu]"
```

## 2. Skills Only From a Job Description

```python
import os
import pandas as pd

from laiser.skill_extractor_refactored import SkillExtractorRefactored

data = pd.DataFrame(
    [
        {
            "Research ID": "job-001",
            "description": "Design scalable Python services and build analytics dashboards.",
        }
    ]
)

extractor = SkillExtractorRefactored(
    model_id="gemini",
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    use_gpu=False,
)

results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
    concepts=["skills"],
    top_k=5,
)

print(results)
```

## 3. Skills, Knowledge, and Tasks From a Job Description

```python
results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
    concepts=["skills", "knowledge", "tasks"],
    top_k=5,
    allowed_sources=["esco", "onet"],
)
```

## 4. Course Syllabus Input

```python
syllabus_df = pd.DataFrame(
    [
        {
            "Research ID": "course-001",
            "description": "Foundations of machine learning and data analysis.",
            "learning_outcomes": "Train models, interpret metrics, and communicate results.",
        }
    ]
)

results = extractor.extract_concepts(
    data=syllabus_df,
    id_column="Research ID",
    text_columns=["description", "learning_outcomes"],
    input_type="course_syllabi",
    concepts=["skills"],
)
```

Accepted syllabus aliases:

- `syllabus`
- `course_syllabus`
- `course_syllabi`

## 5. Restrict Taxonomy Sources

```python
results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
    concepts=["skills"],
    allowed_sources=["esco"],
)
```

## 6. Write Output to CSV

```python
results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    concepts=["skills", "knowledge", "tasks"],
    output_csv_path="alignment_results.csv",
)
```

## 7. Return Graph Edges

```python
graph = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    concepts=["skills", "knowledge", "tasks"],
    return_edges=True,
)

print(graph["nodes"].head())
print(graph["edges"].head())
```

## 8. Backward-Compatible Skills Wrapper

```python
results = extractor.extract_and_align(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
)
```

## 9. Common Flags

Constructor:

- `model_id`
- `hf_token`
- `api_key`
- `use_gpu`
- `backend`

Extraction:

- `input_type`
- `top_k`
- `similarity_threshold`
- `similarity_thresholds`
- `warnings`
- `allowed_sources`
- `concepts`
- `return_edges`
- `timing`
- `output_csv_path`
