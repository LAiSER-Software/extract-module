# Quickstart

This walks through one complete extraction. Pick a model below — the first needs nothing but the package.

## 1. Prepare your input

LAiSER takes a pandas `DataFrame` with an ID column and one or more text columns:

```python
import pandas as pd

data = pd.DataFrame(
    [
        {
            "Research ID": "job-001",
            "description": "Build production machine learning systems in Python. "
            "Deploy models with Docker and monitor them on AWS.",
        }
    ]
)
```

## 2. Create an extractor

=== "Local model (no API key)"

    Runs on CPU. The model downloads on first use.

    ```python
    from laiser.skill_extractor_refactored import SkillExtractorRefactored

    extractor = SkillExtractorRefactored(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        use_gpu=False,
    )
    ```

=== "Gemini"

    ```python
    import os

    from laiser.skill_extractor_refactored import SkillExtractorRefactored

    extractor = SkillExtractorRefactored(
        model_id="gemini",
        api_key=os.environ["GEMINI_API_KEY"],
    )
    ```

=== "OpenAI"

    ```python
    from laiser.skill_extractor_refactored import SkillExtractorRefactored

    extractor = SkillExtractorRefactored(model_id="openai")  # reads OPENAI_API_KEY
    ```

Other options, including GPU and llama.cpp, are on [Model providers](providers.md).

## 3. Extract and align

```python
results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
    concepts=["skills"],
)

print(results[["Raw Concept", "Taxonomy Concept", "Taxonomy Source", "Correlation Coefficient"]])
```

## 4. Read the results

`results` is a `DataFrame` with one row per taxonomy match:

| Column | Meaning |
|---|---|
| `Research ID` | the document the match came from |
| `Type` | skill, knowledge or task |
| `Raw Concept` | the phrase the model extracted from the text |
| `Taxonomy Concept` | the taxonomy entry it matched |
| `Taxonomy Description` | that entry's description |
| `Taxonomy Source` | which taxonomy: `esco`, `onet`, `osn` or `ukos` |
| `Source Url` | link to the entry, where the taxonomy provides one |
| `Correlation Coefficient` | similarity between phrase and entry; higher is closer |

One extracted phrase can match several entries, so a document can produce more rows than phrases.

For the input above, the local 0.5B model on a laptop CPU returned:

| Type | Raw Concept | Taxonomy Concept | Taxonomy Source | Correlation Coefficient |
|---|---|---|---|---:|
| skill | Python programming | Program in Python | `ukos` | 0.70 |
| skill | Docker | Docker | `onet` | 0.75 |

Model output varies between runs and between models. A hosted or larger model typically extracts more
concepts from the same text — this small model missed "machine learning" and "AWS".

## Next steps

- Extract knowledge and tasks too, or restrict taxonomies — see [Usage](usage.md).
- Run a full analysis in the browser with the [Cookbook notebooks](cookbook.md).
