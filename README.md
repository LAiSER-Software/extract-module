> [!CAUTION]
> <h3>LAiSER is currently in development mode, features could be experimental. Use with caution!</h3>


<div align="center">
<img src="https://i.imgur.com/XznvjNi.png" width="70%"/>
<h1>Leveraging ​Artificial ​Intelligence for ​Skill ​Extraction &​ Research (LAiSER)</h1>

<a href="https://laiser-software.github.io/extract-module/"><b>Documentation</b></a> ·
<a href="https://pypi.org/project/laiser/"><b>PyPI</b></a> ·
<a href="https://github.com/LAiSER-Software/laiser-cookbook"><b>Cookbook notebooks</b></a>
</div>

LAiSER turns free text such as job postings and course syllabi into structured skills, knowledge and tasks, each matched to an entry in ESCO, O\*NET, the UK Skills Classification or the Open Skills Network.

### Contents

- [About](#about)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Try it in Google Colab](#try-it-in-google-colab)
- [Funding](#funding)
- [Authors](#authors)
- [Partners](#partners)

## About

LAiSER is a Python package for turning unstructured text about work and learning into structured, comparable skill data.

You give it a table of documents — job postings, course descriptions, syllabi. For each document, a language model extracts the skills it describes and, optionally, the knowledge and tasks behind them. LAiSER then matches every extracted phrase to its closest entries in established taxonomies using embedding similarity. The result is a table with one row per match: the phrase as written, the taxonomy entry it corresponds to, which taxonomy that entry comes from, and a similarity score.

Because results point at shared taxonomy entries rather than free-text phrases, they can be counted and compared across sources — the skills employers ask for can be set against the skills a program teaches, or tracked across thousands of postings.

The language model can be a hosted API (Gemini or OpenAI) or a model running on your own machine, including on CPU with no API key. Taxonomy alignment always runs locally against indexes bundled with the package.

## Architecture

LAiSER runs four stages for every document:

1. **Extraction** — The text is placed into a prompt for its input type and sent to the language model, which returns candidate concepts.
2. **Parsing and deduplication** — The response is parsed into phrases. Exact duplicates are dropped and near-duplicates are collapsed by embedding similarity.
3. **Taxonomy alignment** — Each phrase is embedded and searched against bundled FAISS indexes of taxonomy entries. Matches above a per-type similarity threshold are kept.
4. **Output** — Matches are returned as a single table, optionally with graph edges linking knowledge areas to the tasks they enable.

## Requirements

- Python `>=3.10`; CI tests 3.10 through 3.13.
- No GPU or API key is required. A GPU speeds up larger local models; hosted providers need their own API key.
- Supported model providers and their settings are listed in the [documentation](https://laiser-software.github.io/extract-module/providers/).

## Setup and Installation

- Install LAiSER from PyPI:

  ```shell
  pip install laiser
  ```

- Install with GPU extras:

  ```shell
  pip install "laiser[gpu]"
  ```

- Install development dependencies from source:

  ```shell
  pip install -e ".[dev]"
  ```

You can check if your machine has a GPU available with:

```shell
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

LAiSER is used as a Python package. The recommended API is `SkillExtractorRefactored`.

### Without an API key

This runs a small open model locally on CPU. The model downloads on first use.

```python
import pandas as pd

from laiser.skill_extractor_refactored import SkillExtractorRefactored

data = pd.DataFrame(
    [
        {
            "Research ID": "job-001",
            "description": "Build production machine learning systems in Python.",
        }
    ]
)

extractor = SkillExtractorRefactored(model_id="Qwen/Qwen2.5-0.5B-Instruct", use_gpu=False)

results = extractor.extract_concepts(
    data=data,
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
    concepts=["skills"],
)

print(results[["Raw Concept", "Taxonomy Concept", "Taxonomy Source", "Correlation Coefficient"]])
```

Small local models are convenient for trying LAiSER out; hosted or larger models extract more reliably.

### Job description extraction with Gemini

```python
import os
import pandas as pd

from laiser.skill_extractor_refactored import SkillExtractorRefactored

data = pd.DataFrame(
    [
        {
            "Research ID": "job-001",
            "description": "Build production machine learning systems in Python.",
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
    concepts=["skills", "knowledge", "tasks"],
    allowed_sources=["esco", "onet", "osn", "ukos"],
)

print(results.head())
```

### Course syllabus extraction

```python
import os
import pandas as pd

from laiser.skill_extractor_refactored import SkillExtractorRefactored

data = pd.DataFrame(
    [
        {
            "Research ID": "course-001",
            "description": "Introduction to data visualization and exploratory analysis.",
            "learning_outcomes": "Create dashboards, explain patterns in data, and evaluate charts.",
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
    text_columns=["description", "learning_outcomes"],
    input_type="course_syllabi",
    concepts=["skills"],
    allowed_sources=["esco", "onet", "osn", "ukos"],
)

print(results.head())
```

### Common runtime options

| Option | Description |
|---|---|
| `model_id` | `"gemini"`, `"openai"`, or a Hugging Face model id to run locally |
| `api_key` | API key for hosted providers |
| `use_gpu` | run local models on GPU where available |
| `backend` | `"llama_cpp"` to run a local GGUF model |
| `concepts` | `["skills"]` (default), or add `"knowledge"` and `"tasks"` |
| `allowed_sources` | taxonomies to align against: `"esco"`, `"onet"`, `"osn"`, `"ukos"` |
| `top_k` | maximum aligned matches per document (default 25) |
| `return_edges` | return `{nodes, edges}` instead of only the results table |
| `output_csv_path` | also write the results to this CSV file |

Every option is described in the [usage guide](https://laiser-software.github.io/extract-module/usage/), and more snippets are in [docs/examples.md](docs/examples.md).

## Try it in Google Colab

The [cookbook](https://github.com/LAiSER-Software/laiser-cookbook) has complete analyses that open directly in Colab:

| Notebook | |
|---|---|
| Job skill analysis for job seekers | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAiSER-Software/laiser-cookbook/blob/main/recipes/Job_Skill_Analysis/Job_Skill_Analysis_for_Job_Seekers.ipynb) |
| University program skill analysis | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAiSER-Software/laiser-cookbook/blob/main/recipes/University_Program_Skill_Analysis/University_Program_Skill_Analysis.ipynb) |
| Pay equity analysis | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAiSER-Software/laiser-cookbook/blob/main/recipes/Pay_Equity_Analysis/Pay_Equity_Analysis_Based_on_Skills.ipynb) |

## Funding
<div align="center">
<img src="https://laiser.gwu.edu/sites/g/files/zaxdzs6976/files/2026-07/gwu.jpg" width="100px" alt="George Washington University"/>
<img src="https://laiser.gwu.edu/sites/g/files/zaxdzs6976/files/2026-07/gatesfoundation.jpg" width="130px" alt="Gates Foundation"/>
<img src="https://laiser.gwu.edu/sites/g/files/zaxdzs6976/files/2026-07/walmartfoundation.jpg" width="130px" alt="Walmart Foundation"/>
</div>

## Authors
<a href="https://github.com/LAiSER-Software/extract-module/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=LAiSER-Software/extract-module" />
</a>

## Partners
<div align="center">
<img src="https://i.imgur.com/hMb5n6T.png" width="120px"/>
<img src="https://i.imgur.com/dxz2Udo.png" width="70px"/>
<img src="https://i.imgur.com/5O1EuFU.png" width="100px"/>
</div>



</br>
<!-- <p align='center'> <b> Made with Passion💖, Data Science📊, and a little magic!🪄 </b></p> -->
