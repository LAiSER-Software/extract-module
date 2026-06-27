> [!CAUTION]
> <h3>LAiSER is currently in development mode, features could be experimental. Use with caution!</h3>


<div align="center">
<img src="https://i.imgur.com/XznvjNi.png" width="70%"/>
<h1>Leveraging ​Artificial ​Intelligence for ​Skill ​Extraction &​ Research (LAiSER)</h1>
</div>

### Contents
LAiSER is a tool that helps learners, educators and employers share trusted and mutually intelligible information about skills​.

- [About](#about)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Funding](#funding)
- [Authors](#authors)
- [Partners](#partners)

## About

LAiSER is an innovative tool that harnesses the power of artificial intelligence to simplify the extraction and analysis of skills. It is designed for learners, educators, and employers who want to gain reliable insights into skill sets, ensuring that the information shared is both trusted and mutually intelligible across various sectors.

By leveraging state-of-the-art AI models, LAiSER automates the process of identifying and classifying skills from diverse data sources. This not only saves time but also enhances accuracy, making it easier for users to discover emerging trends and in-demand skills.

The tool emphasizes standardization and transparency, offering a common framework that bridges the communication gap between different stakeholders. With LAiSER, educators can better align their teaching methods with industry requirements, and employers can more effectively identify the competencies required for their teams. The result is a more efficient and strategic approach to skill development, benefiting the entire ecosystem.

## Architecture

LAiSER uses a four-stage extraction and alignment pipeline:

1. Extraction
   Input text is normalized by input type and passed through prompt construction and LLM inference to produce raw concept candidates.
2. Parsing and deduplication
   Model output is parsed into structured concepts and filtered through exact and semantic deduplication.
3. Taxonomy alignment
   Extracted concepts are matched against bundled taxonomy indexes using embedding-based similarity search and threshold filtering.
4. Output normalization
   Alignment results are converted into a unified tabular schema, with optional edge generation for graph-style outputs.

## Requirements
- Python version `>=3.8`.
- The package supports the current tested matrix through Python `3.13`.
- A GPU is recommended for heavy local model workflows, but API-backed extraction can run CPU-only.
- Provider-specific environment variables may be required depending on backend:
  - `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - `OPENAI_API_KEY`

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

**NOTE**: Python 3.8 or later is required. Python 3.12 or 3.13 is recommended for current development and CI parity.

You can check if your machine has a GPU available with:
```shell
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

LAiSER is used as a Python package. The recommended API is `SkillExtractorRefactored`.

### Basic job description extraction

```python
import os
import pandas as pd

from laiser.extractor import SkillExtractorRefactored

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
)

print(results.head())
```

### Course syllabus extraction

```python
import os
import pandas as pd

from laiser.extractor import SkillExtractorRefactored

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
)

print(results.head())
```

### Common runtime options

- `model_id`
  Provider or model selector such as `gemini` or `openai`
- `api_key`
  API key for hosted providers
- `use_gpu`
  Enables GPU-backed initialization where supported
- `allowed_sources`
  Filters alignment sources such as `["esco"]`, `["onet"]`, or `["osn"]`
- `top_k`
  Per-alignment-call cap for matched rows
- `return_edges`
  Returns `{nodes, edges}` instead of only normalized rows
- `output_csv_path`
  Writes CSV output only when explicitly provided

Additional examples are available in [docs/examples.md](docs/examples.md).

## Funding
<div align="center">
<img src="https://i.imgur.com/XtgngBz.png" width="100px"/>
<img src="https://i.imgur.com/a2SNYma.jpeg" width="130px"/>
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
