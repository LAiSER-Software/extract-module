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
- [Reference configuration](#reference-configuration)
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
- `similarity_threshold`
  Minimum cosine similarity for an alignment match to be kept
- `temperature`
  Decoding temperature applied to whichever backend is selected. Defaults to `0.0`, which selects greedy decoding and makes repeated runs over the same input reproducible. Raise it only when sampled variation is wanted
- `seed`
  Sampling seed forwarded to every backend that accepts one, namely vLLM, llama.cpp, Gemini, and local Transformers. Defaults to `42`. The OpenAI Responses API and the Anthropic Messages API accept no seed, so on those backends reproducibility rests on greedy decoding alone
- `output_csv_path`
  Writes CSV output only when explicitly provided

Additional examples are available in [docs/examples.md](docs/examples.md).

## Reference configuration

Results from an LLM pipeline are only reproducible if the configuration that produced them is stated. The settings below are the reference configuration for LAiSER 0.5. We recommend pinning them when reporting results, and reporting any departure from them alongside your findings.

| Setting | Reference value | Why it matters |
| --- | --- | --- |
| `model_id` | `gemini` (hosted) or `TheBloke/Mistral-7B-Instruct-v0.1-AWQ` (local GPU) | Different models extract different skill sets from the same text. This is the single largest source of variation between runs |
| Gemini model | `gemini-2.5-flash` | Pin with the `LAISER_GEMINI_MODEL` environment variable. Hosted model aliases are updated by the provider over time |
| `temperature` | `0.0` | Greedy decoding. This is the default and should not be raised unless sampled variation is the objective |
| `seed` | `42` | The default. Applies to vLLM, llama.cpp, Gemini, and local Transformers |
| `similarity_threshold` | `0.60` for skills, `0.50` for knowledge and tasks | Governs which alignment matches are retained. Lowering it admits weaker matches |
| `top_k` | `25` | Maximum aligned rows returned per document |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Fixed. Changing it invalidates the prebuilt taxonomy index and shifts every similarity score |
| Taxonomy | ESCO, O\*NET, and OSN via the combined index | Select with `allowed_sources`, for example `["esco"]` |

### Reproducing a run

```python
import os

import pandas as pd

from laiser.skill_extractor_refactored import SkillExtractorRefactored

extractor = SkillExtractorRefactored(
    model_id="gemini",
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    use_gpu=False,
    temperature=0.0,  # greedy decoding, the default
    seed=42,          # the default
)

results = extractor.extract_concepts(
    data=pd.read_csv("your_input.csv"),
    id_column="Research ID",
    text_columns=["description"],
    input_type="job_desc",
    concepts=["skills"],
    allowed_sources=["esco"],
    similarity_threshold=0.60,
    top_k=25,
)
```

For a local GPU run, substitute the model identifier and supply a Hugging Face token:

```python
extractor = SkillExtractorRefactored(
    model_id="TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
    hf_token=os.getenv("HF_TOKEN"),
    use_gpu=True,
    temperature=0.0,
    seed=42,
)
```

### What determinism does and does not guarantee

Decoding is deterministic by default, and the alignment stage is deterministic by construction: it embeds each extracted phrase with a fixed sentence-transformer model and retrieves matches from an exact inner-product index, so the same phrase always resolves to the same taxonomy entry with the same similarity score. Aligned output is therefore substantially more stable than the raw text the model generates.

Two limits are worth stating plainly. Hosted APIs can change model behaviour behind a stable model name, so exact reproduction across long time spans is not guaranteed on the cloud backends. And alignment normalizes how a skill is expressed but cannot recover a skill the model failed to emit, so the number and identity of extracted skills can still differ between models. Treat cross-model agreement as a quantity to measure, not to assume.

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
