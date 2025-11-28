---
title: 'LAiSER: A Taxonomy-Aware Framework for Skill Extraction and Research'
tags:
  - Python
  - natural language processing
  - skill extraction
  - labor market intelligence
  - large language models
  - workforce analytics
authors:
  - name: Satya Phanindra Kumar Kalaga
    orcid: 0009-0004-6447-3760
    affiliation: 1
  - name: Bharat Khandelwal
    orcid: 0009-0001-9491-5468
    affiliation: 1
affiliations:
  - name: Program on Skills, Credentials and Workforce Policy, Institute of Public Policy, The George Washington University, USA
    index: 1
date: 1 November 2025
bibliography: paper.bib
---

# Summary

LAiSER is an artificial intelligence framework for extracting, standardizing, and aligning skill information from unstructured text with established skill taxonomies. The system addresses the lack of standardized, machine-readable skill information needed to facilitate communication between learners, educators, and employers. LAiSER employs a two-stage pipeline combining large language models (LLMs) with semantic vector search using FAISS [@Johnson2019] for high-precision skill extraction and taxonomy alignment to standards like ESCO [@ESCO2020]. The framework supports multiple computational backends—GPU-accelerated inference, cloud APIs, and CPU-only environments—making it accessible across diverse research and operational contexts.

# Statement of need

The contemporary labor market is characterized by rapid technological change, evolving skill requirements, and growing disconnect between educational curricula and industry needs [@Autor2013]. Traditional skill extraction approaches rely on manual annotation, expert judgment, or keyword matching—methods that are labor-intensive, error-prone, and fail to capture semantic richness [@Boselli2018]. LAiSER fills this gap by providing an accessible, flexible, and accurate framework for automated skill extraction and taxonomy alignment, enabling researchers, educators, and workforce professionals to analyze skill demands at scale.

# Key Features

LAiSER provides three core innovations:

1. **Intelligent Text Preprocessing**: Domain-aware preprocessing removes extraneous information (company branding, legal boilerplate, benefits) while preserving task-relevant skill content.

2. **Multi-Model Skill Extraction**: Flexible architecture supporting multiple LLMs (vLLM, HuggingFace Transformers, Gemini API) with automatic fallback mechanisms for deployment across different computational environments.

3. **Taxonomy-Aware Alignment**: Semantic similarity-based alignment using sentence transformers [@Reimers2019] and FAISS vector search maps extracted skills to standardized taxonomies, enabling cross-domain skill analysis.

The framework also supports extraction within the Knowledge, Skills, and Abilities (KSA) framework with automatic proficiency level classification using the Scottish Credit and Qualifications Framework (SCQF) [@SCQF2019].

# Implementation

LAiSER is implemented in Python 3.9+ using PyTorch, HuggingFace Transformers, FAISS, spaCy, Sentence-Transformers, and pandas. The modular service architecture separates data access, skill extraction, and taxonomy alignment into loosely coupled layers.

The system accepts pandas DataFrames containing textual descriptions and produces structured output including raw extracted skills, taxonomy-aligned canonical skills, taxonomy identifiers (e.g., ESCO codes), and semantic similarity scores. For KSA extraction, output includes SCQF proficiency levels, knowledge requirements, and task abilities.

# Applications

LAiSER enables diverse applications: labor market intelligence through large-scale job advertisement analysis; curriculum development through skill gap identification between educational programs and industry demands; skills-based hiring through standardized job-candidate matching; career pathway analysis through skill similarity assessment across occupations; and workforce policy research through regional skills assessments.

# Availability

The source code is available at [https://github.com/LAiSER-Software/extract-module](https://github.com/LAiSER-Software/extract-module) under the MIT License. Installation via pip:

```bash
pip install laiser[gpu]  # GPU-accelerated
pip install laiser[cpu]  # CPU-only
```

Example usage:

```python
from laiser.skill_extractor_refactored import SkillExtractorRefactored
import pandas as pd

extractor = SkillExtractorRefactored(model_id="gemini", api_key="YOUR_API_KEY", use_gpu=False)
data = pd.read_csv("job_descriptions.csv")
results = extractor.extract_and_align(data, id_column="job_id", 
    text_columns=["description"], input_type="job_desc")
```

# Acknowledgments

The authors acknowledge the George Washington University Institute of Public Policy and the Program on Skills, Credentials, and Workforce Policy for institutional support. This project was supported by grants from the Walmart Foundation and the Gates Foundation. The authors thank the GW Open Source Program Office and the developers of HuggingFace Transformers, FAISS, and spaCy.

# References
