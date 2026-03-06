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

LAiSER is an open-source Python framework for extracting, standardizing, and aligning skill information from unstructured text with established skill taxonomies. The system addresses the lack of standardized, machine-readable skill information needed to facilitate communication between learners, educators, and employers. LAiSER employs a two-stage pipeline combining large language models (LLMs) with semantic vector search using FAISS [@Johnson2019] for high-precision skill extraction and taxonomy alignment to standards such as ESCO [@ESCO2020]. The framework supports multiple computational backends, including GPU-accelerated inference, cloud APIs, and CPU-only environments, making it accessible across diverse research and operational contexts. LAiSER also supports extraction within the Knowledge, Skills, and Abilities (KSA) framework with automatic proficiency level classification using the Scottish Credit and Qualifications Framework (SCQF) [@SCQF2019]. The source code is available at [https://github.com/LAiSER-Software/extract-module](https://github.com/LAiSER-Software/extract-module) under the MIT License, installable via `pip install laiser[gpu]` or `pip install laiser[cpu]`.

# Statement of need

The contemporary labor market is characterized by rapid technological change, evolving skill requirements, and growing disconnect between educational curricula and industry needs [@Autor2013]. Traditional skill extraction approaches rely on manual annotation, expert judgment, or keyword matching, all of which are labor-intensive, error-prone, and fail to capture semantic richness [@Boselli2018]. The challenge is compounded by the absence of integrated tooling that combines skill identification with alignment to recognized taxonomies, forcing researchers to assemble fragile multi-tool pipelines. LAiSER fills this gap by providing a unified, flexible, and accurate framework for automated skill extraction and taxonomy alignment, enabling researchers, educators, and workforce professionals to analyze skill demands at scale. Target users include labor economists studying occupational transitions, curriculum designers auditing program alignment with industry needs, and credentialing organizations producing standards-aligned digital badges.

# State of the field

Several open-source tools address parts of the skill extraction problem. SkillNER [@SkillNER2022] uses rule-based matching against the EMSI skills database via spaCy, offering fast lookups but limited ability to detect novel or context-dependent skills not present in its dictionary. Nesta's Skills Extractor Library [@Nesta2023] trains a spaCy NER model on labeled job advertisements and maps extracted phrases to ESCO or Lightcast taxonomies using sentence-transformer embeddings. While effective for UK job advertisements, its NER model requires domain-specific labeled training data and does not generalize well to other text types such as syllabi or credential descriptions. The esco-skill-extractor [@EscoExtractor2024] takes a simpler approach, embedding full input texts with a transformer and matching against ESCO entries by cosine similarity, but it lacks preprocessing, multi-model flexibility, and proficiency-level classification.

LAiSER differs from these tools in several key respects. First, it uses LLMs for skill extraction rather than rule-based or NER-based approaches, enabling it to identify skills expressed in varied natural language without requiring labeled training data. Second, LAiSER provides domain-aware preprocessing that removes extraneous content (company branding, legal boilerplate, benefits sections) while preserving task-relevant skill language, an important step that existing tools omit. Third, the framework supports multiple LLM backends (vLLM, HuggingFace Transformers, Gemini API) with automatic fallback, allowing deployment across GPU clusters, cloud APIs, or CPU-only laptops. Finally, LAiSER uniquely supports KSA decomposition with SCQF proficiency-level classification, a capability absent from all comparable tools. These design choices reflect a deliberate decision to build a new framework rather than extend existing tools, whose architectures are not designed to accommodate LLM-based extraction or multi-taxonomy proficiency alignment.

# Software design

LAiSER's architecture is organized into three loosely coupled layers: data access, skill extraction, and taxonomy alignment. This separation was chosen to allow each component to evolve independently and to support diverse deployment scenarios without requiring changes to the overall pipeline.

The extraction layer posed the most significant design trade-off. Training a custom NER model, as Nesta's library does, would yield fast inference but would require substantial labeled data for each new domain (syllabi, credentials, job postings). Instead, LAiSER delegates extraction to LLMs via a prompt engineering approach, trading some inference speed for broad domain generalizability without retraining. To mitigate vendor lock-in and hardware constraints, the extraction layer abstracts over multiple backends (vLLM for GPU clusters, HuggingFace Transformers for local models, and the Gemini API for cloud-based inference), with automatic fallback when a backend is unavailable.

For taxonomy alignment, the framework uses FAISS for approximate nearest-neighbor search over sentence-transformer embeddings [@Reimers2019] rather than brute-force cosine similarity. This choice enables sub-linear scaling as taxonomy size grows, which is essential when aligning against ESCO's 13,000+ skill entries at production volumes. The system accepts pandas DataFrames as input and produces structured output including raw extracted skills, taxonomy-aligned canonical skills, taxonomy identifiers (e.g., ESCO codes), semantic similarity scores, and optional SCQF proficiency levels.

# Research impact statement

LAiSER has demonstrated realized impact across multiple institutions and funding initiatives. The framework's early prototypes contributed to securing a $250,000 grant from the Walmart Foundation and a subsequent $750,000 grant from the Gates Foundation, both supporting the broader Program on Skills, Credentials and Workforce Policy at The George Washington University. LAiSER is integrated into MIT's Digital Credentials Consortium Credential Co-writer tool [@DCC2025], where it provides skill extraction capabilities for generating standards-aligned Open Badges 3.0 credential templates in collaboration with Western Governors University, OneOrigin, and Open edX. Northeastern University has adopted LAiSER for internal academic projects involving skills analysis. The framework has been applied to a large-scale Texas syllabi skills extraction project using the OpenSyllabus Analytics API, analyzing thousands of course documents to map curriculum content to workforce skill demands.

LAiSER received the Bronze Award in the AI in Education (Higher Education Institutions) category at the QS Reimagine Education Awards 2025 [@QSReimagine2025], selected from over 1,600 global submissions. The project has been presented at Badge Summit 2025 (Colorado) and showcased at the GW Open Source Conference, GW CCAS Poster Day, and GW Innovation Fest. LAiSER is listed in the George Washington University Open Source Program Office project registry. A companion cookbook repository provides reproducible use-case notebooks, and a multi-session bootcamp series has been delivered to train researchers and practitioners on the framework.

# AI usage disclosure

Generative AI tools were used during the development of LAiSER and the preparation of this paper. GitHub Copilot was used for code assistance during software development; all Copilot-suggested code was verified through automated test suites and manual code review. Google Gemini and Anthropic Claude were used to draft portions of the software documentation; all AI-generated documentation was reviewed and edited by graduate student volunteers affiliated with the project. No generative AI tools were used for the core algorithmic design, architectural decisions, or research analysis presented in this paper. The paper text was drafted by the authors with AI-assisted editing for clarity.

# Acknowledgments

The authors acknowledge the George Washington University Institute of Public Policy and the Program on Skills, Credentials, and Workforce Policy for institutional support. This project was supported by grants from the Walmart Foundation and the Gates Foundation. The authors thank the GW Open Source Program Office, the MIT Digital Credentials Consortium, and the developers of HuggingFace Transformers, FAISS, and spaCy.

# References
