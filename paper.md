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
  - name: Anket Vilasrao Patil
    orcid: 0009-0000-0619-4174
    affiliation: 1
  - name: Prudhviraju Chekuri
    orcid: 0009-0003-8682-8406
    affiliation: 1
affiliations:
  - name: Program on Skills, Credentials and Workforce Policy, Institute of Public Policy, The George Washington University, USA
    index: 1
date: 1 November 2025
bibliography: paper.bib
---

# Summary

Job postings, course syllabi, and credential descriptions all describe skills, but each in its own words, so the same ability can be phrased many different ways. That makes it hard to count which skills employers ask for, or to check whether a course teaches what jobs require. Skill taxonomies address this by giving each skill one agreed name and identifier; the European Commission's European Skills, Competences, Qualifications and Occupations classification (ESCO) [@ESCO2020] is a widely used example.

LAiSER is open-source software that reads such documents and links each skill they mention to its entry in a taxonomy. It uses a large language model (LLM), a kind of artificial intelligence (AI) system that reads and writes text, to find the skills a document describes, then links each one to the closest taxonomy entry, a step we call alignment, so that differently worded mentions of the same skill are counted together. It is designed for researchers studying labor markets, curriculum designers comparing programs with employer demand, and organizations that issue digital credentials.

# Statement of need

The contemporary labor market is characterized by rapid technological change, evolving skill requirements, and growing disconnect between educational curricula and industry needs [@Autor2013]. Traditional skill extraction approaches rely on manual annotation, expert judgment, or keyword matching, all of which are labor-intensive, error-prone, and fail to capture semantic richness [@Boselli2018]. LAiSER provides a framework for automated skill extraction and taxonomy alignment, enabling researchers, educators, and workforce professionals to analyze skill demands at scale. Target users include labor economists studying occupational transitions, curriculum designers auditing program alignment with industry needs, and credentialing organizations producing standards-aligned digital badges.

# State of the field

Several open-source tools address parts of the skill extraction problem. SkillNER [@SkillNER2022] uses rule-based matching against the EMSI (now Lightcast) skills database via spaCy, an open-source natural language processing library, offering fast lookups but limited ability to detect novel or context-dependent skills not present in its dictionary. Nesta's Skills Extractor Library [@Nesta2023] trains a spaCy named entity recognition (NER) model on labeled job advertisements and maps extracted phrases to ESCO or Lightcast taxonomies using sentence-transformer embeddings. While effective for UK job advertisements, its NER model requires domain-specific labeled training data and does not generalize well to other text types such as syllabi or credential descriptions. The esco-skill-extractor [@EscoExtractor2024] takes a simpler approach, embedding full input texts with a transformer and matching against ESCO entries by cosine similarity, but it lacks preprocessing and multi-model flexibility.

LAiSER differs from these three tools in several key respects. First, it uses LLMs for skill extraction rather than rule-based or NER-based approaches, enabling it to identify skills expressed in varied natural language without requiring labeled training data. Second, LAiSER's extraction prompts instruct the model to disregard extraneous content (company branding, legal boilerplate, benefits sections) while preserving task-relevant skill language. Third, the framework supports multiple LLM backends (vLLM, HuggingFace Transformers, llama.cpp, and the Gemini and OpenAI application programming interfaces, or APIs), allowing deployment on graphics processing unit (GPU) clusters, hosted APIs, or ordinary laptops.

LLMs have also been applied to skill extraction directly. SkillGPT [@Li2023SkillGPT] extracts skills from job descriptions with an open-source LLM and standardizes them to ESCO through vector similarity search, and ESCOX [@Kavargyris2025ESCOX] is a tool that uses LLMs to extract ESCO skills and occupations from unstructured text. Skill-LLM [@Herandi2024SkillLLM] fine-tunes an LLM for skill extraction from job descriptions, and @Nguyen2024Rethinking evaluate in-context learning with LLMs on six skill extraction benchmarks, finding that LLMs trail supervised models overall but better handle syntactically complex skill mentions.

# Software design

LAiSER's architecture is organized into three loosely coupled layers: data access, skill extraction, and taxonomy alignment. This separation was chosen to allow each component to evolve independently and to support diverse deployment scenarios without requiring changes to the overall pipeline.

The extraction layer posed the most significant design trade-off. Training a custom NER model, as Nesta's library does, would yield fast inference but would require substantial labeled data for each new domain (syllabi, credentials, job postings). Instead, LAiSER delegates extraction to LLMs via a prompt engineering approach, trading some inference speed for broad domain generalizability without retraining. To mitigate vendor lock-in and hardware constraints, the extraction layer abstracts over multiple backends: vLLM for GPU clusters, HuggingFace Transformers and llama.cpp for models run locally, and the Gemini and OpenAI APIs for hosted inference. The backend is chosen when an extractor is created, from the requested model and the available hardware. On a GPU, if vLLM cannot load the requested model it first retries with a default checkpoint, and only if that also fails does it fall back to Transformers.

For taxonomy alignment, the framework embeds both the extracted phrases and the taxonomy entries with a sentence-transformer model [@Reimers2019] and retrieves each phrase's closest entry from a precomputed index built with FAISS (Facebook AI Similarity Search) [@Johnson2019]. The index is a flat inner-product index over normalized embeddings, which makes retrieval exhaustive and exact rather than approximate. Exactness was a deliberate choice: at the scale of the bundled taxonomies, about 27,000 skill entries, an exhaustive scan is inexpensive relative to LLM inference in the preceding stage, and it guarantees that a given query returns the same entry and the same similarity score on every run, a precondition for the reproducibility properties described below. Because retrieval is encapsulated behind an index manager, an approximate index could be substituted without changes to the alignment logic should taxonomy size later require it. The system accepts pandas DataFrames as input and returns one row per aligned phrase, containing the extracted text, the matched taxonomy entry, its source taxonomy and link, and the similarity score.

Because LLMs sample their output, identical inputs can yield different generations across runs, and different backends can yield different generations from the same input. LAiSER addresses this in three ways. First, decoding is deterministic by default: every backend decodes greedily at temperature zero, and a fixed seed is supplied to every backend that accepts one, so runs stay reproducible even when a user deliberately raises the temperature. Temperature and seed are constructor arguments, so sampled variation is available when wanted but is never the silent default. Second, the alignment stage acts as a normalizing projection. Generated phrases are embedded with a fixed sentence-transformer model and matched by exact search against a precomputed index, so surface variation in the generated text collapses onto a closed, finite vocabulary of taxonomy entries, and different phrasings of the same competency can converge on the same aligned record, although because each phrase is matched independently this is likely rather than guaranteed. Third, every backend produces the same output schema, which makes results directly comparable across models and lets cross-model agreement be measured on the aligned columns. The test suite checks each backend's decoding defaults, the decoding parameters each backend actually receives, and that repeated alignment of a fixed input returns an identical result, similarity scores included.

These measures reduce variability but do not eliminate it. Hosted models can change behind an API, and providers do not guarantee identical output even at temperature zero. Alignment can normalize how a skill is expressed, but it cannot recover a skill that a model failed to emit, so the number and identity of extracted skills can still differ across models. We therefore recommend that users pin the model identifier, temperature, seed, and similarity thresholds when reporting results, and treat cross-model agreement as an empirical quantity to be measured rather than assumed.

The source code is available at [https://github.com/LAiSER-Software/extract-module](https://github.com/LAiSER-Software/extract-module) under the BSD 3-Clause License, and can be installed with `pip install laiser`, or `pip install laiser[gpu]` for GPU inference.

# Research impact statement

LAiSER has demonstrated realized impact across multiple institutions and funding initiatives. The framework's early prototypes contributed to securing a $250,000 grant from the Walmart Foundation and a subsequent $750,000 grant from the Gates Foundation, both supporting the broader Program on Skills, Credentials and Workforce Policy at The George Washington University (GW). LAiSER is integrated into the Credential Co-writer tool of the Massachusetts Institute of Technology (MIT) Digital Credentials Consortium [@DCC2025], where it provides skill extraction capabilities for generating standards-aligned Open Badges 3.0 credential templates in collaboration with Western Governors University, OneOrigin, and Open edX. Northeastern University has adopted LAiSER for internal academic projects involving skills analysis. The framework has been applied to a large-scale Texas syllabi skills extraction project using the OpenSyllabus Analytics API, analyzing thousands of course documents to map curriculum content to workforce skill demands.

LAiSER received the Bronze Award in the AI in Education (Higher Education Institutions) category at the QS Reimagine Education Awards 2025 [@QSReimagine2025], selected from over 1,600 global submissions. The project has been presented at Badge Summit 2025 (Colorado) and showcased at the GW Open Source Conference, GW Columbian College of Arts and Sciences (CCAS) Poster Day, and GW Innovation Fest. LAiSER is listed in the George Washington University Open Source Program Office project registry. A companion [cookbook repository](https://github.com/LAiSER-Software/laiser-cookbook) provides reproducible use-case notebooks, and a multi-session bootcamp series has been delivered to train researchers and practitioners on the framework.

# AI usage disclosure

Generative AI tools were used during the development of LAiSER and the preparation of this paper. GitHub Copilot was used for code assistance during software development; all Copilot-suggested code was verified through automated test suites and manual code review. Google Gemini and Anthropic Claude were used to draft portions of the software documentation; all AI-generated documentation was reviewed and edited by graduate student volunteers affiliated with the project. No generative AI tools were used for the core algorithmic design, architectural decisions, or research analysis presented in this paper. The paper text was drafted by the authors with AI-assisted editing for clarity.

# Acknowledgments

The authors acknowledge the George Washington University Institute of Public Policy and the Program on Skills, Credentials, and Workforce Policy for institutional support. This project was supported by grants from the Walmart Foundation and the Gates Foundation. The authors thank the GW Open Source Program Office, the MIT Digital Credentials Consortium, and the developers of HuggingFace Transformers, FAISS, and spaCy.

# References
