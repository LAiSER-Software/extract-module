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

Students and job seekers acquire skills through university courses, online courses and previous jobs, while many employers define in-house taxonomies for skills. Across syllabi and job descriptions, skills are just keywords, and without additional context what a skill means is left to subjective inference. Existing approaches that implement classical natural language processing (NLP) methods often miss domain context [@Boselli2018], and works that use transformers or LLMs leave gaps in reproducibility and source grounding [@Li2023SkillGPT; @Herandi2024SkillLLM; @Nguyen2024Rethinking]. Without grounding, researchers and curriculum designers can be misled by irrelevant matches, which could negatively affect the decisions those matches inform. LAiSER is an open-source research tool for skill researchers that acts as a translation layer between the skill language of universities and industry, grounding its results in national taxonomies, with the choice of LLM left to the researcher. Every match carries its taxonomy source, link and similarity score, so users can check it. We share it so that researchers and educators can focus on understanding skills and iterating on their findings instead of building the pipeline themselves. Target users include labor economists studying occupational transitions, curriculum designers auditing program alignment with industry needs, and credentialing organizations producing standards-aligned digital badges.

# State of the field

Several open-source tools address parts of the skill extraction problem. SkillNER [@SkillNER2022] uses rule-based matching against the EMSI (now Lightcast) skills database via spaCy, an open-source NLP library, offering fast lookups but limited ability to detect novel or context-dependent skills not present in its dictionary. Nesta's Skills Extractor Library [@Nesta2023] trains a spaCy named entity recognition (NER) model on labeled job advertisements and maps extracted phrases to ESCO or Lightcast taxonomies using sentence-transformer embeddings. The esco-skill-extractor, also published as ESCOX [@EscoExtractor2024; @Kavargyris2025ESCOX], embeds full input texts with a transformer and matches them against ESCO entries by cosine similarity.

LLMs have also been applied to skill extraction directly. SkillGPT [@Li2023SkillGPT] extracts skills from job descriptions with an open-source LLM and standardizes them to ESCO through vector similarity search. Skill-LLM [@Herandi2024SkillLLM] fine-tunes an LLM for skill extraction from job descriptions, and @Nguyen2024Rethinking evaluate in-context learning with LLMs on six skill extraction benchmarks, finding that LLMs trail supervised models overall but better handle syntactically complex skill mentions.

LAiSER's novelty is in the overall pipeline rather than any single step: it combines multi-LLM compatibility, multi-taxonomy compatibility, and a bring-your-own-model skeleton that accepts any Hugging Face model, a local model file, or the Gemini and OpenAI application programming interfaces (APIs). None of the tools above combines LLM extraction with mapping to more than one taxonomy; LAiSER maps to ESCO, the US Occupational Information Network (O\*NET), the UK Skills Classification and the Open Skills Network. We started out by integrating Nesta's library in our early prototypes to find out its issues and expand on it. At the time its NER model was not good with domain context: a term such as Apache Airflow could be matched with heating, ventilation and air conditioning (HVAC) airflow in the taxonomy. We moved on to letting an LLM read the whole document and return the full skill name, which LAiSER then matches with a sentence-transformer model; "Apache Airflow" now matches O\*NET's Apache Airflow entry. LAiSER grew from this into an umbrella tool that allows researchers to implement whichever setup they please. Like other prompted-LLM approaches, which @Nguyen2024Rethinking found trail supervised models, LAiSER has no published evaluation, and because models can drift, we recommend that users spot-check results on their own documents.

# Software design

When a user shares a job description, or a list of them, LAiSER initializes the model depending on the runtime environment and routes the text to a hosted API, an LLM on a graphics processing unit (GPU), or a smaller local model. The model extracts the skill keywords from the posting, and LAiSER finds the closest match for each in the taxonomies we indexed. The output lists the skills extracted, their taxonomy matches, similarity scores, and the source and link of each matched entry. An optional second model pass also extracts knowledge and tasks.

We use LLMs because many skill keywords carry a domain context that can only be captured by looking at the whole sentence. An LLM costs more than an NER model in speed, dollar spend and resources, but an NER setup needs labeled data for each new domain and would become a bottleneck as we scale the scope of the pipeline; LLMs scale to more domains quickly with zero-shot prompting and improve with few-shot examples. Because some researchers have credits for frontier AI models, some have high-performance computing (HPC) clusters, and some have neither, LAiSER supports hosted APIs, vLLM on GPUs, and Transformers or llama.cpp on ordinary machines.

Matching uses exact search, primarily for reproducibility: as research tooling, LAiSER should not give completely different results from run to run, since that may jeopardize downstream work. Each extracted phrase is embedded with a sentence-transformer model [@Reimers2019] and compared with every entry in a Facebook AI Similarity Search (FAISS) index [@Johnson2019]. At about 27,000 skill entries, an exhaustive search is cheap next to the LLM call, and the same phrase always returns the same entry and score. The closest entry is kept only if its similarity clears a threshold (0.60 for skills by default), so a phrase with no good match, such as a bare "Airflow", is dropped rather than mismatched.

In its default configuration, LAiSER decodes at temperature zero and passes a fixed seed to every backend that accepts one, so that repeated runs with a local model should return the same output. These measures have limits. Hosted models may change under the same name, and providers do not guarantee identical output even at temperature zero; the OpenAI API also accepts no seed. Matching cannot recover a skill the model did not extract, and different models may extract different skills. The prompt asks for at most ten skills per document, a design choice to filter out noise, so long documents are summarized rather than covered exhaustively. The similarity thresholds were tuned against an internal hand-labeled sample that is not yet public. So that a batch keeps running, a document whose model call fails, or whose reply cannot be parsed, is skipped with a warning. All document types, including syllabi, use the same extraction prompt, because different prompts for different documents would give results that are hard to compare, and one prompt is easier to maintain. We therefore recommend that users record the model, temperature, seed and thresholds when reporting results, and spot-check a sample of the output.

The source code is available at [https://github.com/LAiSER-Software/extract-module](https://github.com/LAiSER-Software/extract-module) under the BSD 3-Clause License, and can be installed with `pip install laiser`, or `pip install laiser[gpu]` for GPU inference.

# Research impact statement

The framework's early prototypes contributed to securing a $250,000 grant from the Walmart Foundation and a subsequent $750,000 grant from the Gates Foundation, both supporting the broader Program on Skills, Credentials and Workforce Policy at The George Washington University (GW). LAiSER is integrated into the Credential Co-writer tool of the Massachusetts Institute of Technology (MIT) Digital Credentials Consortium [@DCC2025], where it provides skill extraction capabilities for generating standards-aligned Open Badges 3.0 credential templates in collaboration with Western Governors University, OneOrigin, and Open edX. Northeastern University has adopted LAiSER for internal academic projects involving skills analysis. The framework has been applied to a large-scale Texas syllabi skills extraction project using the OpenSyllabus Analytics API, analyzing thousands of course documents to map curriculum content to workforce skill demands.

LAiSER received the Bronze Award in the AI in Education (Higher Education Institutions) category at the QS Reimagine Education Awards 2025 [@QSReimagine2025], selected from over 1,600 global submissions. The project has been presented at Badge Summit 2025 (Colorado) and showcased at the GW Open Source Conference, GW Columbian College of Arts and Sciences (CCAS) Poster Day, and GW Innovation Fest. LAiSER is listed in the George Washington University Open Source Program Office project registry. A companion [cookbook repository](https://github.com/LAiSER-Software/laiser-cookbook) provides reproducible use-case notebooks, and a multi-session bootcamp series has been delivered to train researchers and practitioners on the framework.

# AI usage disclosure

Generative AI was used in developing LAiSER and preparing this paper. GitHub Copilot (completion, coding agent and Autofix) and Claude Code (Claude Opus 5) wrote and refactored code and tests, all checked by the test suite and reviewed by the authors. Gemini 2.5 Pro and 3.1 Pro, Claude 3.5 Sonnet, and Claude Opus 4, 5 and 5.5 drafted portions of the documentation, which project members reviewed and edited. For the paper, Copilot's coding agent shortened a November 2025 draft, Claude 3.5 Sonnet and Claude Opus 4 drafted the March 2026 version, and Claude Code (Claude Opus 5) co-authored the September 2026 revisions. In this revision, the authors answered structured questions for the Statement of need, State of the field and Software design; Claude Code (Claude Opus 5.5) assembled those answers into paragraphs, added technical details from the code, checked claims against the code and sources, and drafted the Software design limitations paragraph, which the authors revised. The authors made all design decisions, reviewed and validated all AI-assisted content, and take full responsibility for the paper.

# Acknowledgments

The authors acknowledge the George Washington University Institute of Public Policy and the Program on Skills, Credentials, and Workforce Policy for institutional support. This project was supported by grants from the Walmart Foundation and the Gates Foundation; the funders had no role in the design of the software or the writing of this paper. The authors thank the GW Open Source Program Office, the MIT Digital Credentials Consortium, and the developers of HuggingFace Transformers, FAISS, and spaCy.

# References
