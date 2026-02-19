"""
Module Description:
-------------------
This module defines configuration constants for the LAiSER project, including model identifiers, URLs, prompt templates, and other settings used throughout the codebase.

Ownership:
----------
Project: Leveraging Artificial intelligence for Skills Extraction and Research (LAiSER)
Owner:  George Washington University Insitute of Public Policy
  Program on Skills, Credentials and Workforce Policy
  Media and Public Affairs Building
  805 21st Street NW
  Washington, DC 20052
  PSCWP@gwu.edu
  https://gwipp.gwu.edu/program-skills-credentials-workforce-policy-pscwp

License:
--------
Copyright 2025 George Washington University Insitute of Public Policy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
(the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Input Requirements:
-------------------
- No direct input required. This module provides constants and templates for use by other modules.

Output/Return Format:
---------------------
- No direct output. Other modules import and use the constants and templates defined here.

"""

"""
Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     08/13/2025      Satya Phanindra K.  Initial Version


TODO:
-----
"""

import os
from typing import Dict, Any

# Model Configuration
DEFAULT_TRANSFORMER_MODEL_ID = "TheBloke/Mixtral-7B-Instruct-v0.1-AWQ"
DEFAULT_VLLM_MODEL_ID = "TheBloke/Mixtral-7B-Instruct-v0.1-AWQ"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SPACY_MODEL = "en_core_web_lg"

# Similarity thresholds
SIMILARITY_THRESHOLD = 0.85

# ESCO Configuration
ESCO_SKILLS_URL = "https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/taxonomies/ESCO_skills_Taxonomy.csv"
OSN_SKILLS_URL = "https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/taxonomies/osn_taxanomy_merged.csv"
COMBINED_SKILLS_URL = "https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/taxonomies/combined.csv"
FAISS_INDEX_URL = "https://raw.githubusercontent.com/LAiSER-Software/extract-module/main/laiser/input/esco_faiss_index.index"

# Batch processing
DEFAULT_BATCH_SIZE = 32
DEFAULT_TOP_K = 25

# Generation parameters
MAX_NEW_TOKENS = 1000
GENERATION_SEED = 42

# SCQF Level Descriptors
SCQF_LEVELS: Dict[int, str] = {
    1: "Basic awareness of simple concepts.",
    2: "Limited operational understanding, guided application.",
    3: "Moderate knowledge, supervised application of techniques.",
    4: "Clear understanding, independent work in familiar contexts.",
    5: "Advanced knowledge, autonomous problem-solving.",
    6: "Specialized knowledge, critical analysis within defined areas.",
    7: "Advanced specialization, leadership in problem-solving.",
    8: "Expert knowledge, innovation in complex contexts.",
    9: "Highly specialized expertise, contributing original thought.",
    10: "Sustained mastery, influential in areas of specialization.",
    11: "Groundbreaking innovation, professional or academic mastery.",
    12: "Global expertise, leading advancements at the highest level."
}

# Prompt templates
SKILL_EXTRACTION_PROMPT_JOB = """
        task: "Skill Extraction from Job Descriptions"

        description: |
        You are an expert AI system specialized in extracting technical and professional skills from job descriptions for workforce analytics.
        Your goal is to analyze the following job description and output only the specific skill names that are required, mentioned, or strongly implied.

        extraction_instructions:
        - Extract only concrete, job-relevant skills (not soft traits, company values, or general workplace behaviors).
        - Include a skill if it is clearly mentioned or strongly implied as necessary for the role.
        - Exclude company policies, benefit programs, HR or legal statements, and generic terms (e.g., "communication," "leadership") unless used in a technical/professional context.
        - Use only concise skill phrases (prefer noun phrases, avoid sentences).
        - Do not invent new skills or make assumptions beyond the provided text.

        examples:
        Example 1 (Focus: Soft Skills & Communication) Input: "Strong verbal and written communication skills, with the ability to explain complex technical concepts clearly to both technical and non-technical audiences. Confident presenter, capable of articulating insights, results, and strategies to stakeholders." Output: ['Strong verbal and written communication skills', 'explain complex technical concepts', 'Confident presenter', 'capable of articulating insights']

        Example 2 (Focus: Math & Technical Background) Input: "Qualified candidates will have a strong mathematical background (statistics, linear algebra, calculus, probability, and optimization). Experience with deep learning, natural language processing, or application of large language models is preferred." Output: ['Strong mathematical background', 'statistics, linear algebra, calculus, probability, and optimization', 'deep learning', 'natural language processing', 'application of large language models']

        Example 3 (Focus: Core Responsibilities) Input: "Lead the research, design, implementation, and deployment of Machine Learning algorithms. Assist and enable C3 AI’s federal customers to build their own applications on the C3 AI Suite. Contribute to the design of new features." Output: ['Lead the research, design, implementation, and deployment of Machine Learning algorithms', 'Assist and enable C3 AI’s federal customers to build their own applications on the C3 AI Suite.', 'Contribute to the design and implementation of new features of the C3 AI Suite.']

        formatting_rules:
        - Return the output as valid JSON.
        - The JSON must have a single key "skills" whose value is a list of skill strings.
        - Each skill string must be between 1 and 5 words.
        - Do not include explanations, metadata, or anything other than the JSON object.

        job_description: |
        {description}

        ### OUTPUT FORMAT
        {{
        "skills": [
            "skill1",
            "skill2",
            "skill3"
        ]
        }}
      """

## ISSUE: Prompt for course syllabi
SKILL_EXTRACTION_PROMPT_SYLLABUS = """ Work in progress"""
KSA_EXTRACTION_PROMPT = """[INST]user
**Objective:** Given a {input_desc}, complete the following tasks with structured outputs.

### Semantic matches from Taxonomy Skills:
{esco_context_block}

### Tasks:
1. **Skills Extraction:** Identify {num_key_skills} key skills mentioned in the {input_desc}.
  - Extract/Filter contextually relevant skill keywords or phrases from taxonomy semantic matches.

2. **Skill Level Assignment:** Assign a proficiency level to each extracted skill based on the SCQF Level Descriptors (see below).

3. **Knowledge Required:** For each skill, list {num_key_kr} broad areas of understanding or expertise necessary to develop the skill.

4. **Task Abilities:** For each skill, list {num_key_tas} general tasks or capabilities enabled by the skill.

### Guidelines:
- **Skill Extraction:** 
    - If the Semantic matches from the taxonomy skills are provided, use them to identify relevant skills.
    - If none of the semantic matches are contextually relevant to the {input_desc}, infer skills from the {input_desc} directly.

- **Skill Level Assignment:** Use the SCQF Level Descriptors to classify proficiency:
{scqf_levels}

- **Knowledge and Task Abilities:**
  - **Knowledge Required:** Broad areas, e.g., "data visualization techniques."
  - **Task Abilities:** General tasks or capabilities, e.g., "data analysis."
  - Each item in these two lists should be no more than three words.
  - Avoid overly specific or vague terms.

### Answer Format:
- Use this format strictly in the response:
  -> Skill: [Skill Name], Level: [1–12], Knowledge Required: [list], Task Abilities: [list].

{input_text}

**Response:** Provide only the requested structured information without additional explanations.

[/INST]
[INST]model
"""

KSA_DETAILS_PROMPT = """[INST]user
Given the following context, provide concise lists for the specified skill.

Skill: {skill}

Context:
{description}

For the skill above produce:
- Knowledge Required: {num_key_kr} bullet items, each ≤ 3 words.
- Task Abilities: {num_key_tas} bullet items, each ≤ 3 words.

Respond strictly in valid JSON with the exact keys 'Knowledge Required' and 'Task Abilities'.
[/INST]
[INST]model"""

# File paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
