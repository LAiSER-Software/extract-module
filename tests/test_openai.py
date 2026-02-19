import os
import pytest
import time

# ✅ import from wherever you saved that function
# Example:
# from laiser.llm_models.openai_helper import openai_generate
from laiser.llm_models.openai import openai_generate

raw_description = (
    "POSITION SUMMARY: This position requires curriculum development, claim processing, "
    "and provider data services experience.\n\n"
    "RESPONSIBILITIES:\n"
    "- Design and deliver training modules for new hires and existing staff.\n"
    "- Collaborate with subject matter experts to create engaging learning materials.\n"
    "- Review claims workflows and develop simulations for hands-on practice.\n"
    "- Analyze provider data services to identify areas for process improvement.\n\n"
    "QUALIFICATIONS:\n"
    "- Bachelor's degree in Healthcare Administration, Education, or related field.\n"
    "- 3+ years of experience in claims processing and curriculum development.\n"
    "- Excellent communication and analytical skills."
)

standard_prompt = f"""
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
        {raw_description}

        ### OUTPUT FORMAT
        {{
        "skills": [
            "skill1",
            "skill2",
            "skill3"
        ]
        }}
        """

@pytest.mark.openai
def test_openai_generate_once():
    # 1) explicit opt-in so it doesn’t run accidentally
    prompt = 'Return ONLY this JSON: {"skills": ["Python"]}'

    # 3) small retry to survive 429 bursts
    resp_text = ""
    resp_text = openai_generate(prompt=standard_prompt,api_key=os.getenv("OPENAI_API_KEY"),)
  

    assert resp_text.strip(), "Empty response from OpenAI"
    print(resp_text)
