import logging
from typing import Dict, List

from laiser.config import (
    COMBINED_EXTRACTION_PROMPT,
    KSA_DETAILS_PROMPT,
    KSA_EXTRACTION_PROMPT,
    KT_FROM_SKILLS_PROMPT,
    SCQF_LEVELS,
    SKILL_EXTRACTION_PROMPT_SYLLABUS,
)
from laiser.exceptions import InvalidInputError

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build prompts for the supported extraction flows."""

    @staticmethod
    def normalize_input_type(input_type: str) -> str:
        normalized = str(input_type or "").strip().lower()
        aliases = {
            "job_desc": "job_desc",
            "job_description": "job_desc",
            "syllabus": "syllabus",
            "course_syllabus": "syllabus",
            "course_syllabi": "syllabus",
        }
        if normalized not in aliases:
            raise InvalidInputError(f"Unsupported input type: {input_type}")
        return aliases[normalized]

    @staticmethod
    def build_skill_extraction_prompt(input_text: str, input_type: str) -> str:
        input_type = PromptBuilder.normalize_input_type(input_type)
        if input_type == "job_desc":
            return COMBINED_EXTRACTION_PROMPT.format(description=input_text)
        if input_type == "syllabus":
            return SKILL_EXTRACTION_PROMPT_SYLLABUS.format(
                description=input_text.get("description", ""),
                learning_outcomes=input_text.get("learning_outcomes", ""),
            )
        raise InvalidInputError(f"Unsupported input type: {input_type}")

    @staticmethod
    def build_ksa_extraction_prompt(
        query: Dict[str, str],
        input_type: str,
        num_key_skills: int,
        num_key_kr: str,
        num_key_tas: str,
        esco_skills: List[str] = None,
    ) -> str:
        input_type = PromptBuilder.normalize_input_type(input_type)
        input_desc = (
            "job description" if input_type == "job_desc" else "course syllabus description and its learning outcomes"
        )

        if input_type == "syllabus":
            input_text = (
                f"### Input:\\n**Course Description:** {query.get('description', '')}"
                f"\\n**Learning Outcomes:** {query.get('learning_outcomes', '')}"
            )
        else:
            input_text = f"### Input:\\n{query.get('description', '')}"

        scqf_levels_text = "\\n".join([f"  - {level}: {desc}" for level, desc in SCQF_LEVELS.items()])
        esco_context_block = ", ".join(esco_skills) if esco_skills else "No relevant skills found in taxonomy"

        return KSA_EXTRACTION_PROMPT.format(
            input_desc=input_desc,
            num_key_skills=num_key_skills,
            num_key_kr=num_key_kr,
            num_key_tas=num_key_tas,
            input_text=input_text,
            esco_context_block=esco_context_block,
            scqf_levels=scqf_levels_text,
        )

    @staticmethod
    def build_ksa_details_prompt(skill: str, description: str, num_key_kr: int = 3, num_key_tas: int = 3) -> str:
        return KSA_DETAILS_PROMPT.format(
            skill=skill,
            description=description,
            num_key_kr=num_key_kr,
            num_key_tas=num_key_tas,
        )

    @staticmethod
    def build_knowledge_task_prompt(description: str, extracted_skills: List[str]) -> str:
        skills_formatted = "\n".join(f"- {s}" for s in extracted_skills)
        return KT_FROM_SKILLS_PROMPT.format(
            description=description,
            skills=skills_formatted,
        )

    def strong_preprocessing_prompt(self, raw_description):
        raise NotImplementedError("strong_preprocessing_prompt is not yet implemented. Fix llm router params.")
