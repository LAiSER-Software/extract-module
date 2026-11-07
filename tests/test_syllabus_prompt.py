"""Syllabi go through the same extraction prompt as job postings.

Before this, input_type="syllabus" selected a placeholder prompt whose whole
text was " Work in progress", so the model never saw the course text.
"""

from unittest.mock import Mock

import pytest

from laiser.exceptions import InvalidInputError
from laiser.services import PromptBuilder, ResponseParser, SkillExtractionService

COURSE_TEXT = "Foundations of machine learning. Train models and interpret metrics."


@pytest.mark.parametrize("input_type", ["syllabus", "course_syllabus", "course_syllabi"])
def test_syllabus_prompt_matches_job_prompt_and_contains_the_text(input_type):
    syllabus_prompt = PromptBuilder.build_skill_extraction_prompt(COURSE_TEXT, input_type)

    assert COURSE_TEXT in syllabus_prompt
    assert syllabus_prompt == PromptBuilder.build_skill_extraction_prompt(COURSE_TEXT, "job_desc")


def test_unsupported_input_type_still_fails():
    with pytest.raises(InvalidInputError):
        PromptBuilder.build_skill_extraction_prompt(COURSE_TEXT, "resume")


def test_syllabus_extraction_reads_every_text_column():
    service = SkillExtractionService.__new__(SkillExtractionService)
    service.prompt_builder = PromptBuilder()
    service.llm_parser = ResponseParser()
    service.router = Mock()
    service.router.generate.return_value = '{"skills": ["machine learning"]}'

    row = {"description": "Foundations of machine learning.", "learning_outcomes": "Interpret metrics."}
    skills = service.extract_raw_llm_skills(row, ["description", "learning_outcomes"], input_type="syllabus")

    prompt = service.router.generate.call_args.args[0]
    assert "Foundations of machine learning. Interpret metrics." in prompt
    assert skills == ["machine learning"]
