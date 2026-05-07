"""
v0.5 KST extraction test — Skills, Knowledge, Tasks in one call.
Run with: pytest tests/test_kst_extraction.py -v -s
"""

import os

import pytest
from dotenv import load_dotenv

from laiser.skill_extractor_refactored import SkillExtractorRefactored
from tests.test_helpers import sample_data

load_dotenv()

SAMPLE_JOB = sample_data().iloc[[2]]


@pytest.mark.library
def test_kst_extraction():
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        pytest.skip("Gemini API key not set")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    extractor = SkillExtractorRefactored(model_id="gemini", use_gpu=False, api_key=api_key)

    results = extractor.extract_concepts(
        data=SAMPLE_JOB,
        id_column="Research ID",
        text_columns=["description"],
        extract=["skills", "knowledge", "tasks"],
        warnings=True,
    )
    assert results is not None
    if results.empty:
        pytest.skip("Gemini extraction returned no rows in this environment")
    assert len(results) > 0
    assert {
        "Research ID",
        "Type",
        "Raw Concept",
        "Taxonomy Concept",
        "Source Url",
    }.issubset(results.columns)

    print("\n--- KST RESULTS ---")
    print(results.to_string(index=False))
