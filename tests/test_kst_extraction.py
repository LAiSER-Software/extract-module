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

    results = extractor.extract_and_align(
        data=SAMPLE_JOB,
        id_column="Research ID",
        text_columns=["description"],
        extract=["skills", "knowledge", "tasks"],
        warnings=True,
    )

    assert results is not None
    assert len(results) > 0
    assert "Type" in results.columns

    for type_label in ["skill", "knowledge", "task"]:
        subset = results[results["Type"] == type_label]
        col_map = {
            "skill": ("Raw Skill", "Taxonomy Skill"),
            "knowledge": ("Raw Knowledge", "Taxonomy Knowledge"),
            "task": ("Raw Task", "Taxonomy Task"),
        }
        raw_col, tax_col = col_map[type_label]

        print(f"\n--- {type_label.upper()}S ({len(subset)}) ---")
        for _, row in subset.iterrows():
            raw = row.get(raw_col, "")
            tax = row.get(tax_col, "")
            score = row.get("Correlation Coefficient", 0)
            score_str = f"  [{score:.2f}]" if score else ""
            print(f"  {raw}  →  {tax}{score_str}")
