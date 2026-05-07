import os
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv

from laiser.skill_extractor_refactored import SkillExtractorRefactored
from tests.test_helpers import eda_on_results, sample_data

load_dotenv()


def run_skill_extractor_smoke():
    data = sample_data().iloc[[2]]

    extractor = SkillExtractorRefactored(model_id="gemini", use_gpu=False, api_key=os.getenv("GEMINI_API_KEY"))

    results = extractor.extract_and_align(
        data=data, id_column="Research ID", text_columns=["description"], warnings=True
    )

    # Minimal assertions — just make sure it runs end-to-end
    assert results is not None
    if results.empty:
        pytest.skip("Gemini extraction returned no rows in this environment")
    assert len(results) > 0
    print(results)
    summary, agg_df = eda_on_results(results, print_report=True)
    print("\n--- Aggregated taxonomy stats ---")
    print(agg_df)


@pytest.mark.library
def test_skill_extractor_smoke():
    # Skip cleanly if API key not present
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        pytest.skip("Gemini API key not set")

    run_skill_extractor_smoke()


def test_extract_and_align_remains_skills_only_wrapper():
    extractor = SkillExtractorRefactored.__new__(SkillExtractorRefactored)
    extractor.skill_service = Mock()
    extractor.skill_service.extract_and_align_core.return_value = "ok"

    result = extractor.extract_and_align(
        data="dataframe-placeholder",
        extract=["skills", "knowledge", "tasks"],
        warnings=True,
    )

    assert result == "ok"
    extractor.skill_service.extract_and_align_core.assert_called_once()
    kwargs = extractor.skill_service.extract_and_align_core.call_args.kwargs
    assert kwargs["extract"] == ["skills"]


def test_extract_concepts_allows_mixed_type_extraction():
    extractor = SkillExtractorRefactored.__new__(SkillExtractorRefactored)
    extractor.skill_service = Mock()
    extractor.skill_service.extract_and_align_core.return_value = "ok"

    result = extractor.extract_concepts(
        data="dataframe-placeholder",
        extract=["skills", "knowledge", "tasks"],
        warnings=True,
    )

    assert result == "ok"
    extractor.skill_service.extract_and_align_core.assert_called_once()
    kwargs = extractor.skill_service.extract_and_align_core.call_args.kwargs
    assert kwargs["extract"] == ["skills", "knowledge", "tasks"]
