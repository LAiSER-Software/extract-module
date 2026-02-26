import os
import pandas as pd
import pytest

from laiser.skill_extractor_refactored import SkillExtractorRefactored
from tests.test_helpers import eda_on_results, sample_data

from dotenv import load_dotenv
load_dotenv()
def run_skill_extractor_smoke():
    data = sample_data().iloc[[2]]

    extractor = SkillExtractorRefactored(
        model_id="gemini",
        use_gpu=False,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    results = extractor.extract_and_align(
        data=data,
        id_column="Research ID",
        text_columns=["description"],
        warnings=True
    )

    # Minimal assertions — just make sure it runs end-to-end
    assert results is not None
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
