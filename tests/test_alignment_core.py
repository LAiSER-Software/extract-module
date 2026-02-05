import pandas as pd
import pytest

from laiser.data_access import DataAccessLayer, FAISSIndexManager
from laiser.services import SkillAlignmentService


@pytest.mark.alignment
def test_align_skills_to_taxonomy_real_flow():
    """
    Real alignment test:
    - uses real DataAccessLayer + FAISSIndexManager
    - runs the exact align_skills_to_taxonomy end-to-end
    """
    da = DataAccessLayer()
    fm = FAISSIndexManager(da)

    # Try to init (load/build). If environment is missing deps/data, skip cleanly.
    try:
        fm.initialize_index(force_rebuild=False)
    except Exception as e:
        pytest.skip(f"Skipping alignment test: index init failed: {repr(e)}")

    service = SkillAlignmentService(data_access=da, faiss_manager=fm)

    raw_skills = [
    "Contextual Analysis",
    "Analyze a wide range of business contexts for ethical issues.",
    "Business Ethics Strategies Analysis",
    "Business Context Ethics Analysis",
    "Create a Plan to Achieve Goals",
    "Identify the Benefits of Self-Motivated Goals",
    "Create Self-Motivated Goals",
    "Identify Self-Motivation Activities",
    "Prevent Burnout",
    "Develop Action Plans",
    "Plan a Schedule in Advance",
    "Re-evaluate a Point of View",
    "Collaborate to Solve Challenges",
    "Create a Goal Plan",
    "Evaluate Results Against Predetermined Goals",
    "Implement a Plan",
    "Organize Project into Daily Tasks",
    "Track Task Deadlines",
    "Schedule Due Dates",
    "Define Productivity Processes",
    "Work with a Team",
    "Implement Teamwork-Fostering Processes",
    "Technology Assessment",
    "Define Data to be Organized",
    "Analyze a wide range of business contexts for ethical issues."
]


    df = service.align_skills_to_taxonomy(
        raw_skills=raw_skills,
        document_id="it-1",
        description="alignment test",
        similarity_threshold=0.20,
        top_k=1000,
        debug=False,
    )

    # Show full matching table (already doing this)
    print(
        df[
            [
                "Raw Skill",
                "Taxonomy Skill",
                "Taxonomy Source",
                "Correlation Coefficient",
            ]
        ].to_string(index=False)
    )

    # --- NEW: source breakdown ---
    source_counts = df["Taxonomy Source"].value_counts(dropna=False)

    print("\n--- Taxonomy source counts ---")
    print(source_counts.to_string())

    # Optional explicit numbers
    esco_count = source_counts.get("esco", 0)
    osn_count = source_counts.get("osn", 0)

    print(f"\nESCO matches: {esco_count}")
    print(f"OSN matches: {osn_count}")
    print(f"Total matches: {len(df)}")

    # Minimal, meaningful assertions
    assert isinstance(df, pd.DataFrame)
    assert set(["Raw Skill", "Taxonomy Skill", "Correlation Coefficient"]).issubset(df.columns)

    # For these common skills, we usually expect at least one match if index is healthy.
    # (We don't assert exact mapping because taxonomies may change.)
    assert len(df) >= 1

