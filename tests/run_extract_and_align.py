import pandas as pd
from laiser.skill_extractor_refactored import SkillExtractorRefactored
data = pd.DataFrame([
    {
        "Research ID": "aetna_trainer_001",
        "description": (
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
    },
    {
        "Research ID": "aetna_trainer_002",
        "description": (
            "Looking for someone with a strong technical training background and knowledge of "
            "Medicaid state websites.\n\n"
            "RESPONSIBILITIES:\n"
            "- Conduct training sessions for teams navigating state Medicaid portals.\n"
            "- Develop technical documentation and job aids for internal users.\n"
            "- Troubleshoot common issues users face when accessing Medicaid websites.\n"
            "- Coordinate with IT teams to ensure training content is up-to-date with policy changes.\n\n"
            "QUALIFICATIONS:\n"
            "- Experience in healthcare IT systems or state Medicaid platforms.\n"
            "- Prior technical training or instructional design experience preferred.\n"
            "- Strong problem-solving skills and ability to work with cross-functional teams."
        )
    },
])

extractor = SkillExtractorRefactored(
            model_id="gemini",
            api_key="AIzaSyAdYOILwr2YcCJvX7K-I5M8yjwF1wX5bZk",
            use_gpu=False
        )
# Ensure pandas prints everything
pd.set_option("display.max_rows", None)      # show all rows
pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.width", 0)            # auto-detect terminal width
pd.set_option("display.max_colwidth", None)  # don't truncate long strings

results_faiss = extractor.extract_and_align(
            data=data,
            id_column='Research ID',
            text_columns=['description'],
            warnings=True
        )


print(results_faiss)