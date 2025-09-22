import pandas as pd
from laiser.skill_extractor_refactored import SkillExtractorRefactored
data = pd.DataFrame([
        {
            "Research ID": "aetna_trainer_001",
            "description": "POSITION SUMMARY: This position requires curriculum development, claim processing, and provider data services experience."
        },
        {
            "Research ID": "aetna_trainer_002",
            "description": "Looking for someone with technical training background and knowledge of Medicaid state websites."
        },
    ])
extractor = SkillExtractorRefactored(
            model_id="gemini",
            api_key="AIzaSyB7nzaQqx6BMcSuZKn_Ptbrhcj0_t4us14",
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