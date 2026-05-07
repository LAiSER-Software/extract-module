"""
Diagnostic script: print raw Knowledge and Task extractions before dedup and alignment.

Shows exactly what the LLM returns from KT_FROM_SKILLS_PROMPT so we can see
which task patterns are failing to align and why.

Usage:
    python scripts/diagnose_raw_kt.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from laiser.services import SkillExtractionService  # noqa: E402
from tests.test_helpers import sample_data  # noqa: E402

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY")
    sys.exit(1)

svc = SkillExtractionService(model_id="gemini", use_gpu=False, api_key=api_key)

row = sample_data().iloc[[2]]
input_data = {"description": row["description"].iloc[0], "id": str(row["Research ID"].iloc[0])}
text_columns = ["description"]

# Call 1 — skills
print("=== CALL 1: SKILLS ===")
skills = svc.extract_raw_llm_skills(input_data, text_columns)
for s in skills:
    print(f"  • {s}")

# Call 2 — raw KT before any dedup or alignment
print("\n=== CALL 2: RAW KT PER SKILL ===")
kt_results = svc.extract_raw_llm_knowledge_tasks(input_data, text_columns, skills)

for item in kt_results:
    print(f"\n[{item['skill']}]")
    print("  Knowledge:")
    for k in item.get("knowledge", []):
        print(f"    K: {k}")
    print("  Tasks:")
    for t in item.get("tasks", []):
        print(f"    T: {t}")

# After dedup — before alignment
print("\n=== AFTER DEDUP (pre-alignment) ===")
raw_knowledge = svc._deduplicate([k for item in kt_results for k in item.get("knowledge", [])])
raw_tasks = svc._deduplicate([t for item in kt_results for t in item.get("tasks", [])])

print(f"\nKnowledge ({len(raw_knowledge)} unique):")
for k in raw_knowledge:
    print(f"  K: {k}")

print(f"\nTasks ({len(raw_tasks)} unique):")
for t in raw_tasks:
    print(f"  T: {t}")

# Alignment attempt on tasks — show score for every item regardless of threshold
print("\n=== TASK ALIGNMENT SCORES (no threshold) ===")
if svc.task_faiss.index is not None:
    aligned = svc.task_alignment.align(
        raw_items=raw_tasks,
        document_id="diag",
        similarity_threshold=0.0,  # show everything
        top_k=50,
        raw_col="Raw Task",
        taxonomy_col="Taxonomy Task",
    )
    for _, row_ in aligned.sort_values("Correlation Coefficient", ascending=False).iterrows():
        score = row_["Correlation Coefficient"]
        bar = "✓" if score >= 0.55 else ("~" if score >= 0.45 else "✗")
        print(f"  {bar} [{score:.2f}]  {row_['Raw Task']}")
        print(f"         → {row_['Taxonomy Task']}")
else:
    print("  Task index not available.")
