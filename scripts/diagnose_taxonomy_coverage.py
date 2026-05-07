"""
Verify taxonomy coverage: do the right concepts exist in the indexes at all?

Two questions:
  Q1. If we query the task index with ideal phrasing, does it return relevant results?
      → If yes: it's an LLM phrasing problem (right concept exists, LLM wording missed it)
      → If no:  it's a taxonomy coverage problem (concept not in index)

  Q2. If we query the knowledge index with ideal phrasing, same question.

Usage:
    python scripts/diagnose_taxonomy_coverage.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from laiser.data_access import DataAccessLayer, KnowledgeFAISSIndexManager, TaskFAISSIndexManager  # noqa: E402

dal = DataAccessLayer()
model = dal.get_embedding_model()

knowledge_mgr = KnowledgeFAISSIndexManager(dal)
knowledge_mgr.initialize_index()

task_mgr = TaskFAISSIndexManager(dal)
task_mgr.initialize_index()


def search(manager, queries, top_k=5):
    meta = manager.get_metadata()
    for q in queries:
        vec = model.encode([q], normalize_embeddings=True)
        results = manager.search_similar(np.array(vec).astype("float32"), top_k=top_k)
        print(f"\n  Query: '{q}'")
        for r in results:
            idx = r["Index"]
            name = r["Name"]
            score = r["Similarity"]
            src = meta.iloc[idx].get("taxonomy", "") if idx < len(meta) else ""
            print(f"    [{score:.2f}] ({src})  {name}")


# ─────────────────────────────────────────────────────────────────
# TASK INDEX: query with ideal phrasings for this data science JD
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("TASK INDEX — ideal queries for a data science / consulting JD")
print("=" * 70)

task_queries = [
    # What we expect to exist in O*NET for this kind of role
    "train machine learning models",
    "perform statistical analysis",
    "write Python code",
    "deploy models to production",
    "create data visualizations",
    "formulate optimization problems",
    "analyze datasets",
    "build predictive models",
    "design algorithms",
    "present findings to clients",
]

search(task_mgr, task_queries, top_k=3)

# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE INDEX: query with ideal phrasings
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KNOWLEDGE INDEX — ideal queries for a data science / consulting JD")
print("=" * 70)

knowledge_queries = [
    "machine learning",
    "statistical inference",
    "linear algebra",
    "probability theory",
    "optimization theory",
    "data structures",
    "Python programming",
    "software engineering",
    "data visualization",
    "operations research",
]

search(knowledge_mgr, knowledge_queries, top_k=3)

# ─────────────────────────────────────────────────────────────────
# DIRECT COMPARISON: LLM output vs ideal query — same concept?
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIRECT COMPARISON — LLM task output vs ideal query for same concept")
print("(If top-1 match is the same → LLM phrasing problem)")
print("(If top-1 match is different → taxonomy gap)")
print("=" * 70)

comparisons = [
    ("LLM output", "Ideal query"),
    ("Design and train machine learning models for predictive analytics tasks", "train machine learning models"),
    (
        "Aggregate and summarize complex datasets using descriptive statistical measures",
        "perform statistical analysis on datasets",
    ),
    (
        "Write clean, maintainable Python code for data processing and model development",
        "write Python code for data analysis",
    ),
    (
        "Deploy machine learning models into production systems to generate business impact",
        "deploy models to production systems",
    ),
    (
        "Design clear, informative visualizations to communicate complex analytical results",
        "create data visualizations to communicate results",
    ),
    (
        "Formulate business challenges into mathematical optimization problems",
        "formulate mathematical optimization models",
    ),
]

meta = task_mgr.get_metadata()
for llm_q, ideal_q in comparisons[1:]:
    llm_vec = model.encode([llm_q], normalize_embeddings=True)
    ideal_vec = model.encode([ideal_q], normalize_embeddings=True)

    llm_top = task_mgr.search_similar(np.array(llm_vec).astype("float32"), top_k=1)
    ideal_top = task_mgr.search_similar(np.array(ideal_vec).astype("float32"), top_k=1)

    llm_match = llm_top[0] if llm_top else {}
    ideal_match = ideal_top[0] if ideal_top else {}

    same = llm_match.get("Name", "") == ideal_match.get("Name", "")
    verdict = "SAME MATCH → LLM phrasing problem" if same else "DIFFERENT MATCH → taxonomy gap or LLM drift"

    print(f"\n  Concept: {ideal_q}")
    print(f"  LLM  [{llm_match.get('Similarity', 0):.2f}]: {llm_match.get('Name', '')[:80]}")
    print(f"  Ideal[{ideal_match.get('Similarity', 0):.2f}]: {ideal_match.get('Name', '')[:80]}")
    print(f"  → {verdict}")
