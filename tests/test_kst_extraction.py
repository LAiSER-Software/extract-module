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


@pytest.mark.library
def test_kst_extraction_with_enables_edges():
    """
    Test that return_edges=True returns ENABLES edges (Knowledge → Task co-occurrence per skill).
    """
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        pytest.skip("Gemini API key not set")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    extractor = SkillExtractorRefactored(model_id="gemini", use_gpu=False, api_key=api_key)

    output = extractor.extract_and_align(
        data=SAMPLE_JOB,
        id_column="Research ID",
        text_columns=["description"],
        extract=["skills", "knowledge", "tasks"],
        warnings=True,
        return_edges=True,
    )

    # With return_edges=True, output must be a dict with "nodes" and "edges"
    assert isinstance(output, dict), "Expected dict when return_edges=True"
    assert "nodes" in output, "Missing 'nodes' key"
    assert "edges" in output, "Missing 'edges' key"

    nodes = output["nodes"]
    edges = output["edges"]

    # Nodes DataFrame: same assertions as the base test
    assert len(nodes) > 0
    assert "Type" in nodes.columns

    # Edges DataFrame: if knowledge and tasks were extracted, edges must be non-empty
    knowledge_rows = nodes[nodes["Type"] == "knowledge"]
    task_rows = nodes[nodes["Type"] == "task"]

    if not knowledge_rows.empty and not task_rows.empty:
        assert not edges.empty, "Expected ENABLES edges when both knowledge and tasks are present"
        required_edge_cols = {"Research ID", "Skill", "Knowledge", "Task", "Edge Type", "confidence"}
        assert required_edge_cols.issubset(set(edges.columns)), f"Missing edge columns. Got: {list(edges.columns)}"
        assert edges["Edge Type"].eq("ENABLES").all(), "All edges must have Edge Type == 'ENABLES'"

        print(f"\n--- ENABLES EDGES ({len(edges)}) ---")
        for _, row in edges.iterrows():
            print(f"  [{row['Skill']}]  {row['Knowledge']}  ──ENABLES──►  {row['Task']}")
