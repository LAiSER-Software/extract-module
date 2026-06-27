import io
from contextlib import redirect_stdout
from unittest.mock import Mock

import pandas as pd

from laiser.extraction import PromptBuilder, SkillExtractionService
from laiser.extractor import SkillExtractorRefactored


def test_extract_concepts_maps_user_allowed_sources_to_taxonomy_families():
    extractor = SkillExtractorRefactored.__new__(SkillExtractorRefactored)
    extractor.skill_service = Mock()
    extractor.skill_service.extract_and_align_core.return_value = "ok"

    result = extractor.extract_concepts(
        data="dataframe-placeholder",
        concepts=["skills", "knowledge", "tasks"],
        allowed_sources=["esco", "onet", "osn"],
        warnings=True,
    )

    assert result == "ok"
    kwargs = extractor.skill_service.extract_and_align_core.call_args.kwargs
    assert kwargs["allowed_sources"] == [
        "esco",
        "esco_knowledge",
        "esco_task",
        "onet_skill",
        "onet_tech",
        "onet_knowledge",
        "onet_task",
        "osn",
    ]


def test_extract_concepts_uses_concepts_parameter_name():
    extractor = SkillExtractorRefactored.__new__(SkillExtractorRefactored)
    extractor.skill_service = Mock()
    extractor.skill_service.extract_and_align_core.return_value = "ok"

    result = extractor.extract_concepts(
        data="dataframe-placeholder",
        concepts=["skills", "knowledge", "tasks"],
        warnings=True,
    )

    assert result == "ok"
    kwargs = extractor.skill_service.extract_and_align_core.call_args.kwargs
    assert kwargs["extract"] == ["skills", "knowledge", "tasks"]


def test_extract_concepts_extract_alias_still_supported():
    extractor = SkillExtractorRefactored.__new__(SkillExtractorRefactored)
    extractor.skill_service = Mock()
    extractor.skill_service.extract_and_align_core.return_value = "ok"

    result = extractor.extract_concepts(
        data="dataframe-placeholder",
        extract=["all"],
        warnings=True,
    )

    assert result == "ok"
    kwargs = extractor.skill_service.extract_and_align_core.call_args.kwargs
    assert kwargs["extract"] == ["all"]


def test_extract_concepts_quant_alias_enables_timing():
    extractor = SkillExtractorRefactored.__new__(SkillExtractorRefactored)
    extractor.skill_service = Mock()
    extractor.skill_service.extract_and_align_core.return_value = "ok"

    result = extractor.extract_concepts(
        data="dataframe-placeholder",
        extract=["all"],
        quant=True,
    )

    assert result == "ok"
    kwargs = extractor.skill_service.extract_and_align_core.call_args.kwargs
    assert kwargs["timing"] is True


def test_extract_and_align_quant_alias_enables_timing():
    extractor = SkillExtractorRefactored.__new__(SkillExtractorRefactored)
    extractor.skill_service = Mock()
    extractor.skill_service.extract_and_align_core.return_value = "ok"

    result = extractor.extract_and_align(
        data="dataframe-placeholder",
        quant=True,
    )

    assert result == "ok"
    kwargs = extractor.skill_service.extract_and_align_core.call_args.kwargs
    assert kwargs["timing"] is True


def test_extract_concepts_rejects_conflicting_concepts_and_extract():
    extractor = SkillExtractorRefactored.__new__(SkillExtractorRefactored)
    extractor.skill_service = Mock()

    try:
        extractor.extract_concepts(
            data="dataframe-placeholder",
            concepts=["skills"],
            extract=["all"],
        )
    except ValueError as exc:
        assert "either `concepts` or `extract`" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting concepts/extract arguments")


def test_extract_raw_llm_skills_respects_input_type():
    service = SkillExtractionService.__new__(SkillExtractionService)
    service.prompt_builder = Mock()
    service.prompt_builder.normalize_input_type.side_effect = lambda value: "syllabus"
    service.prompt_builder.build_skill_extraction_prompt.return_value = "prompt"
    service.router = Mock()
    service.router.generate.return_value = '{"skills": ["syllabus planning"]}'
    service.llm_parser = Mock()
    service.llm_parser._parse_skills_from_response.return_value = ["syllabus planning"]

    skills = service.extract_raw_llm_skills(
        input_data={"description": "Course description", "learning_outcomes": "Learning outcomes"},
        text_columns=["description", "learning_outcomes"],
        input_type="syllabus",
    )

    assert skills == ["syllabus planning"]
    service.prompt_builder.build_skill_extraction_prompt.assert_called_once_with(
        input_text={
            "description": "Course description",
            "learning_outcomes": "Learning outcomes",
        },
        input_type="syllabus",
    )


def test_prompt_builder_normalizes_course_syllabi_alias():
    assert PromptBuilder.normalize_input_type("course_syllabi") == "syllabus"
    assert PromptBuilder.normalize_input_type("course_syllabus") == "syllabus"
    assert PromptBuilder.normalize_input_type("job_desc") == "job_desc"


def test_extract_and_align_core_keeps_top_k_per_document_and_type():
    service = SkillExtractionService.__new__(SkillExtractionService)
    service._deduplicate = lambda items, semantic_threshold=0.92: items
    service.extract_raw_llm_skills = lambda input_data, text_columns, input_type="job_desc": ["python", "sql", "ml"]
    service.extract_raw_llm_knowledge_tasks = lambda input_data, text_columns, extracted_skills: [
        {"skill": "python", "knowledge": ["statistics", "probability"], "tasks": ["build models"]}
    ]
    service.align_extracted_skills = lambda *args, **kwargs: pd.DataFrame(
        [
            {
                "Research ID": "job-1",
                "Raw Skill": "python",
                "Taxonomy Skill": "Python",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/python",
                "Correlation Coefficient": 0.91,
            },
            {
                "Research ID": "job-1",
                "Raw Skill": "sql",
                "Taxonomy Skill": "SQL",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/sql",
                "Correlation Coefficient": 0.84,
            },
            {
                "Research ID": "job-1",
                "Raw Skill": "ml",
                "Taxonomy Skill": "Machine learning",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/ml",
                "Correlation Coefficient": 0.62,
            },
        ]
    )
    service.align_extracted_knowledge = lambda *args, **kwargs: pd.DataFrame(
        [
            {
                "Research ID": "job-1",
                "Raw Knowledge": "statistics",
                "Taxonomy Knowledge": "Statistics",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/statistics",
                "Correlation Coefficient": 0.88,
            },
            {
                "Research ID": "job-1",
                "Raw Knowledge": "probability",
                "Taxonomy Knowledge": "Probability",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/probability",
                "Correlation Coefficient": 0.57,
            },
        ]
    )
    service.align_extracted_tasks = lambda *args, **kwargs: pd.DataFrame(
        [
            {
                "Research ID": "job-1",
                "Raw Task": "build models",
                "Taxonomy Task": "Build predictive models",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/build-models",
                "Correlation Coefficient": 0.79,
            }
        ]
    )

    data = pd.DataFrame([{"Research ID": "job-1", "description": "Role text"}])
    results = service.extract_and_align_core(
        data=data,
        id_column="Research ID",
        text_columns=["description"],
        extract=["skills", "knowledge", "tasks"],
        top_k=3,
        warnings=False,
    )

    assert len(results) == 6
    assert set(results["Type"]) == {"skill", "knowledge", "task"}
    assert results["Correlation Coefficient"].tolist() == [0.91, 0.84, 0.62, 0.88, 0.57, 0.79]


def test_extract_and_align_core_prints_stage_timing_when_enabled():
    service = SkillExtractionService.__new__(SkillExtractionService)
    service._deduplicate = lambda items, semantic_threshold=0.92: items
    service.extract_raw_llm_skills = lambda input_data, text_columns, input_type="job_desc": ["python"]
    service.extract_raw_llm_knowledge_tasks = lambda input_data, text_columns, extracted_skills: [
        {"skill": "python", "knowledge": ["statistics"], "tasks": ["build models"]}
    ]
    service.align_extracted_skills = lambda *args, **kwargs: pd.DataFrame(
        [
            {
                "Research ID": "job-1",
                "Raw Skill": "python",
                "Taxonomy Skill": "Python",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/python",
                "Correlation Coefficient": 0.91,
            }
        ]
    )
    service.align_extracted_knowledge = lambda *args, **kwargs: pd.DataFrame(
        [
            {
                "Research ID": "job-1",
                "Raw Knowledge": "statistics",
                "Taxonomy Knowledge": "Statistics",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/statistics",
                "Correlation Coefficient": 0.88,
            }
        ]
    )
    service.align_extracted_tasks = lambda *args, **kwargs: pd.DataFrame(
        [
            {
                "Research ID": "job-1",
                "Raw Task": "build models",
                "Taxonomy Task": "Build predictive models",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/build-models",
                "Correlation Coefficient": 0.79,
            }
        ]
    )

    data = pd.DataFrame([{"Research ID": "job-1", "description": "Role text"}])
    capture = io.StringIO()

    with redirect_stdout(capture):
        service.extract_and_align_core(
            data=data,
            id_column="Research ID",
            text_columns=["description"],
            extract=["skills", "knowledge", "tasks"],
            warnings=False,
            timing=True,
        )

    output = capture.getvalue()
    assert "[TIMING] doc=job-1 stage=extract_1_skills" in output
    assert "[TIMING] doc=job-1 stage=align_skills" in output
    assert "[TIMING] doc=job-1 stage=extract_2_knowledge_tasks" in output
    assert "[TIMING] doc=job-1 stage=deduplicate_knowledge" in output
    assert "[TIMING] doc=job-1 stage=deduplicate_tasks" in output
    assert "[TIMING] doc=job-1 stage=align_knowledge" in output
    assert "[TIMING] doc=job-1 stage=align_tasks" in output
    assert "[TIMING] doc=job-1 stage=document_total" in output
    assert "[TIMING] batch_total" in output


def test_extract_and_align_core_writes_csv_only_when_requested(tmp_path):
    service = SkillExtractionService.__new__(SkillExtractionService)
    service._deduplicate = lambda items, semantic_threshold=0.92: items
    service.extract_raw_llm_skills = lambda input_data, text_columns, input_type="job_desc": ["python"]
    service.extract_raw_llm_knowledge_tasks = lambda input_data, text_columns, extracted_skills: []
    service.align_extracted_skills = lambda *args, **kwargs: pd.DataFrame(
        [
            {
                "Research ID": "job-1",
                "Raw Skill": "python",
                "Taxonomy Skill": "Python",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/python",
                "Correlation Coefficient": 0.91,
            }
        ]
    )

    data = pd.DataFrame([{"Research ID": "job-1", "description": "Role text"}])
    csv_path = tmp_path / "alignment.csv"

    results = service.extract_and_align_core(
        data=data,
        id_column="Research ID",
        text_columns=["description"],
        extract=["skills"],
        warnings=False,
    )
    assert len(results) == 1
    assert not csv_path.exists()

    results = service.extract_and_align_core(
        data=data,
        id_column="Research ID",
        text_columns=["description"],
        extract=["skills"],
        warnings=False,
        output_csv_path=str(csv_path),
    )
    assert len(results) == 1
    assert csv_path.exists()


def test_extract_and_align_core_passes_allowed_sources_to_skill_alignment():
    service = SkillExtractionService.__new__(SkillExtractionService)
    captured = {}

    service._deduplicate = lambda items, semantic_threshold=0.92: items
    service.extract_raw_llm_skills = lambda input_data, text_columns, input_type="job_desc": ["python"]
    service.extract_raw_llm_knowledge_tasks = lambda input_data, text_columns, extracted_skills: []

    def _align_skills(*args, **kwargs):
        captured["allowed_sources"] = kwargs.get("allowed_sources")
        return pd.DataFrame(
            [
                {
                    "Research ID": "job-1",
                    "Raw Skill": "python",
                    "Taxonomy Skill": "Python",
                    "Taxonomy Description": "",
                    "Taxonomy Source": "esco",
                    "Source Url": "https://example.com/python",
                    "Correlation Coefficient": 0.91,
                }
            ]
        )

    service.align_extracted_skills = _align_skills

    data = pd.DataFrame([{"Research ID": "job-1", "description": "Role text"}])
    results = service.extract_and_align_core(
        data=data,
        id_column="Research ID",
        text_columns=["description"],
        extract=["skills"],
        allowed_sources=["esco"],
        warnings=False,
    )

    assert captured["allowed_sources"] == ["esco"]
    assert len(results) == 1
    assert results.iloc[0]["Taxonomy Source"] == "esco"


def test_extract_and_align_core_passes_allowed_sources_to_knowledge_and_task_alignment():
    service = SkillExtractionService.__new__(SkillExtractionService)
    captured = {}

    service._deduplicate = lambda items, semantic_threshold=0.92: items
    service.extract_raw_llm_skills = lambda input_data, text_columns, input_type="job_desc": ["python"]
    service.extract_raw_llm_knowledge_tasks = lambda input_data, text_columns, extracted_skills: [
        {"skill": "python", "knowledge": ["statistics"], "tasks": ["build models"]}
    ]
    service.align_extracted_skills = lambda *args, **kwargs: pd.DataFrame(
        [
            {
                "Research ID": "job-1",
                "Raw Skill": "python",
                "Taxonomy Skill": "Python",
                "Taxonomy Description": "",
                "Taxonomy Source": "esco",
                "Source Url": "https://example.com/python",
                "Correlation Coefficient": 0.91,
            }
        ]
    )

    def _align_knowledge(*args, **kwargs):
        captured["knowledge_sources"] = kwargs.get("allowed_sources")
        return pd.DataFrame(
            [
                {
                    "Research ID": "job-1",
                    "Raw Knowledge": "statistics",
                    "Taxonomy Knowledge": "Statistics",
                    "Taxonomy Description": "",
                    "Taxonomy Source": "esco_knowledge",
                    "Source Url": "https://example.com/statistics",
                    "Correlation Coefficient": 0.88,
                }
            ]
        )

    def _align_tasks(*args, **kwargs):
        captured["task_sources"] = kwargs.get("allowed_sources")
        return pd.DataFrame(
            [
                {
                    "Research ID": "job-1",
                    "Raw Task": "build models",
                    "Taxonomy Task": "Build predictive models",
                    "Taxonomy Description": "",
                    "Taxonomy Source": "esco_task",
                    "Source Url": "https://example.com/build-models",
                    "Correlation Coefficient": 0.79,
                }
            ]
        )

    service.align_extracted_knowledge = _align_knowledge
    service.align_extracted_tasks = _align_tasks

    data = pd.DataFrame([{"Research ID": "job-1", "description": "Role text"}])
    results = service.extract_and_align_core(
        data=data,
        id_column="Research ID",
        text_columns=["description"],
        extract=["skills", "knowledge", "tasks"],
        allowed_sources=["esco", "esco_knowledge", "esco_task"],
        warnings=False,
    )

    assert captured["knowledge_sources"] == ["esco", "esco_knowledge", "esco_task"]
    assert captured["task_sources"] == ["esco", "esco_knowledge", "esco_task"]
    assert len(results) == 3


def test_align_extracted_skills_passes_allowed_sources_to_alignment_service():
    service = SkillExtractionService.__new__(SkillExtractionService)
    service.alignment_service = Mock()
    service.alignment_service.align_skills_to_taxonomy.return_value = "ok"

    result = service.align_extracted_skills(
        raw_skills=["python"],
        document_id="job-1",
        description="Role text",
        similarity_threshold=0.2,
        top_k=5,
        allowed_sources=["esco"],
    )

    assert result == "ok"
    service.alignment_service.align_skills_to_taxonomy.assert_called_once()
    kwargs = service.alignment_service.align_skills_to_taxonomy.call_args.kwargs
    assert kwargs["allowed_sources"] == ["esco"]
