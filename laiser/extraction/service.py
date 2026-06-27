import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from laiser._internal.llm_router import LLMRouter
from laiser.alignment.service import AlignmentService
from laiser.config import DEFAULT_BATCH_SIZE, DEFAULT_SIMILARITY_THRESHOLDS, DEFAULT_TOP_K
from laiser.exceptions import InvalidInputError, LAiSERError
from laiser.extraction.prompt_builder import PromptBuilder
from laiser.extraction.response_parser import ResponseParser
from laiser.taxonomy.index import FAISSIndexManager, KnowledgeFAISSIndexManager, TaskFAISSIndexManager
from laiser.taxonomy.loader import DataAccessLayer

logger = logging.getLogger(__name__)


class SkillExtractionService:
    """Main service for skill extraction operations."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        api_key: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        backend: Optional[str] = None,
    ):
        self.model_id = model_id
        self.hf_token = hf_token
        self.api_key = api_key
        self.use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        self.backend = backend
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.nlp = None
        self.data_access = DataAccessLayer()

        self.faiss_manager = FAISSIndexManager(self.data_access)
        self.faiss_manager.initialize_index(force_rebuild=False)
        self.alignment_service = AlignmentService(self.data_access, self.faiss_manager)

        self.knowledge_faiss = KnowledgeFAISSIndexManager(self.data_access)
        self.knowledge_faiss.initialize_index(force_rebuild=False)
        self.knowledge_alignment = AlignmentService(self.data_access, self.knowledge_faiss)

        self.task_faiss = TaskFAISSIndexManager(self.data_access)
        self.task_faiss.initialize_index(force_rebuild=False)
        self.task_alignment = AlignmentService(self.data_access, self.task_faiss)

        self.prompt_builder = PromptBuilder()
        self.llm_parser = ResponseParser()
        self.router = LLMRouter(
            self.model_id,
            self.use_gpu,
            self.hf_token,
            self.api_key,
            backend=self.backend,
        )

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def extract_and_align_core(
        self,
        data: pd.DataFrame,
        id_column: str = "Research ID",
        text_columns: List[str] = None,
        input_type: str = "job_desc",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        levels: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        warnings: bool = True,
        allowed_sources: Optional[List[str]] = None,
        extract: List[str] = None,
        return_edges: bool = False,
        similarity_thresholds: Optional[Dict[str, float]] = None,
        timing: bool = False,
        quant: bool = False,
        output_csv_path: Optional[str] = None,
    ):
        if quant:
            timing = True

        if text_columns is None:
            text_columns = ["description"]

        if extract is None:
            extract = ["skills"]
        if extract == ["all"] or extract == "all":
            extract = ["skills", "knowledge", "tasks"]
        extract = [e.lower().strip() for e in extract]

        if data is None:
            raise InvalidInputError(
                "extract_and_align_core: `data` is None. Please pass a pandas.DataFrame with rows to process."
            )
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        effective_top_k = top_k if top_k is not None else DEFAULT_TOP_K
        resolved: Dict[str, float] = dict(DEFAULT_SIMILARITY_THRESHOLDS)
        if similarity_threshold is not None:
            resolved = {k: similarity_threshold for k in resolved}
        if similarity_thresholds:
            resolved.update({k: float(v) for k, v in similarity_thresholds.items()})

        try:
            batch_started = time.perf_counter()
            results = []
            all_edges: List[pd.DataFrame] = []

            for idx, row in data.iterrows():
                try:
                    doc_started = time.perf_counter()
                    input_data = {col: row.get(col, "") for col in text_columns}
                    input_data["id"] = row.get(id_column, str(idx))
                    full_description = " ".join([str(input_data.get(col, "")) for col in text_columns])
                    doc_id = str(input_data["id"])

                    stage_started = time.perf_counter()
                    skills = self.extract_raw_llm_skills(input_data, text_columns, input_type=input_type)
                    if timing:
                        print(
                            f"[TIMING] doc={doc_id} stage=extract_1_skills took={time.perf_counter() - stage_started:.3f}s | count={len(skills)}"
                        )

                    if "skills" in extract:
                        stage_started = time.perf_counter()
                        aligned_skills = self.align_extracted_skills(
                            skills,
                            doc_id,
                            full_description,
                            similarity_threshold=resolved["skill"],
                            top_k=effective_top_k,
                            allowed_sources=allowed_sources,
                        )
                        aligned_skills["Type"] = "skill"
                        results.extend(aligned_skills.to_dict("records"))
                        if timing:
                            print(
                                f"[TIMING] doc={doc_id} stage=align_skills took={time.perf_counter() - stage_started:.3f}s | count={len(aligned_skills)}"
                            )

                    if "knowledge" in extract or "tasks" in extract:
                        stage_started = time.perf_counter()
                        kt_results = self.extract_raw_llm_knowledge_tasks(input_data, text_columns, skills)
                        if timing:
                            print(
                                f"[TIMING] doc={doc_id} stage=extract_2_knowledge_tasks took={time.perf_counter() - stage_started:.3f}s | count={len(kt_results)}"
                            )

                        stage_started = time.perf_counter()
                        raw_knowledge = self._deduplicate([k for item in kt_results for k in item.get("knowledge", [])])
                        if timing:
                            print(
                                f"[TIMING] doc={doc_id} stage=deduplicate_knowledge took={time.perf_counter() - stage_started:.3f}s | count={len(raw_knowledge)}"
                            )

                        stage_started = time.perf_counter()
                        raw_tasks = self._deduplicate([t for item in kt_results for t in item.get("tasks", [])])
                        if timing:
                            print(
                                f"[TIMING] doc={doc_id} stage=deduplicate_tasks took={time.perf_counter() - stage_started:.3f}s | count={len(raw_tasks)}"
                            )

                        if "knowledge" in extract and raw_knowledge:
                            stage_started = time.perf_counter()
                            aligned_knowledge = self.align_extracted_knowledge(
                                raw_knowledge,
                                doc_id,
                                full_description,
                                similarity_threshold=resolved["knowledge"],
                                top_k=effective_top_k,
                                allowed_sources=allowed_sources,
                            )
                            aligned_knowledge["Type"] = "knowledge"
                            results.extend(aligned_knowledge.to_dict("records"))
                            if timing:
                                print(
                                    f"[TIMING] doc={doc_id} stage=align_knowledge took={time.perf_counter() - stage_started:.3f}s | count={len(aligned_knowledge)}"
                                )

                        if "tasks" in extract and raw_tasks:
                            stage_started = time.perf_counter()
                            aligned_tasks = self.align_extracted_tasks(
                                raw_tasks,
                                doc_id,
                                full_description,
                                similarity_threshold=resolved["task"],
                                top_k=effective_top_k,
                                allowed_sources=allowed_sources,
                            )
                            aligned_tasks["Type"] = "task"
                            results.extend(aligned_tasks.to_dict("records"))
                            if timing:
                                print(
                                    f"[TIMING] doc={doc_id} stage=align_tasks took={time.perf_counter() - stage_started:.3f}s | count={len(aligned_tasks)}"
                                )

                        if return_edges and kt_results:
                            stage_started = time.perf_counter()
                            edges_df = self._derive_enables_edges(kt_results, doc_id)
                            if not edges_df.empty:
                                all_edges.append(edges_df)
                            if timing:
                                print(
                                    f"[TIMING] doc={doc_id} stage=derive_enables_edges took={time.perf_counter() - stage_started:.3f}s | count={len(edges_df)}"
                                )

                    if timing:
                        print(
                            f"[TIMING] doc={doc_id} stage=document_total took={time.perf_counter() - doc_started:.3f}s"
                        )

                except Exception as exc:
                    if warnings:
                        print(f"Warning: Failed to process row {idx}: {exc}")
                    continue

            normalize_started = time.perf_counter()
            df = self._normalize_mixed_concept_rows(pd.DataFrame(results))
            if timing:
                print(
                    f"[TIMING] stage=normalize_results took={time.perf_counter() - normalize_started:.3f}s | count={len(df)}"
                )
            if output_csv_path:
                csv_started = time.perf_counter()
                df.to_csv(output_csv_path, index=False, encoding="utf-8")
                if timing:
                    print(f"[TIMING] stage=write_output_csv took={time.perf_counter() - csv_started:.3f}s")

            if return_edges:
                edges = (
                    pd.concat(all_edges, ignore_index=True)
                    if all_edges
                    else pd.DataFrame(columns=["Research ID", "Skill", "Knowledge", "Task", "Edge Type", "confidence"])
                )
                if timing:
                    print(f"[TIMING] batch_total took={time.perf_counter() - batch_started:.3f}s docs={len(data)}")
                return {"nodes": df, "edges": edges}

            if timing:
                print(f"[TIMING] batch_total took={time.perf_counter() - batch_started:.3f}s docs={len(data)}")
            return df
        except Exception as exc:
            raise LAiSERError(f"Batch extraction failed: {exc}")

    def _normalize_mixed_concept_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        output_columns = [
            "Research ID",
            "Type",
            "Raw Concept",
            "Taxonomy Concept",
            "Taxonomy Description",
            "Taxonomy Source",
            "Source Url",
            "Correlation Coefficient",
        ]

        if df.empty:
            return pd.DataFrame(columns=output_columns)

        type_to_cols = {
            "skill": ("Raw Skill", "Taxonomy Skill"),
            "knowledge": ("Raw Knowledge", "Taxonomy Knowledge"),
            "task": ("Raw Task", "Taxonomy Task"),
        }

        normalized_rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item_type = str(row.get("Type", "") or "").strip().lower()
            raw_col, taxonomy_col = type_to_cols.get(item_type, ("Raw Concept", "Taxonomy Concept"))
            raw_value = row.get(raw_col, row.get("Raw Concept", ""))
            taxonomy_value = row.get(taxonomy_col, row.get("Taxonomy Concept", ""))

            normalized_rows.append(
                {
                    "Research ID": row.get("Research ID", ""),
                    "Type": item_type or str(row.get("Type", "") or ""),
                    "Raw Concept": raw_value,
                    "Taxonomy Concept": taxonomy_value,
                    "Taxonomy Description": row.get("Taxonomy Description", ""),
                    "Taxonomy Source": row.get("Taxonomy Source", ""),
                    "Source Url": row.get("Source Url", row.get("Taxonomy URL", "")),
                    "Correlation Coefficient": row.get("Correlation Coefficient", ""),
                }
            )

        normalized = pd.DataFrame(normalized_rows)
        return normalized.loc[:, output_columns]

    def _deduplicate(self, items: List[str], semantic_threshold: float = 0.92) -> List[str]:
        if not items:
            return items

        seen = set()
        exact_deduped = []
        for item in items:
            key = item.lower().strip()
            if key not in seen:
                seen.add(key)
                exact_deduped.append(item)

        if len(exact_deduped) <= 1:
            return exact_deduped

        model = self.data_access.get_embedding_model()
        embeddings = model.encode(exact_deduped, normalize_embeddings=True)
        kept_indices = []
        for i in range(len(exact_deduped)):
            is_duplicate = False
            for j in kept_indices:
                score = float(np.dot(embeddings[i], embeddings[j]))
                if score >= semantic_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_indices.append(i)

        return [exact_deduped[i] for i in kept_indices]

    def extract_raw_llm_skills(self, input_data, text_columns, input_type: str = "job_desc"):
        normalized_input_type = self.prompt_builder.normalize_input_type(input_type)
        if normalized_input_type == "syllabus":
            prompt_input: Any = {
                "description": str(input_data.get("description", "")).strip(),
                "learning_outcomes": str(input_data.get("learning_outcomes", "")).strip(),
            }
        else:
            prompt_input = " ".join(str(input_data.get(col, "")) for col in text_columns).strip()
        extraction_prompt = self.prompt_builder.build_skill_extraction_prompt(
            input_text=prompt_input, input_type=normalized_input_type
        )
        response = self.router.generate(extraction_prompt)
        skills = self.llm_parser._parse_skills_from_response(response)
        if not skills:
            preview = response.strip().replace("\n", " ")[:200]
            print(f"Warning: failed to parse skills from response: {preview}")
        return skills

    def extract_raw_llm_knowledge_tasks(
        self,
        input_data: Dict,
        text_columns: List[str],
        extracted_skills: List[str],
    ) -> List[Dict[str, Any]]:
        if not extracted_skills:
            return []

        text_blob = " ".join(str(input_data.get(col, "")) for col in text_columns).strip()
        prompt = self.prompt_builder.build_knowledge_task_prompt(text_blob, extracted_skills)
        response = self.router.generate(prompt)
        results = self.llm_parser.parse_knowledge_task_response(response)

        if not results:
            preview = response.strip().replace("\n", " ")[:200]
            logger.warning(f"Failed to parse knowledge/task response: {preview}")

        return results

    def align_extracted_knowledge(
        self,
        raw_knowledge: List[str],
        document_id: str = "0",
        description: str = "",
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
        allowed_sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if self.knowledge_faiss.index is None:
            logger.warning("Knowledge alignment index not available. Returning raw extracted knowledge.")
            return pd.DataFrame(
                {
                    "Research ID": document_id,
                    "Raw Knowledge": raw_knowledge,
                    "Taxonomy Knowledge": raw_knowledge,
                    "Taxonomy Description": [""] * len(raw_knowledge),
                    "Taxonomy Source": ["pending"] * len(raw_knowledge),
                    "Source Url": [""] * len(raw_knowledge),
                    "Correlation Coefficient": [0.0] * len(raw_knowledge),
                }
            )
        return self.knowledge_alignment.align(
            raw_items=raw_knowledge,
            document_id=document_id,
            description=description,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            raw_col="Raw Knowledge",
            taxonomy_col="Taxonomy Knowledge",
            allowed_sources=allowed_sources,
        )

    def align_extracted_tasks(
        self,
        raw_tasks: List[str],
        document_id: str = "0",
        description: str = "",
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
        allowed_sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if self.task_faiss.index is None:
            logger.warning("Task alignment index not available. Returning raw extracted tasks.")
            return pd.DataFrame(
                {
                    "Research ID": document_id,
                    "Raw Task": raw_tasks,
                    "Taxonomy Task": raw_tasks,
                    "Taxonomy Description": [""] * len(raw_tasks),
                    "Taxonomy Source": ["pending"] * len(raw_tasks),
                    "Source Url": [""] * len(raw_tasks),
                    "Correlation Coefficient": [0.0] * len(raw_tasks),
                }
            )
        return self.task_alignment.align(
            raw_items=raw_tasks,
            document_id=document_id,
            description=description,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            raw_col="Raw Task",
            taxonomy_col="Taxonomy Task",
            allowed_sources=allowed_sources,
        )

    def align_extracted_skills(
        self,
        raw_skills: List[str],
        document_id: str = "0",
        description: str = "",
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
        allowed_sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if raw_skills is None:
            print("Warning: No skills to align (raw_skills is None)")
            return pd.DataFrame(
                columns=[
                    "Research ID",
                    "Description",
                    "Raw Skill",
                    "Taxonomy Skill",
                    "Skill Tag",
                    "Correlation Coefficient",
                ]
            )

        if not isinstance(raw_skills, list):
            print(f"Warning: raw_skills is not a list, converting from {type(raw_skills)}")
            raw_skills = [str(raw_skills)] if raw_skills else []

        return self.alignment_service.align_skills_to_taxonomy(
            raw_skills=raw_skills,
            document_id=document_id,
            description=description,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            allowed_sources=allowed_sources,
        )

    def _derive_enables_edges(
        self,
        kt_results: List[Dict[str, Any]],
        document_id: str,
    ) -> pd.DataFrame:
        rows = []
        for item in kt_results:
            skill = item.get("skill", "")
            knowledge_items = item.get("knowledge", [])
            task_items = item.get("tasks", [])
            for k in knowledge_items:
                for t in task_items:
                    rows.append(
                        {
                            "Research ID": document_id,
                            "Skill": skill,
                            "Knowledge": k,
                            "Task": t,
                            "Edge Type": "ENABLES",
                            "confidence": "low",
                        }
                    )
        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=["Research ID", "Skill", "Knowledge", "Task", "Edge Type", "confidence"])
        )
