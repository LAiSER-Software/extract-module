"""
Service layer for skill extraction and processing

This module contains the core business logic for skill extraction.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from laiser.config import (
    COMBINED_EXTRACTION_PROMPT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TOP_K,
    KSA_DETAILS_PROMPT,
    KSA_EXTRACTION_PROMPT,
    KT_FROM_SKILLS_PROMPT,
    SCQF_LEVELS,
    SKILL_EXTRACTION_PROMPT_SYLLABUS,
)
from laiser.data_access import DataAccessLayer, FAISSIndexManager
from laiser.exceptions import InvalidInputError, LAiSERError
from laiser.llm_models.llm_router import LLMRouter

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts for different types of skill extraction tasks"""

    @staticmethod
    def build_skill_extraction_prompt(input_text: str, input_type: str) -> str:
        """Build prompt for basic skill extraction"""
        if input_type == "job_desc":
            extraction_prompt = COMBINED_EXTRACTION_PROMPT.format(description=input_text)
            return extraction_prompt
        elif input_type == "syllabus":
            return SKILL_EXTRACTION_PROMPT_SYLLABUS.format(
                description=input_text.get("description", ""),
                learning_outcomes=input_text.get("learning_outcomes", ""),
            )
        else:
            raise InvalidInputError(f"Unsupported input type: {input_type}")

    @staticmethod
    def build_ksa_extraction_prompt(
        query: Dict[str, str],
        input_type: str,
        num_key_skills: int,
        num_key_kr: str,
        num_key_tas: str,
        esco_skills: List[str] = None,
    ) -> str:
        """Build prompt for KSA (Knowledge, Skills, Abilities) extraction"""

        input_desc = (
            "job description" if input_type == "job_desc" else "course syllabus description and its learning outcomes"
        )

        if input_type == "syllabus":
            input_text = f"### Input:\\n**Course Description:** {query.get('description', '')}\\n**Learning Outcomes:** {query.get('learning_outcomes', '')}"
        else:
            input_text = f"### Input:\\n{query.get('description', '')}"

        # Format SCQF levels
        scqf_levels_text = "\\n".join([f"  - {level}: {desc}" for level, desc in SCQF_LEVELS.items()])

        # Prepare ESCO context
        esco_context_block = ", ".join(esco_skills) if esco_skills else "No relevant skills found in taxonomy"

        return KSA_EXTRACTION_PROMPT.format(
            input_desc=input_desc,
            num_key_skills=num_key_skills,
            num_key_kr=num_key_kr,
            num_key_tas=num_key_tas,
            input_text=input_text,
            esco_context_block=esco_context_block,
            scqf_levels=scqf_levels_text,
        )

    @staticmethod
    def build_ksa_details_prompt(skill: str, description: str, num_key_kr: int = 3, num_key_tas: int = 3) -> str:
        """Build prompt for getting detailed KSA information for a specific skill"""
        return KSA_DETAILS_PROMPT.format(
            skill=skill,
            description=description,
            num_key_kr=num_key_kr,
            num_key_tas=num_key_tas,
        )

    @staticmethod
    def build_knowledge_task_prompt(description: str, extracted_skills: List[str]) -> str:
        """
        Build prompt for extracting Knowledge and Tasks (Call 2 of v0.5 pipeline).

        Uses the full job description as context alongside already-extracted skills
        so that Knowledge and Task outputs are specific to this role, not generic.
        """
        skills_formatted = "\n".join(f"- {s}" for s in extracted_skills)
        return KT_FROM_SKILLS_PROMPT.format(
            description=description,
            skills=skills_formatted,
        )

    def strong_preprocessing_prompt(self, raw_description):
        raise NotImplementedError("strong_preprocessing_prompt is not yet implemented. Fix llm router params.")


class ResponseParser:
    """Parses responses from LLM models"""

    @staticmethod
    def _parse_skills_from_response(response: str) -> List[str]:
        if not response or not response.strip():
            return []

        fragments: List[str] = []
        stripped = response.strip()

        code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if code_match:
            fragments.append(code_match.group(1).strip())

        brace_match = re.search(r"\{.*?\}", stripped, re.DOTALL)
        if brace_match:
            fragments.append(brace_match.group(0).strip())

        list_match = re.search(r"\[.*?\]", stripped, re.DOTALL)
        if list_match:
            fragments.append(list_match.group(0).strip())

        fragments.append(stripped)

        seen = set()
        for fragment in fragments:
            if not fragment or fragment in seen:
                continue
            seen.add(fragment)
            for candidate in (fragment,):
                try:
                    loaded = json.loads(candidate)
                except json.JSONDecodeError:
                    continue

                if isinstance(loaded, dict):
                    skills = loaded.get("skills")
                    if isinstance(skills, list):
                        return [str(s).strip() for s in skills if str(s).strip()]
                elif isinstance(loaded, list):
                    return [str(s).strip() for s in loaded if str(s).strip()]

        quoted_skills = re.findall(r"\"([^\"]{1,100})\"", stripped)
        if quoted_skills:
            cleaned = []
            for skill in quoted_skills:
                skill = skill.strip()
                if not skill:
                    continue
                if not (1 <= len(skill.split()) <= 5):
                    continue
                if skill.lower().startswith("skills"):
                    continue
                cleaned.append(skill)
            if cleaned:
                return cleaned

        return []

    @staticmethod
    def parse_skill_extraction_response(response: str) -> List[str]:
        """Parse basic skill extraction response"""
        try:
            if not response:
                return []

            # Find the content between model tags (original format)
            pattern = r"<start_of_turn>model\\s*<eos>(.*?)<eos>\\s*$"
            match = re.search(pattern, response, re.DOTALL)

            if match:
                content = match.group(1).strip()
                lines = [line.strip() for line in content.split("\\n") if line.strip()]
                skills = [line[1:].strip() for line in lines if line.startswith("-")]
                return skills if skills is not None else []

            # Fallback: parse the response directly (current Gemini format)
            lines = [line.strip() for line in response.split("\n") if line.strip()]

            # Remove any unwanted prefixes and tags
            clean_lines = []
            for line in lines:
                if line.startswith("<start_of_turn>") or line.startswith("<end_of_turn>"):
                    continue
                if "--" in line:  # Skip separator lines
                    continue
                clean_lines.append(line)

            return clean_lines

        except Exception as e:
            print(f"Warning: Failed to parse skill extraction response: {e}")
            return []

    @staticmethod
    def parse_ksa_extraction_response(response: str) -> List[Dict[str, Any]]:
        """Parse KSA extraction response"""
        try:
            if not response:
                return []

            out = []
            # Split into items, handling optional '->' prefix and multi-line input
            items = [item.strip() for item in response.split("->") if item.strip()]

            for i, item in enumerate(items):
                skill_data = {}
                try:
                    # Extract skill
                    skill_match = re.search(r"Skill:\s*([^,\n]+)", item)
                    if skill_match:
                        skill_data["Skill"] = skill_match.group(1).strip()

                    # Extract level
                    level_match = re.search(r"Level:\s*(\d+)", item)
                    if level_match:
                        skill_data["Level"] = int(level_match.group(1).strip())

                    # Extract knowledge required (multi-line support)
                    knowledge_match = re.search(
                        r"Knowledge Required:\s*(.*?)(?=\s*Task Abilities:|\s*$)",
                        item,
                        re.DOTALL,
                    )
                    if knowledge_match:
                        knowledge_raw = knowledge_match.group(1).strip()
                        skill_data["Knowledge Required"] = [k.strip() for k in knowledge_raw.split(",") if k.strip()]

                    # Extract task abilities (multi-line support)
                    task_match = re.search(r"Task Abilities:\s*(.*?)(?=\s*$)", item, re.DOTALL)
                    if task_match:
                        task_raw = task_match.group(1).strip()
                        skill_data["Task Abilities"] = [t.strip() for t in task_raw.split(",") if t.strip()]

                    if skill_data:  # Only add if we found some data
                        out.append(skill_data)

                except Exception as e:
                    print(f"Warning: Error processing KSA item {i}: {e}")
                    continue

            return out

        except Exception as e:
            print(f"Warning: Failed to parse KSA extraction response: {e}")
            return []

    @staticmethod
    @staticmethod
    def parse_knowledge_task_response(response: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON response from KT_FROM_SKILLS_PROMPT.

        Returns a list of dicts, each with keys:
            skill     str
            knowledge List[str]
            tasks     List[str]
        """
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?", "", response).strip().rstrip("`").strip()
            parsed = json.loads(cleaned)
            results = parsed.get("results", [])

            validated = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                skill = str(item.get("skill", "")).strip()
                knowledge = item.get("knowledge", [])
                tasks = item.get("tasks", [])

                if not isinstance(knowledge, list):
                    knowledge = [str(knowledge)] if knowledge else []
                if not isinstance(tasks, list):
                    tasks = [str(tasks)] if tasks else []

                if skill:
                    validated.append(
                        {
                            "skill": skill,
                            "knowledge": [k.strip() for k in knowledge if k.strip()],
                            "tasks": [t.strip() for t in tasks if t.strip()],
                        }
                    )
            return validated

        except Exception as e:
            logger.warning(f"Failed to parse knowledge/task response: {e}")
            return []

    def parse_ksa_details_response(response: str) -> Tuple[List[str], List[str]]:
        """Parse KSA details response"""
        try:
            if not response:
                return [], []

            json_match = re.search(r"\\{.*\\}", response, re.DOTALL)
            if not json_match:
                return [], []

            parsed = json.loads(json_match.group())
            knowledge = parsed.get("Knowledge Required", [])
            task_abilities = parsed.get("Task Abilities", [])

            # Ensure they are lists
            if not isinstance(knowledge, list):
                knowledge = [str(knowledge)] if knowledge else []
            if not isinstance(task_abilities, list):
                task_abilities = [str(task_abilities)] if task_abilities else []

            return knowledge, task_abilities
        except Exception as e:
            print(f"Warning: Failed to parse KSA details response: {e}")
            return [], []


class SkillAlignmentService:
    """Service for aligning extracted skills with taxonomies"""

    def __init__(self, data_access: DataAccessLayer, faiss_manager: FAISSIndexManager):
        self.data_access = data_access
        self.faiss_manager = faiss_manager
        self.faiss_manager.initialize_index(force_rebuild=False)

    def align_skills_to_taxonomy(
        self,
        raw_skills: List[str],
        document_id: str = "0",
        description: str = "",
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
        debug: bool = False,
        allowed_sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        mapped_skills = []
        raw_skills_matched = []
        taxonomy_descriptions = []
        taxonomy_sources = []
        correlations = []

        def log_debug(msg: str):
            if debug:
                logger.debug(msg)

        log_debug(f"[align] raw_skills={len(raw_skills)} threshold={similarity_threshold} top_k={top_k}")

        model = self.data_access.get_embedding_model()

        # metadata loaded once
        metadata = self.faiss_manager.get_metadata()
        log_debug(f"[align] metadata type={type(metadata).__name__} len={len(metadata)}")
        if isinstance(metadata, pd.DataFrame) and not metadata.empty:
            log_debug(f"[align] metadata columns={list(metadata.columns)}")

        for i, skill in enumerate(raw_skills):
            log_debug(f"[skill {i}] raw='{skill}'")

            query_vec = model.encode([skill], normalize_embeddings=True)

            results = self.faiss_manager.search_similar_skills(
                np.array(query_vec).astype("float32"),
                top_k=1,
                allowed_sources=allowed_sources,
            )
            log_debug(f"[skill {i}] results={results}")

            if not results:
                log_debug(f"[skill {i}] no results -> skip")
                continue

            best = results[0]
            similarity = float(best.get("Similarity", 0.0))
            meta_idx = best.get("Index")
            canonical_skill = str(best.get("Skill", "")).strip()

            log_debug(f"[skill {i}] best='{canonical_skill}' sim={similarity:.4f} meta_idx={meta_idx}")

            if similarity < similarity_threshold:
                log_debug(f"[skill {i}] below threshold -> skip")
                continue
            meta = {}
            if not canonical_skill:
                log_debug(f"[skill {i}] empty canonical_skill -> skip")
                continue
            if meta_idx is None:
                log_debug(f"[skill {i}] meta_idx is None (search_similar_skills may not return Index)")
            elif int(meta_idx) >= len(metadata) or int(meta_idx) < 0:
                log_debug(f"[skill {i}] meta_idx out of range: {meta_idx} (metadata len={len(metadata)})")
            else:
                # ✅ DataFrame row by position
                meta = metadata.iloc[int(meta_idx)].to_dict()
                log_debug(f"[skill {i}] meta keys={list(meta.keys())}")

            # handle possible key casing differences
            taxonomy_description = meta.get("description", meta.get("Description", ""))
            taxonomy_source = meta.get("taxonomy", meta.get("taxonomy", ""))

            log_debug(f"[skill {i}] source='{taxonomy_source}' desc_len={len(taxonomy_description)}")

            mapped_skills.append(canonical_skill)
            raw_skills_matched.append(skill)
            taxonomy_descriptions.append(taxonomy_description)
            taxonomy_sources.append(taxonomy_source)
            correlations.append(similarity)

        log_debug(f"[align] matched={len(mapped_skills)} of {len(raw_skills)}")

        # Apply top_k limit: sort by correlation (descending) and take top_k
        if len(mapped_skills) > top_k:
            log_debug(f"[align] trimming to top_k={top_k}")

            combined = list(
                zip(
                    correlations,
                    raw_skills_matched,
                    mapped_skills,
                    taxonomy_descriptions,
                    taxonomy_sources,
                )
            )
            combined.sort(key=lambda x: x[0], reverse=True)
            combined = combined[:top_k]

            (
                correlations,
                raw_skills_matched,
                mapped_skills,
                taxonomy_descriptions,
                taxonomy_sources,
            ) = map(
                list, zip(*combined)
            )  # ✅ FIX #2: keep lists aligned

        result_df = pd.DataFrame(
            {
                "Research ID": document_id,
                "Raw Skill": raw_skills_matched,
                "Taxonomy Skill": mapped_skills,
                "Taxonomy Description": taxonomy_descriptions,
                "Taxonomy Source": taxonomy_sources,
                "Correlation Coefficient": correlations,
            }
        )

        log_debug(f"[align] result_df shape={result_df.shape}")
        return result_df


class SkillExtractionService:
    """Main service for skill extraction operations"""

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
        self.alignment_service = SkillAlignmentService(self.data_access, self.faiss_manager)
        self.prompt_builder = PromptBuilder()
        self.llm_parser = ResponseParser()
        self.response_parser = ResponseParser()
        self.router = LLMRouter(
            self.model_id,
            self.use_gpu,
            self.hf_token,
            self.api_key,
            backend=self.backend,
        )

        # Initialize FAISS index
        self.faiss_manager.initialize_index(force_rebuild=False)
        # Log router initialization state for debugging
        try:
            print(
                f"SkillExtractionService: router.llama_llm present: {getattr(self.router, 'llama_llm', None) is not None}"
            )
        except Exception:
            pass

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
    ) -> pd.DataFrame:
        """
        Extract and align skills from a dataset (main interface method).

        This method maintains backward compatibility with the original API.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        id_column : str
            Column name for document IDs
        text_columns : List[str]
            Column names containing text data
        input_type : str
            Type of input data
        top_k : int, optional
            Maximum number of aligned skills to return per document (default: 25)
        similarity_threshold : float, optional
            Minimum similarity score for a match to be included (default: 0.20).
            Higher values = stricter matching, fewer results.
            Lower values = more lenient matching, more results.
        levels : bool
            Whether to extract skill levels
        batch_size : int
            Batch size for processing
        warnings : bool
            Whether to show warnings

        extract : list, optional
            Types to extract. Options: "skills", "knowledge", "tasks".
            Defaults to ["skills"] for backward compatibility.
            Pass ["skills", "knowledge", "tasks"] or ["all"] for full v0.5 extraction.

        Returns
        -------
        pd.DataFrame
            DataFrame with extracted and aligned items.
            Includes a "Type" column when extracting more than skills.
        """
        if text_columns is None:
            text_columns = ["description"]

        if extract is None:
            extract = ["skills"]
        if extract == ["all"] or extract == "all":
            extract = ["skills", "knowledge", "tasks"]
        extract = [e.lower().strip() for e in extract]

        # --- input validation: ensure `data` is a DataFrame and not None ---
        if data is None:
            raise InvalidInputError(
                "extract_and_align_core: `data` is None. Please pass a pandas.DataFrame with rows to process."
            )
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Apply defaults for top_k and similarity_threshold
        effective_top_k = top_k if top_k is not None else DEFAULT_TOP_K
        effective_threshold = similarity_threshold if similarity_threshold is not None else 0.20

        try:
            results = []

            for idx, row in data.iterrows():
                try:
                    input_data = {col: row.get(col, "") for col in text_columns}
                    input_data["id"] = row.get(id_column, str(idx))
                    full_description = " ".join([str(input_data.get(col, "")) for col in text_columns])
                    doc_id = str(input_data["id"])

                    # --- Call 1: Skills extraction (always runs — required for Call 2) ---
                    skills = self.extract_raw_llm_skills(input_data, text_columns)

                    if "skills" in extract:
                        aligned_skills = self.align_extracted_skills(
                            skills,
                            doc_id,
                            full_description,
                            similarity_threshold=effective_threshold,
                            top_k=effective_top_k,
                            allowed_sources=allowed_sources,
                        )
                        aligned_skills["Type"] = "skill"
                        results.extend(aligned_skills.to_dict("records"))

                    # --- Call 2: Knowledge + Tasks extraction (uses full description + skills) ---
                    if "knowledge" in extract or "tasks" in extract:
                        kt_results = self.extract_raw_llm_knowledge_tasks(input_data, text_columns, skills)

                        raw_knowledge = self._deduplicate([k for item in kt_results for k in item.get("knowledge", [])])
                        raw_tasks = self._deduplicate([t for item in kt_results for t in item.get("tasks", [])])

                        if "knowledge" in extract and raw_knowledge:
                            aligned_knowledge = self.align_extracted_knowledge(
                                raw_knowledge,
                                doc_id,
                                full_description,
                                similarity_threshold=effective_threshold,
                                top_k=effective_top_k,
                            )
                            aligned_knowledge["Type"] = "knowledge"
                            results.extend(aligned_knowledge.to_dict("records"))

                        if "tasks" in extract and raw_tasks:
                            aligned_tasks = self.align_extracted_tasks(
                                raw_tasks,
                                doc_id,
                                full_description,
                                similarity_threshold=effective_threshold,
                                top_k=effective_top_k,
                            )
                            aligned_tasks["Type"] = "task"
                            results.extend(aligned_tasks.to_dict("records"))

                except Exception as e:
                    if warnings:
                        print(f"Warning: Failed to process row {idx}: {e}")
                    continue

            df = pd.DataFrame(results)
            df.to_csv("skills_alignment_results.csv", index=False, encoding="utf-8")
            return df

        except Exception as e:
            raise LAiSERError(f"Batch extraction failed: {e}")

    def _deduplicate(self, items: List[str], semantic_threshold: float = 0.92) -> List[str]:
        """
        Two-pass deduplication before alignment.

        Pass 1 — exact: lowercased string match, preserves first occurrence order.
        Pass 2 — semantic: embedding cosine similarity, drops near-duplicates
                           above threshold using the already-loaded embedding model.
        """
        if not items:
            return items

        # Pass 1 — exact
        seen = set()
        exact_deduped = []
        for item in items:
            key = item.lower().strip()
            if key not in seen:
                seen.add(key)
                exact_deduped.append(item)

        if len(exact_deduped) <= 1:
            return exact_deduped

        # Pass 2 — semantic
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

    def extract_raw_llm_skills(self, input_data, text_columns):

        text_blob = " ".join(str(input_data.get(col, "")) for col in text_columns).strip()
        extraction_prompt = self.prompt_builder.build_skill_extraction_prompt(
            input_text=text_blob, input_type="job_desc"
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
        """
        Call 2 of the v0.5 pipeline.

        Uses the full job description + already-extracted skills to derive
        Knowledge and Tasks for each skill, keeping job-specific context intact.

        Parameters
        ----------
        input_data : dict
            Row data containing text columns
        text_columns : list
            Column names to use as job description text
        extracted_skills : list
            Skills extracted by Call 1 (extract_raw_llm_skills)

        Returns
        -------
        list of dicts with keys: skill, knowledge, tasks
        """
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
    ) -> pd.DataFrame:
        """
        Align extracted knowledge items to the Knowledge taxonomy FAISS index.

        Note: Knowledge FAISS index and taxonomy data pipeline are pending v0.5
        data work. Returns empty DataFrame until index is ready.
        """
        # TODO(v0.5): replace with KnowledgeFAISSIndexManager once
        # knowledge taxonomy data is downloaded, embedded and indexed.
        logger.warning("Knowledge alignment index not yet available. Returning raw extracted knowledge.")
        return pd.DataFrame(
            {
                "Research ID": document_id,
                "Raw Knowledge": raw_knowledge,
                "Taxonomy Knowledge": raw_knowledge,
                "Taxonomy Description": [""] * len(raw_knowledge),
                "Taxonomy Source": ["pending"] * len(raw_knowledge),
                "Correlation Coefficient": [0.0] * len(raw_knowledge),
            }
        )

    def align_extracted_tasks(
        self,
        raw_tasks: List[str],
        document_id: str = "0",
        description: str = "",
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
    ) -> pd.DataFrame:
        """
        Align extracted tasks to the Task Abilities taxonomy FAISS index.

        Note: Task FAISS index and taxonomy data pipeline are pending v0.5
        data work. Returns empty DataFrame until index is ready.
        """
        # TODO(v0.5): replace with TaskFAISSIndexManager once
        # task taxonomy data is downloaded, embedded and indexed.
        logger.warning("Task alignment index not yet available. Returning raw extracted tasks.")
        return pd.DataFrame(
            {
                "Research ID": document_id,
                "Raw Task": raw_tasks,
                "Taxonomy Task": raw_tasks,
                "Taxonomy Description": [""] * len(raw_tasks),
                "Taxonomy Source": ["pending"] * len(raw_tasks),
                "Correlation Coefficient": [0.0] * len(raw_tasks),
            }
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
        """
        Align extracted skills to taxonomy.

        Parameters
        ----------
        raw_skills : List[str]
            List of raw extracted skills
        document_id : str
            Document identifier
        description : str
            Full description text for context
        similarity_threshold : float
            Minimum similarity score for a match to be included (default: 0.20)
        top_k : int
            Maximum number of aligned skills to return (default: 25)

        Returns
        -------
        pd.DataFrame
            DataFrame with aligned skills
        """
        # Add validation before calling alignment service
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
            raw_skills,
            document_id,
            description,
            similarity_threshold,
            top_k,
            allowed_sources,
        )
