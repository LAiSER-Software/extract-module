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
    DEFAULT_SIMILARITY_THRESHOLDS,
    DEFAULT_TOP_K,
    KSA_DETAILS_PROMPT,
    KSA_EXTRACTION_PROMPT,
    KT_FROM_SKILLS_PROMPT,
    SCQF_LEVELS,
    SKILL_EXTRACTION_PROMPT_SYLLABUS,
)
from laiser.data_access import DataAccessLayer, FAISSIndexManager, KnowledgeFAISSIndexManager, TaskFAISSIndexManager
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
    def parse_knowledge_task_response(response: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON response from KT_FROM_SKILLS_PROMPT.

        Returns a list of dicts, each with keys:
            skill     str
            knowledge List[str]
            tasks     List[str]
        """
        if not response or not response.strip():
            return []

        def validate(results: Any) -> List[Dict[str, Any]]:
            validated = []
            if results is None:
                return validated
            if not isinstance(results, list):
                results = [results]

            for item in results:
                if not isinstance(item, dict):
                    continue

                skill = str(item.get("skill", item.get("Skill", ""))).strip()
                knowledge = item.get("knowledge", item.get("Knowledge", item.get("Knowledge Required", [])))
                tasks = item.get("tasks", item.get("task", item.get("Task Abilities", [])))

                if not isinstance(knowledge, list):
                    knowledge = [str(knowledge)] if knowledge else []
                if not isinstance(tasks, list):
                    tasks = [str(tasks)] if tasks else []

                if skill:
                    validated.append(
                        {
                            "skill": skill,
                            "knowledge": [str(k).strip() for k in knowledge if str(k).strip()],
                            "tasks": [str(t).strip() for t in tasks if str(t).strip()],
                        }
                    )
            return validated

        def parse_array_fragment(fragment: str) -> List[str]:
            fragment = fragment.strip()
            if not fragment:
                return []

            try:
                parsed = json.loads(fragment)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

            return [m.strip() for m in re.findall(r'"([^"]+)"', fragment) if m.strip()]

        try:
            candidates: List[str] = []
            stripped = response.strip()

            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
            if fenced:
                candidates.append(fenced.group(1).strip())

            obj_match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if obj_match:
                candidates.append(obj_match.group(0).strip())

            list_match = re.search(r"\[.*\]", stripped, re.DOTALL)
            if list_match:
                candidates.append(list_match.group(0).strip())

            candidates.append(stripped)

            seen = set()
            for candidate in candidates:
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)

                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue

                if isinstance(parsed, dict):
                    for key in ("results", "data", "items"):
                        validated = validate(parsed.get(key, []))
                        if validated:
                            return validated

                    validated = validate([parsed])
                    if validated:
                        return validated

                elif isinstance(parsed, list):
                    validated = validate(parsed)
                    if validated:
                        return validated

            block_pattern = re.compile(
                r'"skill"\s*:\s*"(?P<skill>[^"]+)"'
                r'.*?"knowledge"\s*:\s*(?P<knowledge>\[[^\]]*\])'
                r'.*?"tasks"\s*:\s*(?P<tasks>\[[^\]]*\])',
                re.DOTALL | re.IGNORECASE,
            )

            fallback_results: List[Dict[str, Any]] = []
            for match in block_pattern.finditer(stripped):
                skill = match.group("skill").strip()
                knowledge = parse_array_fragment(match.group("knowledge"))
                tasks = parse_array_fragment(match.group("tasks"))
                if skill:
                    fallback_results.append(
                        {
                            "skill": skill,
                            "knowledge": knowledge,
                            "tasks": tasks,
                        }
                    )

            if fallback_results:
                return fallback_results

            logger.warning(
                "Failed to parse knowledge/task response into usable blocks. Preview: %s",
                stripped.replace("\n", " ")[:300],
            )
            return []
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


class AlignmentService:
    """
    Generalized alignment service for Skills, Knowledge, and Tasks.

    Uses any FAISS index manager that exposes:
        .get_metadata() -> pd.DataFrame
        .search_similar(query_vec, top_k, allowed_sources) -> List[Dict]  (key "Name")
        .search_similar_skills(...)  (legacy key "Skill", for FAISSIndexManager)
    """

    def __init__(self, data_access: DataAccessLayer, faiss_manager):
        self.data_access = data_access
        self.faiss_manager = faiss_manager
        # FAISSIndexManager.initialize_index is a no-op if already loaded
        if hasattr(self.faiss_manager, "initialize_index"):
            self.faiss_manager.initialize_index(force_rebuild=False)

    def align(
        self,
        raw_items: List[str],
        document_id: str = "0",
        description: str = "",
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
        raw_col: str = "Raw Item",
        taxonomy_col: str = "Taxonomy Item",
        allowed_sources: Optional[List[str]] = None,
        debug: bool = False,
    ) -> pd.DataFrame:
        """
        Align a list of raw extracted strings to the loaded FAISS taxonomy.

        Parameters
        ----------
        raw_col : str
            Name for the "raw extracted" column in the output DataFrame.
        taxonomy_col : str
            Name for the "matched taxonomy" column in the output DataFrame.

        Returns
        -------
        pd.DataFrame with columns: Research ID, raw_col, taxonomy_col,
            Taxonomy Description, Taxonomy Source, Source Url, Correlation Coefficient
        """
        mapped_items: List[str] = []
        raw_matched: List[str] = []
        taxonomy_descriptions: List[str] = []
        taxonomy_sources: List[str] = []
        taxonomy_urls: List[str] = []
        correlations: List[float] = []

        def log_debug(msg: str):
            if debug:
                logger.debug(msg)

        log_debug(f"[align] raw_items={len(raw_items)} threshold={similarity_threshold} top_k={top_k}")

        model = self.data_access.get_embedding_model()
        try:
            metadata = self.faiss_manager.get_metadata()
        except Exception:
            # Index not yet available — return empty DataFrame
            return pd.DataFrame(
                {
                    "Research ID": pd.Series([], dtype=str),
                    raw_col: [],
                    taxonomy_col: [],
                    "Taxonomy Description": [],
                    "Taxonomy Source": [],
                    "Source Url": [],
                    "Correlation Coefficient": [],
                }
            )

        for i, item in enumerate(raw_items):
            log_debug(f"[item {i}] raw='{item}'")
            query_vec = model.encode([item], normalize_embeddings=True)

            # Support both legacy FAISSIndexManager (search_similar_skills, key "Skill")
            # and new managers (search_similar, key "Name")
            if hasattr(self.faiss_manager, "search_similar"):
                results = self.faiss_manager.search_similar(
                    np.array(query_vec).astype("float32"),
                    top_k=1,
                    allowed_sources=allowed_sources,
                )
                name_key = "Name"
            else:
                results = self.faiss_manager.search_similar_skills(
                    np.array(query_vec).astype("float32"),
                    top_k=1,
                    allowed_sources=allowed_sources,
                )
                name_key = "Skill"

            if not results:
                log_debug(f"[item {i}] no results -> skip")
                continue

            best = results[0]
            similarity = float(best.get("Similarity", 0.0))
            meta_idx = best.get("Index")
            canonical = str(best.get(name_key, "")).strip()

            log_debug(f"[item {i}] best='{canonical}' sim={similarity:.4f}")

            if similarity < similarity_threshold or not canonical:
                continue

            meta: Dict = {}
            if meta_idx is not None and isinstance(metadata, pd.DataFrame):
                idx_int = int(meta_idx)
                if 0 <= idx_int < len(metadata):
                    meta = metadata.iloc[idx_int].to_dict()

            taxonomy_description = str(meta.get("description", meta.get("Description", "")))
            taxonomy_source = str(meta.get("taxonomy", ""))
            taxonomy_url = str(meta.get("source_url", meta.get("Source URL", meta.get("sourceUrl", ""))))

            mapped_items.append(canonical)
            raw_matched.append(item)
            taxonomy_descriptions.append(taxonomy_description)
            taxonomy_sources.append(taxonomy_source)
            taxonomy_urls.append(taxonomy_url)
            correlations.append(similarity)

        log_debug(f"[align] matched={len(mapped_items)} of {len(raw_items)}")

        # Apply top_k trim
        if len(mapped_items) > top_k:
            combined = sorted(
                zip(correlations, raw_matched, mapped_items, taxonomy_descriptions, taxonomy_sources, taxonomy_urls),
                key=lambda x: x[0],
                reverse=True,
            )[:top_k]
            correlations, raw_matched, mapped_items, taxonomy_descriptions, taxonomy_sources, taxonomy_urls = map(
                list, zip(*combined)
            )

        return pd.DataFrame(
            {
                "Research ID": document_id,
                raw_col: raw_matched,
                taxonomy_col: mapped_items,
                "Taxonomy Description": taxonomy_descriptions,
                "Taxonomy Source": taxonomy_sources,
                "Source Url": taxonomy_urls,
                "Correlation Coefficient": correlations,
            }
        )

    # ---------- Legacy method kept for backward compatibility ----------
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
        return self.align(
            raw_items=raw_skills,
            document_id=document_id,
            description=description,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            raw_col="Raw Skill",
            taxonomy_col="Taxonomy Skill",
            allowed_sources=allowed_sources,
            debug=debug,
        )


# Backward-compatible alias
SkillAlignmentService = AlignmentService


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

        # Skills index (v0.4 / combined taxonomy)
        self.faiss_manager = FAISSIndexManager(self.data_access)
        self.alignment_service = AlignmentService(self.data_access, self.faiss_manager)

        # Knowledge + Task indexes (v0.5) — initialized lazily; no-op if CSV not yet built
        self.knowledge_faiss = KnowledgeFAISSIndexManager(self.data_access)
        self.knowledge_faiss.initialize_index(force_rebuild=False)
        self.knowledge_alignment = AlignmentService(self.data_access, self.knowledge_faiss)

        self.task_faiss = TaskFAISSIndexManager(self.data_access)
        self.task_faiss.initialize_index(force_rebuild=False)
        self.task_alignment = AlignmentService(self.data_access, self.task_faiss)

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

        # Initialize skills FAISS index
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
        return_edges: bool = False,
        similarity_thresholds: Optional[Dict[str, float]] = None,
        timing: bool = False,
    ):
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
            Maximum number of aligned items to return per document (default: 25)
        similarity_threshold : float, optional
            Global minimum similarity score applied to all types unless overridden
            by similarity_thresholds. Defaults to 0.20 for backward compatibility.
        similarity_thresholds : dict, optional
            Per-type similarity thresholds. Keys: "skill", "knowledge", "task".
            Overrides similarity_threshold for each specified type.
            Defaults: {"skill": 0.20, "knowledge": 0.45, "task": 0.55}
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
        return_edges : bool, optional
            If True, return a dict {"nodes": pd.DataFrame, "edges": pd.DataFrame}
            where "edges" contains ENABLES edges (Knowledge → Task co-occurrence per skill).
            If False (default), return a plain DataFrame — no breaking change.
        timing : bool, optional
            Accepted for compatibility with benchmark callers.

        Returns
        -------
        pd.DataFrame  (when return_edges=False, default)
            DataFrame with normalized mixed-concept rows:
            Research ID, Type, Raw Concept, Taxonomy Concept,
            Taxonomy Description, Taxonomy Source, Correlation Coefficient.
        dict  (when return_edges=True)
            {"nodes": pd.DataFrame, "edges": pd.DataFrame}
            edges columns: Research ID, Skill, Knowledge, Task, Edge Type, confidence
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

        # Apply defaults for top_k and similarity thresholds
        effective_top_k = top_k if top_k is not None else DEFAULT_TOP_K

        # Build per-type thresholds: start from DEFAULT_SIMILARITY_THRESHOLDS,
        # override with the scalar similarity_threshold if provided (backward compat),
        # then overlay any explicitly provided similarity_thresholds dict.
        resolved: Dict[str, float] = dict(DEFAULT_SIMILARITY_THRESHOLDS)
        if similarity_threshold is not None:
            resolved = {k: similarity_threshold for k in resolved}
        if similarity_thresholds:
            resolved.update({k: float(v) for k, v in similarity_thresholds.items()})

        try:
            results = []
            all_edges: List[pd.DataFrame] = []

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
                            similarity_threshold=resolved["skill"],
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
                                similarity_threshold=resolved["knowledge"],
                                top_k=effective_top_k,
                                allowed_sources=allowed_sources,
                            )
                            aligned_knowledge["Type"] = "knowledge"
                            results.extend(aligned_knowledge.to_dict("records"))

                        if "tasks" in extract and raw_tasks:
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

                        # Derive ENABLES edges (K × T per skill, before alignment)
                        if return_edges and kt_results:
                            edges_df = self._derive_enables_edges(kt_results, doc_id)
                            if not edges_df.empty:
                                all_edges.append(edges_df)

                except Exception as e:
                    if warnings:
                        print(f"Warning: Failed to process row {idx}: {e}")
                    continue

            df = self._normalize_mixed_concept_rows(pd.DataFrame(results))
            if not df.empty and len(df) > effective_top_k:
                if "Correlation Coefficient" in df.columns:
                    scored_df = df.copy()
                    scored_df["Correlation Coefficient"] = pd.to_numeric(
                        scored_df["Correlation Coefficient"], errors="coerce"
                    )
                    df = scored_df.sort_values(
                        by="Correlation Coefficient",
                        ascending=False,
                        na_position="last",
                    ).head(effective_top_k)
                else:
                    df = df.head(effective_top_k)
            df.to_csv("skills_alignment_results.csv", index=False, encoding="utf-8")

            if return_edges:
                edges = (
                    pd.concat(all_edges, ignore_index=True)
                    if all_edges
                    else pd.DataFrame(columns=["Research ID", "Skill", "Knowledge", "Task", "Edge Type", "confidence"])
                )
                return {"nodes": df, "edges": edges}

            return df

        except Exception as e:
            raise LAiSERError(f"Batch extraction failed: {e}")

    def _normalize_mixed_concept_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse per-type sparse columns into a unified mixed-concept schema.

        Output columns:
            Research ID, Type, Raw Concept, Taxonomy Concept,
            Taxonomy Description, Taxonomy Source, Source Url, Correlation Coefficient
        """
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
        allowed_sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Align extracted knowledge items to the Knowledge taxonomy FAISS index.

        Falls back to returning raw strings if the index has not been built yet
        (run scripts/build_knowledge_index.py to build it).
        """
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
        """
        Align extracted tasks to the Task taxonomy FAISS index.

        Falls back to returning raw strings if the index has not been built yet
        (run scripts/build_task_index.py to build it).
        """
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
        """
        Derive ENABLES edges from co-occurrence of Knowledge and Tasks within the same skill.

        For each skill block in kt_results, every (Knowledge, Task) pair is an ENABLES edge:
        Knowledge ──ENABLES──► Task

        This is derived directly from LLM output before alignment, so it works even
        when Knowledge/Task FAISS indexes are not yet available.

        Returns
        -------
        pd.DataFrame with columns:
            Research ID, Skill, Knowledge, Task, Edge Type, confidence
        """
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
