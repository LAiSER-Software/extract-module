"""
Service layer for skill extraction and processing

This module contains the core business logic for skill extraction.
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd

from laiser.config import (
    DEFAULT_TOP_K,
    SCQF_LEVELS,
    SKILL_EXTRACTION_PROMPT_JOB,
    SKILL_EXTRACTION_PROMPT_SYLLABUS,
    KSA_EXTRACTION_PROMPT,
    KSA_DETAILS_PROMPT
)
from laiser.llm_models.llm_router import LLMRouter
from laiser.config import DEFAULT_BATCH_SIZE, DEFAULT_TOP_K
from laiser.exceptions import SkillExtractionError, InvalidInputError, LAiSERError
from laiser.data_access import DataAccessLayer, FAISSIndexManager

import logging
logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builds prompts for different types of skill extraction tasks"""
    
    @staticmethod
    def build_skill_extraction_prompt(input_text: str, input_type: str) -> str:
        """Build prompt for basic skill extraction"""
        if input_type == "job_desc":
            extraction_prompt = SKILL_EXTRACTION_PROMPT_JOB.format(description=input_text)
            return extraction_prompt
        elif input_type == "syllabus":
            return SKILL_EXTRACTION_PROMPT_SYLLABUS.format(
                description=input_text.get("description", ""),
                learning_outcomes=input_text.get("learning_outcomes", "")
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
        esco_skills: List[str] = None
    ) -> str:
        """Build prompt for KSA (Knowledge, Skills, Abilities) extraction"""
        
        input_desc = "job description" if input_type == "job_desc" else "course syllabus description and its learning outcomes"
        
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
            scqf_levels=scqf_levels_text
        )
    
    @staticmethod
    def build_ksa_details_prompt(skill: str, description: str, num_key_kr: int = 3, num_key_tas: int = 3) -> str:
        """Build prompt for getting detailed KSA information for a specific skill"""
        return KSA_DETAILS_PROMPT.format(
            skill=skill,
            description=description,
            num_key_kr=num_key_kr,
            num_key_tas=num_key_tas
        )
    
    def strong_preprocessing_prompt(self,raw_description):
        prompt = f"""
    You are a data preprocessing assistant trained to clean job descriptions for skill extraction.

    Your task is to remove the following from the text:
    - Company names, slogans, branding language
    - Locations, phone numbers, email addresses, URLs
    - Salary information, job ID, dates, scheduling info (e.g. 9am-5pm, weekends required)
    - HR/legal boilerplate (EEO, diversity statements, veteran status, disability policies)
    - Culture fluff like "fun environment", "fast-paced", "initiative", "self-motivated", "join us", "own your tomorrow", "apply now"
    - Internal team names or product names (e.g. ACE, THD, IMT)
    - Benefits sections (e.g. health & wellness, sabbatical, 401k, maternity, vacation)

    Your output must *only retain the task-related job duties, technical responsibilities, required skills, qualifications, and tools* without rephrasing.

    Input:
    \"\"\"
    {raw_description}
    \"\"\"

    Return only the cleaned job description.
    ### CLEANED JOB DESCRIPTION:
    """

    ## ISSUE: Fix llm router params
        response = llm_router(prompt, self.model_id, self.use_gpu, self.llm, 
                                self.tokenizer, self.model, self.api_key)
        cleaned = response.split("### CLEANED JOB DESCRIPTION:")[-1].strip()
        return cleaned
      

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
            pattern = r'<start_of_turn>model\\s*<eos>(.*?)<eos>\\s*$'
            match = re.search(pattern, response, re.DOTALL)

            if match:
                content = match.group(1).strip()
                lines = [line.strip() for line in content.split('\\n') if line.strip()]
                skills = [line[1:].strip() for line in lines if line.startswith('-')]
                return skills if skills is not None else []
            
            # Fallback: parse the response directly (current Gemini format)
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Remove any unwanted prefixes and tags
            clean_lines = []
            for line in lines:
                if line.startswith('<start_of_turn>') or line.startswith('<end_of_turn>'):
                    continue
                if '--' in line:  # Skip separator lines
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
            items = [item.strip() for item in response.split('->') if item.strip()]

            for i, item in enumerate(items):
                skill_data = {}
                try:
                    # Extract skill
                    skill_match = re.search(r"Skill:\s*([^,\n]+)", item)
                    if skill_match:
                        skill_data['Skill'] = skill_match.group(1).strip()

                    # Extract level
                    level_match = re.search(r"Level:\s*(\d+)", item)
                    if level_match:
                        skill_data['Level'] = int(level_match.group(1).strip())

                    # Extract knowledge required (multi-line support)
                    knowledge_match = re.search(r"Knowledge Required:\s*(.*?)(?=\s*Task Abilities:|\s*$)", item, re.DOTALL)
                    if knowledge_match:
                        knowledge_raw = knowledge_match.group(1).strip()
                        skill_data['Knowledge Required'] = [k.strip() for k in knowledge_raw.split(',') if k.strip()]

                    # Extract task abilities (multi-line support)
                    task_match = re.search(r"Task Abilities:\s*(.*?)(?=\s*$)", item, re.DOTALL)
                    if task_match:
                        task_raw = task_match.group(1).strip()
                        skill_data['Task Abilities'] = [t.strip() for t in task_raw.split(',') if t.strip()]

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
        document_id: str = '0',
        description: str = '',
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
        debug: bool = False,
        allowed_sources: Optional[List[str]] = None
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
                np.array(query_vec).astype("float32"), top_k=1,allowed_sources = allowed_sources
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

            log_debug(
                f"[skill {i}] source='{taxonomy_source}' desc_len={len(taxonomy_description)}"
            )

            mapped_skills.append(canonical_skill)
            raw_skills_matched.append(skill)
            taxonomy_descriptions.append(taxonomy_description)
            taxonomy_sources.append(taxonomy_source)
            correlations.append(similarity)

        log_debug(f"[align] matched={len(mapped_skills)} of {len(raw_skills)}")

        # Apply top_k limit: sort by correlation (descending) and take top_k
        if len(mapped_skills) > top_k:
            log_debug(f"[align] trimming to top_k={top_k}")

            combined = list(zip(
                correlations,
                raw_skills_matched,
                mapped_skills,
                taxonomy_descriptions,
                taxonomy_sources
            ))
            combined.sort(key=lambda x: x[0], reverse=True)
            combined = combined[:top_k]

            (correlations,
            raw_skills_matched,
            mapped_skills,
            taxonomy_descriptions,
            taxonomy_sources) = map(list, zip(*combined))  # ✅ FIX #2: keep lists aligned

        result_df = pd.DataFrame({
            "Research ID": document_id,
            "Raw Skill": raw_skills_matched,
            "Taxonomy Skill": mapped_skills,
            "Taxonomy Description": taxonomy_descriptions,
            "Taxonomy Source": taxonomy_sources,
            "Correlation Coefficient": correlations
        })

        log_debug(f"[align] result_df shape={result_df.shape}")
        return result_df
class SkillExtractionService:
    """Main service for skill extraction operations"""
    
    def __init__(
        self,
        model_id: Optional[str] = None, 
        hf_token: Optional[str] = None,
        api_key: Optional[str] = None, 
        use_gpu: Optional[bool] = None
        ):

        self.model_id = model_id
        self.hf_token = hf_token
        self.api_key = api_key
        self.use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
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
        self.router = LLMRouter(self.model_id, self.use_gpu, self.hf_token, self.api_key)

        # Initialize FAISS index
        self.faiss_manager.initialize_index(force_rebuild=False)
    
    def extract_and_align_core(
        self,
        data: pd.DataFrame,
        id_column: str = 'Research ID',
        text_columns: List[str] = None,
        input_type: str = "job_desc",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        levels: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        warnings: bool = True,
        allowed_sources: Optional[List[str]] = None,
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
        
        Returns
        -------
        pd.DataFrame
            DataFrame with extracted and aligned skills
        """
        if text_columns is None:
            text_columns = ["description"]
        

        # --- input validation: ensure `data` is a DataFrame and not None ---
        if data is None:
            raise InvalidInputError("extract_and_align_core: `data` is None. Please pass a pandas.DataFrame with rows to process.")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Apply defaults for top_k and similarity_threshold
        effective_top_k = top_k if top_k is not None else DEFAULT_TOP_K
        effective_threshold = similarity_threshold if similarity_threshold is not None else 0.20
        
        try:
            results = []
            
            for idx, row in data.iterrows():
                try:
                    # Prepare input data
                    input_data = {col: row.get(col, '') for col in text_columns}
                    input_data['id'] = row.get(id_column, str(idx))
                    skills = self.extract_raw_llm_skills(input_data, text_columns)
                    full_description = ' '.join([str(input_data.get(col, '')) for col in text_columns])
                    aligned_df = self.align_extracted_skills(
                        skills, 
                        str(input_data['id']), 
                        full_description,
                        similarity_threshold=effective_threshold,
                        top_k=effective_top_k,
                        allowed_sources = allowed_sources
                    )

                    results.extend(aligned_df.to_dict('records'))
        
                except Exception as e:
                    if warnings:
                        print(f"Warning: Failed to process row {idx}: {e}")
                    continue
            df = pd.DataFrame(results)
            df.to_csv("skills_alignment_results.csv", index=False, encoding="utf-8")
            return pd.DataFrame(df)
            
        except Exception as e:
            raise LAiSERError(f"Batch extraction failed: {e}")

    def extract_raw_llm_skills(self,input_data,text_columns):
        
        text_blob = " ".join(str(input_data.get(col, "")) for col in text_columns).strip()
        extraction_prompt = self.prompt_builder.build_skill_extraction_prompt(input_text=text_blob,input_type="job_desc")
        response = self.router.generate(extraction_prompt)
        skills = self.llm_parser._parse_skills_from_response(response)
        if not skills:
            preview = response.strip().replace("\n", " ")[:200]
            print(f"Warning: failed to parse skills from response: {preview}")
        return skills

    def align_extracted_skills(self, raw_skills: List[str], document_id: str = '0',description: str = '',similarity_threshold: float = 0.20,top_k: int = DEFAULT_TOP_K,allowed_sources: Optional[List[str]] = None,) -> pd.DataFrame:
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
            return pd.DataFrame(columns=['Research ID', 'Description', 'Raw Skill', 'Taxonomy Skill', 'Skill Tag', 'Correlation Coefficient'])
        
        if not isinstance(raw_skills, list):
            print(f"Warning: raw_skills is not a list, converting from {type(raw_skills)}")
            raw_skills = [str(raw_skills)] if raw_skills else []
        
        return self.alignment_service.align_skills_to_taxonomy(
            raw_skills,
            document_id,
            description,
            similarity_threshold,
            top_k,
            allowed_sources=allowed_sources,
        )
