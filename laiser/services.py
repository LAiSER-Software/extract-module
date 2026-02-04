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
from laiser.exceptions import SkillExtractionError, InvalidInputError
from laiser.data_access import DataAccessLayer, FAISSIndexManager

class PromptBuilder:
    """Builds prompts for different types of skill extraction tasks"""
    
    @staticmethod
    def build_skill_extraction_prompt(input_text: Dict[str, str], input_type: str) -> str:
        """Build prompt for basic skill extraction"""
        if input_type == "job_desc":
            return SKILL_EXTRACTION_PROMPT_JOB.format(query=input_text.get("description", ""))
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


class ResponseParser:
    """Parses responses from LLM models"""
    
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
    ) -> pd.DataFrame:
        """
        Align raw skills to taxonomy using the FAISS index.
        
        Parameters
        ----------
        raw_skills : List[str]
            List of raw extracted skills to align
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
            DataFrame with aligned skills, limited to top_k results above threshold
        """
        mapped_skills = []
        raw_skills_matched = []
        skill_tags = []
        correlations = []


        model = self.data_access.get_embedding_model()

        # Get SkillLabel to SkillTag mapping
        label_to_tag_mapping = self.data_access.get_skill_label_to_tag_mapping()

        for skill in raw_skills:
            query_vec = model.encode([skill], normalize_embeddings=True)
            # Search for the single best match for each raw skill
            results = self.faiss_manager.search_similar_skills(
                np.array(query_vec).astype("float32"), top_k=1
            )

            if not results:
                continue

            best = results[0]
            similarity = float(best.get("Similarity", 0.0))

            # Only include matches above the threshold
            if similarity >= similarity_threshold:
                canonical_skill = str(best.get("Skill", "")).strip()
                if not canonical_skill:
                    continue

                mapped_skills.append(canonical_skill)
                raw_skills_matched.append(skill)

                # Get the corresponding SkillTag
                skill_tag = label_to_tag_mapping.get(canonical_skill, "")
                skill_tags.append(skill_tag)
                correlations.append(similarity)

        # Apply top_k limit: sort by correlation (descending) and take top_k
        if len(mapped_skills) > top_k:
            # Create temporary list of tuples for sorting
            combined = list(zip(correlations, raw_skills_matched, mapped_skills, skill_tags))
            # Sort by correlation score descending
            combined.sort(key=lambda x: x[0], reverse=True)
            # Take only top_k results
            combined = combined[:top_k]
            # Unzip back to separate lists
            correlations, raw_skills_matched, mapped_skills, skill_tags = zip(*combined) if combined else ([], [], [], [])
            correlations = list(correlations)
            raw_skills_matched = list(raw_skills_matched)
            mapped_skills = list(mapped_skills)
            skill_tags = list(skill_tags)

        # Create DataFrame with all the required columns
        result_df = pd.DataFrame({
            "Research ID": document_id,
            "Description": description,
            "Raw Skill": raw_skills_matched,
            "Taxonomy Skill": mapped_skills,
            "Skill Tag": skill_tags,
            "Correlation Coefficient": correlations
        })

        return result_df


class SkillExtractionService:
    """Main service for skill extraction operations"""
    
    def __init__(self):
        print("Initalisng skill extraction service")
        self.data_access = DataAccessLayer()
        self.faiss_manager = FAISSIndexManager(self.data_access)
        self.alignment_service = SkillAlignmentService(self.data_access, self.faiss_manager)
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        # Initialize FAISS index
        self.faiss_manager.initialize_index(force_rebuild=False)
        print("done Initalisng skill extraction service")
    
    def align_extracted_skills(self, raw_skills: List[str], document_id: str = '0',description: str = '',similarity_threshold: float = 0.20,top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
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
            raw_skills, document_id, description, similarity_threshold, top_k
        )
       
    def extract_skills_basic(
        self, 
        input_data: Union[str, Dict[str, str]], 
        input_type: str = "job_desc"
    ) -> List[str]:
        """Extract basic skills from text using simple extraction"""
        try:
            if isinstance(input_data, str):
                input_data = {"description": input_data}
            
            prompt = self.prompt_builder.build_skill_extraction_prompt(input_data, input_type)
            # Note: This would need to be connected to the actual LLM inference
            # For now, returning empty list as placeholder
            result = []
            
            # Ensure we always return a list, never None
            return result if result is not None else []
        except Exception as e:
            print(f"Warning: Skill extraction failed: {e}")
            return []
    
    def extract_skills_with_ksa(
        self,
        input_data: Union[str, Dict[str, str]],
        input_type: str = "job_desc",
        num_skills: int = 5,
        num_knowledge: str = "3-5",
        num_abilities: str = "3-5"
    ) -> List[Dict[str, Any]]:
        """Extract skills with Knowledge, Skills, Abilities details"""
        try:
            if isinstance(input_data, str):
                input_data = {"description": input_data}
            
            # Build prompt without ESCO skills context
            prompt = self.prompt_builder.build_ksa_extraction_prompt(
                input_data, input_type, num_skills, num_knowledge, num_abilities, None
            )
            
            # Note: This would need to be connected to the actual LLM inference
            # For now, returning empty list as placeholder
            result = []
            
            # Ensure we always return a list, never None
            return result if result is not None else []
        except Exception as e:
            print(f"Warning: KSA extraction failed: {e}")
            return []
    
    def get_skill_details(
        self, 
        skill: str, 
        context: str, 
        num_knowledge: int = 3, 
        num_abilities: int = 3
    ) -> Tuple[List[str], List[str]]:
        """Get detailed KSA information for a specific skill"""
        prompt = self.prompt_builder.build_ksa_details_prompt(
            skill, context, num_knowledge, num_abilities
        )
        
        # Note: This would need to be connected to the actual LLM inference
        # For now, returning empty lists as placeholder
        return [], []
    
