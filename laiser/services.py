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
            # Find the content between model tags
            pattern = r'<start_of_turn>model\\s*<eos>(.*?)<eos>\\s*$'
            match = re.search(pattern, response, re.DOTALL)

            if match:
                content = match.group(1).strip()
                lines = [line.strip() for line in content.split('\\n') if line.strip()]
                skills = [line[1:].strip() for line in lines if line.startswith('-')]
                return skills
            return []
        except Exception as e:
            raise SkillExtractionError(f"Failed to parse skill extraction response: {e}")
    
    @staticmethod
    def parse_ksa_extraction_response(response: str) -> List[Dict[str, Any]]:
        """Parse KSA extraction response"""
        try:
            out = []
            # Split into items, handling optional '->' prefix and multi-line input
            items = [item.strip() for item in response.split('->') if item.strip()]

            for item in items:
                skill_data = {}
                try:
                    # Extract skill
                    skill_match = re.search(r"Skill:\\s*([^,\\n]+)", item)
                    if skill_match:
                        skill_data['Skill'] = skill_match.group(1).strip()

                    # Extract level
                    level_match = re.search(r"Level:\\s*(\\d+)", item)
                    if level_match:
                        skill_data['Level'] = int(level_match.group(1).strip())

                    # Extract knowledge required (multi-line support)
                    knowledge_match = re.search(r"Knowledge Required:\\s*(.*?)(?=\\s*Task Abilities:|\\s*$)", item, re.DOTALL)
                    if knowledge_match:
                        knowledge_raw = knowledge_match.group(1).strip()
                        skill_data['Knowledge Required'] = [k.strip() for k in knowledge_raw.split(',') if k.strip()]

                    # Extract task abilities (multi-line support)
                    task_match = re.search(r"Task Abilities:\\s*(.*?)(?=\\s*$)", item, re.DOTALL)
                    if task_match:
                        task_raw = task_match.group(1).strip()
                        skill_data['Task Abilities'] = [t.strip() for t in task_raw.split(',') if t.strip()]

                    if skill_data:  # Only add if we found some data
                        out.append(skill_data)
                except Exception:
                    continue

            return out
        except Exception as e:
            raise SkillExtractionError(f"Failed to parse KSA extraction response: {e}")
    
    @staticmethod
    def parse_ksa_details_response(response: str) -> Tuple[List[str], List[str]]:
        """Parse KSA details response"""
        try:
            json_match = re.search(r"\\{.*\\}", response, re.DOTALL)
            if not json_match:
                return [], []

            parsed = json.loads(json_match.group())
            knowledge = parsed.get("Knowledge Required", [])
            task_abilities = parsed.get("Task Abilities", [])

            # Ensure they are lists
            if not isinstance(knowledge, list):
                knowledge = [str(knowledge)]
            if not isinstance(task_abilities, list):
                task_abilities = [str(task_abilities)]

            return knowledge, task_abilities
        except Exception as e:
            raise SkillExtractionError(f"Failed to parse KSA details response: {e}")


class SkillAlignmentService:
    """Service for aligning extracted skills with taxonomies"""
    
    def __init__(self, data_access: DataAccessLayer, faiss_manager: FAISSIndexManager):
        self.data_access = data_access
        self.faiss_manager = faiss_manager
    
    def get_top_esco_skills(self, input_text: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Get top matching ESCO skills for given text"""
        try:
            # Get embedding for input text
            model = self.data_access.get_embedding_model()
            query_embedding = model.encode([input_text], convert_to_numpy=True)
            
            # Search using FAISS
            results = self.faiss_manager.search_similar_skills(query_embedding[0], top_k)
            return results
        except Exception as e:
            raise SkillExtractionError(f"Failed to get top ESCO skills: {e}")
    
    def align_skills_to_taxonomy(self, raw_skills: List[str], document_id: str = '0') -> pd.DataFrame:
        """Align raw skills to skill taxonomy using similarity"""
        try:
            combined_df = self.data_access.load_combined_skills()
            model = self.data_access.get_embedding_model()
            
            # Get embeddings for skills in taxonomy
            skill_embeddings = model.encode(combined_df['SkillLabel'].tolist(), convert_to_numpy=True)
            
            aligned_results = []
            
            for skill in raw_skills:
                if not skill or len(skill.strip()) == 0:
                    continue
                
                # Get embedding for raw skill
                skill_embedding = model.encode([skill], convert_to_numpy=True)
                
                # Calculate similarities
                similarities = np.dot(skill_embeddings, skill_embedding.T).flatten()
                
                # Find best match
                best_idx = np.argmax(similarities)
                best_similarity = similarities[best_idx]
                
                aligned_results.append({
                    'Research ID': document_id,
                    'Skill Name': skill,
                    'Skill Tag': combined_df.iloc[best_idx]['SkillLabel'],
                    'Correlation Coefficient': float(best_similarity)
                })
            
            return pd.DataFrame(aligned_results)
        except Exception as e:
            raise SkillExtractionError(f"Failed to align skills to taxonomy: {e}")


class SkillExtractionService:
    """Main service for skill extraction operations"""
    
    def __init__(self):
        self.data_access = DataAccessLayer()
        self.faiss_manager = FAISSIndexManager(self.data_access)
        self.alignment_service = SkillAlignmentService(self.data_access, self.faiss_manager)
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        
        # Initialize FAISS index
        self.faiss_manager.initialize_index()
    
    def extract_skills_basic(
        self, 
        input_data: Union[str, Dict[str, str]], 
        input_type: str = "job_desc"
    ) -> List[str]:
        """Extract basic skills from text using simple extraction"""
        if isinstance(input_data, str):
            input_data = {"description": input_data}
        
        prompt = self.prompt_builder.build_skill_extraction_prompt(input_data, input_type)
        # Note: This would need to be connected to the actual LLM inference
        # For now, returning empty list as placeholder
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
        if isinstance(input_data, str):
            input_data = {"description": input_data}
        
        # Get relevant ESCO skills for context
        esco_skills = self.alignment_service.get_top_esco_skills(
            input_data.get("description", ""), 
            top_k=10
        )
        esco_skill_names = [skill['Skill'] for skill in esco_skills]
        
        prompt = self.prompt_builder.build_ksa_extraction_prompt(
            input_data, input_type, num_skills, num_knowledge, num_abilities, esco_skill_names
        )
        
        # Note: This would need to be connected to the actual LLM inference
        # For now, returning empty list as placeholder
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
    
    def align_extracted_skills(
        self, 
        raw_skills: List[str], 
        document_id: str = '0'
    ) -> pd.DataFrame:
        """Align extracted skills to taxonomy"""
        return self.alignment_service.align_skills_to_taxonomy(raw_skills, document_id)
