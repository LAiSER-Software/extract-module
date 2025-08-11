"""
Refactored Skill Extractor - Main interface for skill extraction

This module provides a clean interface for skill extraction while maintaining
backward compatibility with the existing API.
"""

import torch
import spacy
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from laiser.config import DEFAULT_BATCH_SIZE, DEFAULT_TOP_K
from laiser.exceptions import LAiSERError, InvalidInputError
from laiser.services import SkillExtractionService
from laiser.llm_models.model_loader import load_model_from_vllm, load_model_from_transformer
from laiser.llm_models.llm_router import llm_router
from laiser.llm_methods import get_completion, get_completion_vllm, get_ksa_details


class SkillExtractorRefactored:
    """
    Refactored skill extractor with improved separation of concerns.
    
    This class provides a clean interface while delegating specific responsibilities
    to appropriate service classes.
    """
    
    def __init__(
        self, 
        model_id: Optional[str] = None, 
        hf_token: Optional[str] = None,
        api_key: Optional[str] = None, 
        use_gpu: Optional[bool] = None
    ):
        """
        Initialize the skill extractor.
        
        Parameters
        ----------
        model_id : str, optional
            Model ID for the LLM
        hf_token : str, optional
            HuggingFace token for accessing gated repositories
        api_key : str, optional
            API key for external services (e.g., Gemini)
        use_gpu : bool, optional
            Whether to use GPU for model inference
        """
        self.model_id = model_id
        self.hf_token = hf_token
        self.api_key = api_key
        self.use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        
        # Initialize service layer
        self.skill_service = SkillExtractionService()
        
        # Initialize model components
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.nlp = None
        
        # Initialize based on configuration
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize required components based on configuration"""
        try:
            # Initialize SpaCy model
            self._initialize_spacy()
            
            # Initialize LLM components
            if self.model_id == 'gemini':
                print("Using Gemini API for skill extraction...")
                # No local model needed for Gemini
            elif self.use_gpu and torch.cuda.is_available():
                print("GPU available. Initializing vLLM model...")
                self._initialize_vllm()
            else:
                print("Using CPU/transformer model...")
                self._initialize_transformer()
                
        except Exception as e:
            raise LAiSERError(f"Failed to initialize components: {e}")
    
    def _initialize_spacy(self):
        """Initialize SpaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_lg")
            print("Loaded en_core_web_lg model successfully.")
        except OSError:
            print("Downloading en_core_web_lg model...")
            spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")
    
    def _initialize_vllm(self):
        """Initialize vLLM model"""
        try:
            self.llm = load_model_from_vllm(self.model_id, self.hf_token)
        except Exception as e:
            print(f"Failed to initialize vLLM: {e}")
            print("Falling back to transformer model...")
            self._initialize_transformer()
    
    def _initialize_transformer(self):
        """Initialize transformer model"""
        try:
            self.tokenizer, self.model = load_model_from_transformer(self.model_id, self.hf_token)
        except Exception as e:
            print(f"Failed to initialize transformer model: {e}")
            # For CPU fallback, we might want to use SkillNer or other alternatives
            print("Consider using SkillNer for CPU-only extraction.")
    
    def extract_skills(
        self,
        input_text: Union[str, Dict[str, str], pd.DataFrame],
        input_type: str = "job_desc",
        method: str = "basic",
        id_column: str = "job_id"
    ) -> List[str]:
        """
        Extract skills from input text.
        
        Parameters
        ----------
        input_text : str, dict, or DataFrame
            Input text or data to extract skills from
        input_type : str
            Type of input ("job_desc" or "syllabus")
        method : str
            Extraction method ("basic" or "ksa")
        
        Returns
        -------
        List[str]
            List of extracted skills
        """
        try:
            if isinstance(input_text, str):
                input_data = {"description": input_text}
            elif isinstance(input_text, dict):
                input_data = input_text
            
            if method == "basic":
                return self._extract_basic_skills(input_data, id_column= id_column)
            elif method == "ksa":
                return self._extract_ksa_skills(input_data, input_type, id_column= id_column)
            else:
                raise InvalidInputError(f"Unknown extraction method: {method}")
                
        except Exception as e:
            raise LAiSERError(f"Skill extraction failed: {e}")

    def _extract_basic_skills(self, input_data: Dict[str, str], input_type: str, id_column: str) -> List[str]:
        """Extract basic skills using simple prompts"""
        if self.model_id == 'gemini':
            # Use Gemini API
            from laiser.services import PromptBuilder
            prompt = PromptBuilder.build_skill_extraction_prompt(input_data, input_type)
            response = llm_router(prompt, self.model_id, self.use_gpu, self.llm, 
                                self.tokenizer, self.model, self.api_key)
            
            from laiser.services import ResponseParser
            return ResponseParser.parse_skill_extraction_response(response)
        
        elif self.llm is not None:
            # Use vLLM
            df = pd.DataFrame([input_data])
            text_columns = ["description"] if input_type == "job_desc" else ["description", "learning_outcomes"]
            result = get_completion_vllm(df, text_columns, id_column, input_type, self.llm, 1)
            return [item.get('Skill', '') for item in result if 'Skill' in item]
        
        elif self.model is not None and self.tokenizer is not None:
            # Use transformer model
            text_columns = ["description"] if input_type == "job_desc" else ["description", "learning_outcomes"]
            return get_completion(input_data, text_columns, input_type, self.model, self.tokenizer)
        
        else:
            raise LAiSERError("No suitable model available for skill extraction")

    def _extract_ksa_skills(self, input_data: Dict[str, str], input_type: str, id_column: str) -> List[Dict[str, Any]]:
        """Extract skills with KSA details"""
        if self.llm is not None:
            df = pd.DataFrame([input_data])
            text_columns = ["description"] if input_type == "job_desc" else ["description", "learning_outcomes"]
            return get_completion_vllm(df, text_columns, id_column, input_type, self.llm, 1)
        else:
            raise LAiSERError("KSA extraction requires vLLM model")
    
    def align_skills(self, raw_skills: List[str], document_id: str = '0') -> pd.DataFrame:
        """
        Align raw skills to taxonomy.
        
        Parameters
        ----------
        raw_skills : List[str]
            List of raw extracted skills
        document_id : str
            Document identifier
        
        Returns
        -------
        pd.DataFrame
            DataFrame with aligned skills and similarity scores
        """
        return self.skill_service.align_extracted_skills(raw_skills, document_id)
    
    def get_top_esco_skills(self, input_text: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Get top matching ESCO skills for input text.
        
        Parameters
        ----------
        input_text : str
            Input text to find matches for
        top_k : int
            Number of top matches to return
        
        Returns
        -------
        List[Dict[str, Any]]
            List of top matching skills with similarity scores
        """
        return self.skill_service.alignment_service.get_top_esco_skills(input_text, top_k)
    
    def extract_and_align(
        self,
        data: pd.DataFrame,
        id_column: str = 'Research ID',
        text_columns: List[str] = None,
        input_type: str = "job_desc",
        top_k: Optional[int] = None,
        levels: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        warnings: bool = True
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
            Number of top skills to return
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
        
        try:
            results = []
            
            for idx, row in data.iterrows():
                try:
                    # Prepare input data
                    input_data = {col: row.get(col, '') for col in text_columns}
                    input_data['id'] = row.get(id_column, str(idx))
                    
                    # Extract skills
                    if levels:
                        extracted = self._extract_ksa_skills(input_data, input_type)
                        for item in extracted:
                            item[id_column] = input_data['id']
                            results.append(item)
                    else:
                        skills = self._extract_basic_skills(input_data, input_type)
                        aligned = self.align_skills(skills, str(input_data['id']))
                        results.extend(aligned.to_dict('records'))
                        
                except Exception as e:
                    if warnings:
                        print(f"Warning: Failed to process row {idx}: {e}")
                    continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            raise LAiSERError(f"Batch extraction failed: {e}")
    
    def get_skill_details(
        self, 
        skill: str, 
        description: str, 
        num_knowledge: int = 3, 
        num_abilities: int = 3
    ) -> Dict[str, List[str]]:
        """
        Get detailed KSA information for a specific skill.
        
        Parameters
        ----------
        skill : str
            Skill name
        description : str
            Context description
        num_knowledge : int
            Number of knowledge items to extract
        num_abilities : int
            Number of ability items to extract
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary with 'knowledge' and 'abilities' keys
        """
        try:
            knowledge, abilities = get_ksa_details(
                skill, description, self.model_id, self.use_gpu, 
                self.llm, self.tokenizer, self.model, self.api_key,
                num_knowledge, num_abilities
            )
            
            return {
                'knowledge': knowledge,
                'abilities': abilities
            }
        except Exception as e:
            raise LAiSERError(f"Failed to get skill details: {e}")


# Backward compatibility: alias to the original class name
Skill_Extractor = SkillExtractorRefactored
