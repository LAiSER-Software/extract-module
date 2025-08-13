"""
Module Description:
-------------------
Refactored Class to extract skills from text and align them to existing taxonomy data efficiently.

Ownership:
----------
Project: Leveraging Artificial intelligence for Skills Extraction and Research (LAiSER)
Owner:  George Washington University Insitute of Public Policy
    Program on Skills, Credentials and Workforce Policy
    Media and Public Affairs Building
    805 21st Street NW
    Washington, DC 20052
    PSCWP@gwu.edu
    https://gwipp.gwu.edu/program-skills-credentials-workforce-policy-pscwp

License:
--------
Copyright 2025 George Washington University Insitute of Public Policy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
(the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Input Requirements:
-------------------
- Pandas Dataframe with ID and Text Column

Output/Return Format:
----------------------------
- Pandas dataframe with below columns:
    - "Research ID": text_id
    - "Skill Name": Raw skill extracted,
    - "Skill Tag": skill tag from taxonomy,
    - "Correlation Coefficient": similarity_score

"""

"""
Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     08/13/2025      Satya Phanindra K.            Initial Version


TODO:
-----

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
                return
            elif self.use_gpu and torch.cuda.is_available():
                print("GPU available. Attempting to initialize vLLM model...")
                try:
                    self._initialize_vllm()
                    if self.llm is not None:
                        print("vLLM initialization successful!")
                        return
                except Exception as e:
                    print(f"WARNING: vLLM initialization failed: {e}")
                    print("Falling back to transformer model...")
                    
                # Fallback to transformer
                try:
                    self._initialize_transformer()
                    if self.model is not None:
                        print("Transformer model fallback successful!")
                        return
                except Exception as e:
                    print(f"WARNING: Transformer model fallback also failed: {e}")
            else:
                print("Using CPU/transformer model...")
                try:
                    self._initialize_transformer()
                    if self.model is not None:
                        print("Transformer model initialization successful!")
                        return
                except Exception as e:
                    print(f"WARNING: Transformer model initialization failed: {e}")
            
            # If all else fails, warn but continue
            print("WARNING: No model successfully initialized. Extraction methods may have limited functionality.")
            print("TIP: Consider using Gemini API by setting model_id='gemini' and providing an api_key.")
                
        except Exception as e:
            raise LAiSERError(f"Critical failure during component initialization: {e}")
    
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
            from laiser.exceptions import VLLMNotAvailableError, ModelLoadError
            self.llm = load_model_from_vllm(self.model_id, self.hf_token)
            print(f"Successfully initialized vLLM with model: {self.model_id}")
        except VLLMNotAvailableError as e:
            print(f"WARNING: vLLM not available: {e}")
            self.llm = None
            raise e
        except ModelLoadError as e:
            print(f"WARNING: vLLM model loading failed: {e}")
            self.llm = None
            raise e
        except Exception as e:
            print(f"WARNING: Unexpected vLLM initialization error: {e}")
            self.llm = None
            raise e
    
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
        input_text: Union[str, Dict[str, str], pd.DataFrame, pd.Series],
        input_type: str = "job_desc",
        method: str = "basic",
        id_column: str = "job_id"
    ) -> List[str]:
        """
        Extract skills from input text.
        
        Parameters
        ----------
        input_text : str, dict, DataFrame, or Series
            Input text or data to extract skills from
        input_type : str
            Type of input ("job_desc" or "syllabus")
        method : str
            Extraction method ("basic" or "ksa")
        id_column : str
            Column name for ID when input is DataFrame/Series
        
        Returns
        -------
        List[str] or List[Dict[str, Any]]
            List of extracted skills (strings for basic, dicts for ksa)
        """
        try:
            # Handle different input types
            if isinstance(input_text, str):
                input_data = {"description": input_text, id_column: "0"}
            elif isinstance(input_text, pd.Series):
                input_data = input_text.to_dict()
                # Ensure we have the required columns
                if "description" not in input_data:
                    # Try to find a description column
                    desc_cols = [col for col in input_data.keys() if 'desc' in col.lower()]
                    if desc_cols:
                        input_data["description"] = input_data[desc_cols[0]]
                    else:
                        raise InvalidInputError("No description column found in input data")
            elif isinstance(input_text, dict):
                input_data = input_text.copy()
            elif isinstance(input_text, pd.DataFrame):
                if len(input_text) > 1:
                    raise InvalidInputError("DataFrame with multiple rows not supported in extract_skills. Use extract_and_align instead.")
                input_data = input_text.iloc[0].to_dict()
            else:
                raise InvalidInputError(f"Unsupported input type: {type(input_text)}")
            
            if method == "basic":
                return self._extract_basic_skills(input_data, input_type, id_column=id_column)
            elif method == "ksa":
                return self._extract_ksa_skills(input_data, input_type, id_column=id_column)
            else:
                raise InvalidInputError(f"Unknown extraction method: {method}")
                
        except Exception as e:
            raise LAiSERError(f"Skill extraction failed: {e}")

    def _extract_basic_skills(self, input_data: Dict[str, str], input_type: str, id_column: str) -> List[str]:
        """Extract basic skills using simple prompts"""
        # Ensure input_data is a dictionary
        if not isinstance(input_data, dict):
            if hasattr(input_data, 'to_dict'):
                input_data = input_data.to_dict()
            else:
                raise InvalidInputError(f"Expected dictionary input, got {type(input_data)}")
        
        if self.model_id == 'gemini':
            # Use Gemini API
            from laiser.services import PromptBuilder
            prompt = PromptBuilder.build_skill_extraction_prompt(input_data, input_type)
            response = llm_router(prompt, self.model_id, self.use_gpu, self.llm, 
                                self.tokenizer, self.model, self.api_key)
            
            from laiser.services import ResponseParser
            parsed_skills = ResponseParser.parse_skill_extraction_response(response)
            return parsed_skills
        
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
            # Fallback: return empty list with warning
            print("Warning: No suitable model available for skill extraction. Returning empty list.")
            return []

    def _extract_ksa_skills(self, input_data: Dict[str, str], input_type: str, id_column: str) -> List[Dict[str, Any]]:
        """Extract skills with KSA details"""
        if self.llm is not None:
            df = pd.DataFrame([input_data])
            text_columns = ["description"] if input_type == "job_desc" else ["description", "learning_outcomes"]
            result = get_completion_vllm(df, text_columns, id_column, input_type, self.llm, 1)
            return result
        elif self.model_id == 'gemini':
            # Use Gemini API for KSA extraction
            from laiser.services import PromptBuilder, ResponseParser
            
            prompt = PromptBuilder.build_ksa_extraction_prompt(
                input_data, input_type, 5, "3-5", "3-5", None
            )
            response = llm_router(prompt, self.model_id, self.use_gpu, self.llm, 
                                self.tokenizer, self.model, self.api_key)
            
            parsed_results = ResponseParser.parse_ksa_extraction_response(response)
            # Add document ID to each result
            for item in parsed_results:
                item[id_column] = input_data.get(id_column, input_data.get('id', '0'))
            return parsed_results
        elif self.model is not None and self.tokenizer is not None:
            # Fallback to basic extraction for transformer models
            print("Warning: KSA extraction not fully supported with transformer models. Using basic extraction.")
            basic_skills = self._extract_basic_skills(input_data, input_type, id_column)
            # Convert basic skills to KSA format
            ksa_results = []
            for skill in basic_skills:
                ksa_results.append({
                    'Skill': skill,
                    'Level': 0,  # Default level
                    'Knowledge Required': [],
                    'Task Abilities': [],
                    id_column: input_data.get(id_column, input_data.get('id', '0'))
                })
            return ksa_results
        else:
            raise LAiSERError("KSA extraction requires either vLLM model, Gemini API, or transformer model. No suitable model available.")
    
    def align_skills(self, raw_skills: List[str], document_id: str = '0', description: str = '') -> pd.DataFrame:
        """
        Align raw skills to taxonomy.
        
        Parameters
        ----------
        raw_skills : List[str]
            List of raw extracted skills
        document_id : str
            Document identifier
        description : str
            Full description text for context
        
        Returns
        -------
        pd.DataFrame
            DataFrame with aligned skills and similarity scores
        """
        return self.skill_service.align_extracted_skills(raw_skills, document_id, description)
    

    
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
                        extracted = self._extract_ksa_skills(input_data, input_type, id_column)
                        for item in extracted:
                            item[id_column] = input_data['id']
                            results.append(item)
                    else:
                        skills = self._extract_basic_skills(input_data, input_type, id_column)
                        print("Extracted raw skills before alignment:", skills)
                        if len(skills) == 0:
                            print("No skills extracted.")
                            continue
                        else:
                            # Create description from text columns
                            full_description = ' '.join([str(input_data.get(col, '')) for col in text_columns])
                            aligned = self.align_skills(skills, str(input_data['id']), full_description)
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
    
    def display_ksa_results(self, skills: List[Dict[str, Any]], detailed: bool = True, show_levels: bool = False) -> None:
        """
        Display KSA extraction results in a formatted way.
        
        Parameters
        ----------
        skills : List[Dict[str, Any]]
            List of extracted skills with KSA details
        detailed : bool, optional
            Whether to show Knowledge Required and Task Abilities (default: True)
        show_levels : bool, optional
            Whether to show skill levels (default: False)
        """
        if not skills:
            print("No skills extracted.")
            return
        
        print(f"Extracted {len(skills)} skills with KSA details")
        
        for i, skill in enumerate(skills, 1):
            skill_name = skill.get('Skill', 'Unknown')
            skill_level = skill.get('Level', 'N/A')
            knowledge_required = skill.get('Knowledge Required', [])
            task_abilities = skill.get('Task Abilities', [])
            
            # Display skill name with optional level
            if show_levels:
                print(f"  {i}. {skill_name} (Level: {skill_level})")
            else:
                print(f"  {i}. {skill_name}")
            
            # Note: ESCO matching functionality has been removed

            if detailed:
                if knowledge_required:
                    print(f"     Knowledge Required: {', '.join(knowledge_required)}")
                
                if task_abilities:
                    print(f"     Task Abilities: {', '.join(task_abilities)}")
                
                print()  # Add empty line for readability
    
    def display_esco_analysis(self, input_text: str, top_k: int = 10) -> None:
        """
        Display top ESCO skill matches for given input text.
        Note: This functionality has been removed as requested.
        
        Parameters
        ----------
        input_text : str
            Input text to analyze
        top_k : int, optional
            Number of top matches to show (default: 10)
        """
        print(f"\nESCO Skills matching functionality has been removed.")
        print("This method is no longer available.")
        return
    
    def extract_skills_to_dataframe(
        self, 
        input_data: Union[str, Dict[str, str], pd.Series], 
        method: str = "ksa",
        input_type: str = "job_desc", 
        id_column: str = "id",
        include_alignment: bool = True
    ) -> pd.DataFrame:
        """
        Extract skills and return results in a CSV-ready DataFrame format.
        
        Parameters
        ----------
        input_data : Union[str, Dict[str, str], pd.Series]
            Input text or data structure
        method : str, optional
            Extraction method: "ksa" or "basic" (default: "ksa")
        input_type : str, optional
            Type of input: "job_desc" or "syllabus" (default: "job_desc")
        id_column : str, optional
            Name of the ID column (default: "id")
        include_alignment : bool, optional
            Whether to include skill taxonomy alignment (default: True)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with extracted skills, levels, KSA details, and optional alignment
        """
        try:
            # Handle different input types
            if isinstance(input_data, pd.Series):
                input_dict = input_data.to_dict()
            elif isinstance(input_data, str):
                input_dict = {"description": input_data, id_column: "0"}
            elif isinstance(input_data, dict):
                input_dict = input_data.copy()
            else:
                raise InvalidInputError(f"Unsupported input data type: {type(input_data)}")
            
            # Ensure ID is present
            if id_column not in input_dict:
                input_dict[id_column] = "0"
            
            # Extract skills based on method
            if method.lower() == "ksa":
                skills = self._extract_ksa_skills(input_dict, input_type, id_column)
                
                # Convert to DataFrame format
                results = []
                for skill in skills:
                    skill_name = skill.get('Skill', '')
                    skill_level = skill.get('Level', 0)
                    knowledge_required = skill.get('Knowledge Required', [])
                    task_abilities = skill.get('Task Abilities', [])
                    doc_id = skill.get(id_column, input_dict.get(id_column, '0'))
                    
                    # Base record
                    record = {
                        id_column: doc_id,
                        'Skill': skill_name,
                        'Level': skill_level,
                        'Knowledge Required': ', '.join(knowledge_required) if knowledge_required else '',
                        'Task Abilities': ', '.join(task_abilities) if task_abilities else '',
                        'description': input_dict.get('description', '')
                    }
                    
                    # Add learning outcomes if syllabus
                    if input_type == "syllabus" and 'learning_outcomes' in input_dict:
                        record['learning_outcomes'] = input_dict.get('learning_outcomes', '')
                    
                    results.append(record)
                
                df = pd.DataFrame(results)
                
                # Add skill alignment if requested
                if include_alignment and len(skills) > 0:
                    skill_names = [skill.get('Skill', '') for skill in skills if skill.get('Skill')]
                    if skill_names:
                        aligned_df = self.align_skills(skill_names, input_dict.get(id_column, '0'))
                        
                        # Merge alignment data
                        if not aligned_df.empty:
                            # Rename columns for clarity
                            aligned_df = aligned_df.rename(columns={
                                'Skill Name': 'Skill',
                                'Skill Tag': 'Aligned_Skill_Tag',
                                'Correlation Coefficient': 'Alignment_Score'
                            })
                            
                            # Merge on skill name
                            df = df.merge(aligned_df[['Skill', 'Aligned_Skill_Tag', 'Alignment_Score']], 
                                        on='Skill', how='left')
                
            else:  # basic method
                basic_skills = self._extract_basic_skills(input_dict, input_type, id_column)
                
                if include_alignment and basic_skills:
                    # For basic extraction, return alignment format
                    aligned_df = self.align_skills(basic_skills, input_dict.get(id_column, '0'))
                    
                    # Add additional columns for consistency
                    aligned_df['Level'] = 0  # No level info in basic extraction
                    aligned_df['Knowledge Required'] = ''
                    aligned_df['Task Abilities'] = ''
                    aligned_df['description'] = input_dict.get('description', '')
                    
                    if input_type == "syllabus" and 'learning_outcomes' in input_dict:
                        aligned_df['learning_outcomes'] = input_dict.get('learning_outcomes', '')
                    
                    # Rename for consistency
                    aligned_df = aligned_df.rename(columns={
                        'Skill Name': 'Skill',
                        'Skill Tag': 'Aligned_Skill_Tag',
                        'Correlation Coefficient': 'Alignment_Score'
                    })
                    
                    df = aligned_df
                else:
                    # Create basic DataFrame without alignment
                    results = []
                    for skill in basic_skills:
                        record = {
                            id_column: input_dict.get(id_column, '0'),
                            'Skill': skill,
                            'Level': 0,
                            'Knowledge Required': '',
                            'Task Abilities': '',
                            'description': input_dict.get('description', '')
                        }
                        
                        if input_type == "syllabus" and 'learning_outcomes' in input_dict:
                            record['learning_outcomes'] = input_dict.get('learning_outcomes', '')
                        
                        results.append(record)
                    
                    df = pd.DataFrame(results)
            
            return df
            
        except Exception as e:
            print(f"Error in extract_skills_to_dataframe: {e}")
            # Return empty DataFrame with expected columns
            columns = [id_column, 'Skill', 'Level', 'Knowledge Required', 'Task Abilities', 'description']
            if input_type == "syllabus":
                columns.append('learning_outcomes')
            if include_alignment:
                columns.extend(['Aligned_Skill_Tag', 'Alignment_Score'])
            return pd.DataFrame(columns=columns)


# Backward compatibility: alias to the original class name
Skill_Extractor = SkillExtractorRefactored
