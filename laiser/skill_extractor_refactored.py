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

import json
import re

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

    def _parse_skills_from_response(self, response: str) -> List[str]:
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
    
    def align_skills(
        self, 
        raw_skills: List[str], 
        document_id: str = '0', 
        description: str = '',
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K
    ) -> pd.DataFrame:
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
        similarity_threshold : float
            Minimum similarity score for a match to be included (default: 0.20)
        top_k : int
            Maximum number of aligned skills to return per document (default: 25)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with aligned skills and similarity scores
        """
        return self.skill_service.align_extracted_skills(
            raw_skills, document_id, description, similarity_threshold, top_k
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
        response = llm_router(prompt, self.model_id, self.use_gpu, self.llm, 
                                self.tokenizer, self.model, self.api_key)
        cleaned = response.split("### CLEANED JOB DESCRIPTION:")[-1].strip()
        return cleaned
        
    def fs_gold_skill_extraction_prompt(cleaned_description):
        standard_prompt = f"""
        task: "Skill Extraction from Job Descriptions"

        description: |
        You are an expert AI system specialized in extracting technical and professional skills from job descriptions for workforce analytics.
        Your goal is to analyze the following job description and output only the specific skill names that are required, mentioned, or strongly implied.

        extraction_instructions:
        - Extract only concrete, job-relevant skills (not soft traits, company values, or general workplace behaviors).
        - Include a skill if it is clearly mentioned or strongly implied as necessary for the role.
        - Exclude company policies, benefit programs, HR or legal statements, and generic terms (e.g., "communication," "leadership") unless used in a technical/professional context.
        - Use only concise skill phrases (prefer noun phrases, avoid sentences).
        - Do not invent new skills or make assumptions beyond the provided text.

        examples:
        Example 1 (Focus: Soft Skills & Communication) Input: "Strong verbal and written communication skills, with the ability to explain complex technical concepts clearly to both technical and non-technical audiences. Confident presenter, capable of articulating insights, results, and strategies to stakeholders." Output: ['Strong verbal and written communication skills', 'explain complex technical concepts', 'Confident presenter', 'capable of articulating insights']

        Example 2 (Focus: Math & Technical Background) Input: "Qualified candidates will have a strong mathematical background (statistics, linear algebra, calculus, probability, and optimization). Experience with deep learning, natural language processing, or application of large language models is preferred." Output: ['Strong mathematical background', 'statistics, linear algebra, calculus, probability, and optimization', 'deep learning', 'natural language processing', 'application of large language models']

        Example 3 (Focus: Core Responsibilities) Input: "Lead the research, design, implementation, and deployment of Machine Learning algorithms. Assist and enable C3 AI’s federal customers to build their own applications on the C3 AI Suite. Contribute to the design of new features." Output: ['Lead the research, design, implementation, and deployment of Machine Learning algorithms', 'Assist and enable C3 AI’s federal customers to build their own applications on the C3 AI Suite.', 'Contribute to the design and implementation of new features of the C3 AI Suite.']

        formatting_rules:
        - Return the output as valid JSON.
        - The JSON must have a single key "skills" whose value is a list of skill strings.
        - Each skill string must be between 1 and 5 words.
        - Do not include explanations, metadata, or anything other than the JSON object.

        job_description: |
        {cleaned_description}

        ### OUTPUT FORMAT
        {{
        "skills": [
            "skill1",
            "skill2",
            "skill3"
        ]
        }}
        """
        return standard_prompt

fs_gold_prompt = fs_gold_skill_extraction_prompt(cleaned_description)

    def extract_and_map_skills(self,input_data,text_columns):
        # 1. Clean job description (build text from dict)
        text_blob = " ".join(str(input_data.get(col, "")) for col in text_columns).strip()
        cleaned_desc = self.strong_preprocessing_prompt(text_blob)
        # print("Cleaned Desc:::::::",cleaned_desc)
        extraction_prompt = self.skill_extraction_prompt(cleaned_desc)
        response = llm_router(extraction_prompt, self.model_id, self.use_gpu, self.llm, 
                                self.tokenizer, self.model, self.api_key)
        skills = self._parse_skills_from_response(response)
        if not skills:
            preview = response.strip().replace("\n", " ")[:200]
            print(f"Warning: failed to parse skills from response: {preview}")
        return skills

    def extract_and_align(
        self,
        data: pd.DataFrame,
        id_column: str = 'Research ID',
        text_columns: List[str] = None,
        input_type: str = "job_desc",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
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
                    skills = self.extract_and_map_skills(input_data, text_columns)
                    full_description = ' '.join([str(input_data.get(col, '')) for col in text_columns])
                    aligned_df = self.align_skills(
                        skills, 
                        str(input_data['id']), 
                        full_description,
                        similarity_threshold=effective_threshold,
                        top_k=effective_top_k
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

# Backward compatibility: alias to the original class name
Skill_Extractor = SkillExtractorRefactored
