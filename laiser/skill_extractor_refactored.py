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
from typing import List, Any, Optional
from laiser.config import DEFAULT_BATCH_SIZE, DEFAULT_TOP_K
from laiser.services import SkillExtractionService

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
        # Initialize service layer
        self.skill_service = SkillExtractionService(model_id = model_id,api_key=api_key,hf_token= hf_token,use_gpu=use_gpu  )
        
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
        warnings: bool = False
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
        return self.skill_service.extract_and_align_core(
        data=data,
        id_column=id_column,
        text_columns=text_columns,
        input_type=input_type,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        levels=levels,
        batch_size=batch_size,
        warnings=warnings,
    )

Skill_Extractor = SkillExtractorRefactored
