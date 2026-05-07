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

from typing import Dict, List, Optional

import pandas as pd

from laiser.config import DEFAULT_BATCH_SIZE
from laiser.services import SkillExtractionService


class SkillExtractorRefactored:
    """
    Refactored skill extractor with improved separation of concerns.

    This class provides a clean interface while delegating specific responsibilities
    to appropriate service classes.
    """

    @staticmethod
    def _expand_allowed_sources(allowed_sources: Optional[List[str]]) -> Optional[List[str]]:
        if not allowed_sources:
            return allowed_sources

        expanded: List[str] = []
        seen = set()
        source_map = {
            "esco": ["esco", "esco_knowledge", "esco_task"],
            "onet": ["onet_skill", "onet_tech", "onet_knowledge", "onet_task"],
            "osn": ["osn"],
        }

        for source in allowed_sources:
            normalized = str(source or "").strip().lower()
            if not normalized:
                continue
            mapped_values = source_map.get(normalized, [normalized])
            for value in mapped_values:
                if value in seen:
                    continue
                seen.add(value)
                expanded.append(value)

        return expanded

    def __init__(
        self,
        model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        api_key: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        backend: Optional[str] = None,
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
        backend : str, optional
            Backend to use for LLM inference (e.g., "llama_cpp", "huggingface", "openai", "gemini")
        """
        # Initialize service layer
        self.skill_service = SkillExtractionService(
            model_id=model_id,
            api_key=api_key,
            hf_token=hf_token,
            use_gpu=use_gpu,
            backend=backend,
        )

    def extract_concepts(
        self,
        data: pd.DataFrame,
        id_column: str = "Research ID",
        text_columns: List[str] = None,
        input_type: str = "job_desc",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        levels: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        warnings: bool = False,
        allowed_sources: Optional[List[str]] = None,
        concepts: List[str] = None,
        extract: List[str] = None,
        return_edges: bool = False,
        similarity_thresholds: Optional[Dict[str, float]] = None,
        timing: bool = False,
    ):
        """
        Extract concepts from text and align them to taxonomy data.

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
            Per-type thresholds. Keys: "skill", "knowledge", "task".
            Defaults: {"skill": 0.20, "knowledge": 0.45, "task": 0.55}
        levels : bool
            Whether to extract skill levels
        batch_size : int
            Batch size for processing
        warnings : bool
            Whether to show warnings
        concepts : list, optional
            Types to extract: "skills", "knowledge", "tasks", or ["all"].
            Defaults to ["skills"].
        extract : list, optional
            Backward-compatible alias for `concepts`.
        return_edges : bool, optional
            If True, return {"nodes": pd.DataFrame, "edges": pd.DataFrame} where
            "edges" contains ENABLES edges (Knowledge → Task per skill).
            If False (default), return a plain pd.DataFrame.
        timing : bool, optional
            If True, print per-stage timing for extraction and alignment steps.

        Returns
        -------
        pd.DataFrame  (when return_edges=False)
            Normalized mixed-concept rows with:
            Research ID, Type, Raw Concept, Taxonomy Concept,
            Taxonomy Description, Taxonomy Source, Source Url, Correlation Coefficient
        dict          (when return_edges=True): {"nodes": pd.DataFrame, "edges": pd.DataFrame}
        """
        if concepts is not None and extract is not None and concepts != extract:
            raise ValueError("Pass either `concepts` or `extract`, not both")

        requested_concepts = concepts if concepts is not None else extract
        expanded_sources = self._expand_allowed_sources(allowed_sources)

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
            allowed_sources=expanded_sources,
            extract=requested_concepts,
            return_edges=return_edges,
            similarity_thresholds=similarity_thresholds,
            timing=timing,
        )

    def extract_and_align(
        self,
        data: pd.DataFrame,
        id_column: str = "Research ID",
        text_columns: List[str] = None,
        input_type: str = "job_desc",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        levels: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        warnings: bool = False,
        allowed_sources: Optional[List[str]] = None,
        extract: List[str] = None,
        return_edges: bool = False,
        similarity_thresholds: Optional[Dict[str, float]] = None,
        timing: bool = False,
    ):
        """
        Backward-compatible skills-only extraction wrapper.

        This method is retained to avoid breaking existing integrations that
        still call `extract_and_align(...)`. It always runs the skills path.
        Use `extract_concepts(...)` for mixed concept extraction.
        """
        result = self.extract_concepts(
            data=data,
            id_column=id_column,
            text_columns=text_columns,
            input_type=input_type,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            levels=levels,
            batch_size=batch_size,
            warnings=warnings,
            allowed_sources=allowed_sources,
            extract=["skills"],
            return_edges=return_edges,
            similarity_thresholds=similarity_thresholds,
            timing=timing,
        )

        if isinstance(result, dict) and isinstance(result.get("nodes"), pd.DataFrame):
            updated = dict(result)
            updated["nodes"] = result["nodes"].rename(
                columns={
                    "Raw Concept": "Raw Skill",
                    "Taxonomy Concept": "Taxonomy Skill",
                }
            )
            return updated

        if isinstance(result, pd.DataFrame):
            return result.rename(
                columns={
                    "Raw Concept": "Raw Skill",
                    "Taxonomy Concept": "Taxonomy Skill",
                }
            )

        return result


Skill_Extractor = SkillExtractorRefactored
