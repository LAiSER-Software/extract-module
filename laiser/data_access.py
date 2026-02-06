"""
Module Description:
-------------------
This module provides the data access layer for the LAiSER project, handling all data loading, external API calls, and FAISS index management for skill extraction and research.

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
- Requires CSV files for ESCO and combined skills taxonomies.
- Requires access to remote FAISS index files and skill embedding models.

Output/Return Format:
---------------------
- Returns pandas DataFrames for skills data.
- Produces and manages FAISS index files for skill similarity search.

"""

"""
Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     08/10/2025      Satya Phanindra K.            Initial Version


TODO:
-----
- 1: Add more robust error handling for data loading.
- 2: Implement caching for remote data sources.
"""

import os
import requests
import pandas as pd
import faiss
import numpy as np
from pathlib import Path
import json
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer

from laiser.config import (
    ESCO_SKILLS_URL, 
    OSN_SKILLS_URL,
    COMBINED_SKILLS_URL, 
    FAISS_INDEX_URL, 
    DEFAULT_EMBEDDING_MODEL
)
from laiser.exceptions import FAISSIndexError, LAiSERError

import logging
logger = logging.getLogger(__name__)

class DataAccessLayer:
    """Handles data loading and external API calls"""
    
    def __init__(self):
        self.embedding_model = None
        self._esco_df = None
        self._osn_df = None

        self._combined_df = None
    
    def get_embedding_model(self) -> SentenceTransformer:
        """Get or initialize the embedding model"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        return self.embedding_model
    
    def load_esco_skills(self) -> pd.DataFrame:
        """Load ESCO skills taxonomy data"""
        if self._esco_df is None:
            try:
                self._esco_df = pd.read_csv(ESCO_SKILLS_URL)
            except Exception as e:
                raise LAiSERError(f"Failed to load ESCO skills data: {e}")
        return self._esco_df
    
    def load_osn_skills(self) -> pd.DataFrame:
        """Load OSN skills taxonomy data"""
        if self._osn_df is None:
            try:
                self._osn_df = pd.read_csv(OSN_SKILLS_URL)
            except Exception as e:
                raise LAiSERError(f"Failed to load OSN skills data: {e}")
        return self._osn_df
    
    def load_skill_metadata(self,file_path: str) -> pd.DataFrame:
        """Load skills metadata JSON generated during FAISS index build"""
        if self._combined_df is None:
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Skills metadata file not found at {file_path}. "
                        "Build or download the FAISS index first."
                    )

                self._combined_df = pd.read_json(
                    file_path,
                    orient="records"
                )
            except Exception as e:
                raise LAiSERError(f"Failed to load skills metadata: {e}")

        return self._combined_df


    def load_combined_skills(self) -> pd.DataFrame:
        """Load combined skills taxonomy data"""
        if self._combined_df is None:
            try:
                self._combined_df = pd.read_csv(COMBINED_SKILLS_URL)
            except Exception as e:
                raise LAiSERError(f"Failed to load combined skills data: {e}")
        return self._combined_df

    def get_skill_label_to_tag_mapping(self) -> Dict[str, str]:
        """Create mapping from SkillLabel to SkillTag"""
        combined_df = self.load_combined_skills()
        if combined_df is None or combined_df.empty:
            return {}

        # Create mapping from SkillLabel to SkillTag
        mapping = {}
        for _, row in combined_df.iterrows():
            skill_label = str(row.get('SkillLabel', '')).strip()
            skill_tag = str(row.get('SkillTag', '')).strip()
            if skill_label and skill_tag:
                mapping[skill_label] = skill_tag

        return mapping

    def get_skill_tag_to_label_mapping(self) -> Dict[str, str]:
        """Create mapping from SkillTag to SkillLabel"""
        combined_df = self.load_combined_skills()
        if combined_df is None or combined_df.empty:
            return {}

        # Create mapping from SkillTag to SkillLabel
        mapping = {}
        for _, row in combined_df.iterrows():
            skill_label = str(row.get('SkillLabel', '')).strip()
            skill_tag = str(row.get('SkillTag', '')).strip()
            if skill_label and skill_tag:
                mapping[skill_tag] = skill_label

        return mapping
    
    def build_faiss_index(self, text: List[str]) -> faiss.IndexFlatIP:
        """Build FAISS index for given skill names"""
        try:
            model = self.get_embedding_model()
            embeddings = model.encode(text,normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(np.asarray(embeddings, dtype='float32'))
            return index, embeddings 
        except Exception as e:
            raise FAISSIndexError(f"Failed to build FAISS index: {e}")
    
    def save_faiss_index(self, index: faiss.IndexFlatIP, file_path: str) -> None:
        """Save FAISS index to file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            faiss.write_index(index, file_path)
        except Exception as e:
            raise FAISSIndexError(f"Failed to save FAISS index: {e}")
    
    def save_skill_metadata_json(self, metadata_df: pd.DataFrame, file_path: str) -> None:
        """
        Save skills metadata DataFrame to JSON file.
        Expects a pandas DataFrame and serializes it safely.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            records = metadata_df.to_dict(orient="records")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise RuntimeError(f"Failed to save skill metadata JSON: {e}")
    def load_faiss_index(self, file_path: str) -> Optional[faiss.IndexFlatIP]:
        """Load FAISS index from file"""
        try:
            if os.path.exists(file_path):
                return faiss.read_index(file_path)
            return None
        except Exception as e:
            raise FAISSIndexError(f"Failed to load FAISS index: {e}")
    
    def download_faiss_index(self, url: str, local_path: str) -> bool:
        """Download FAISS index from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if response.headers.get("Content-Type") != "application/octet-stream":
                raise ValueError(f"Unexpected content type: {response.headers.get('Content-Type')}")
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            print(f"Failed to download FAISS index: {e}")
            return False


class FAISSIndexManager:
    """Manages FAISS index operations"""
    
    def __init__(self, data_access: DataAccessLayer):
        self.data_access = data_access
        self.index = None
        self.skill_names = None
        self.embeddings = None
        self.metadata = None
    
    # Issue: Do we even need this? Can't this be done in init
    # Issue [GFI_OddEven]: Split these into two seperate modules load and build index
    def initialize_index(self, force_rebuild: bool = False, debug: bool = True ) -> faiss.IndexFlatIP:
        """Initialize FAISS index (load or build)"""

        # Issue [GFI_HelloWorld]: config this not hardcode
        script_dir = Path(__file__).parent
        local_index_path = script_dir / "public" / "skills_v04.index"
        local_json_path = script_dir / "public" / "skills_df.json" 
        local_npy_path = script_dir / "public" / "skill_embeddings.npy"
        local_combined_csv_path = script_dir / "public" / "combined.csv"
        if not force_rebuild:
            
            ## Issue: Embedding (npy) is not accessed. Cosine Calculations might be faster if npy is accessed.
            try:
                self.index = self.data_access.load_faiss_index(str(local_index_path))
                self.metadata = self.data_access.load_skill_metadata(str(local_json_path))
            except Exception as e:
                if debug:
                    logger.warning(f"[initialize_index] Load failed, rebuilding: {e}")
            
            # Issue: Handle all casses where any file is missing and we can recreate those without force rebuild. Might also verify files available and then decide whether to rebuild or not.
            if self.index is not None and self.metadata is not None:
                return self.index, self.metadata

        esco_df = self.data_access.load_esco_skills()
        osn_df = self.data_access.load_osn_skills()
        # === Standardize columns ===
        esco_df = esco_df[['preferredLabel','altLabels', 'description','conceptUri']].copy()
        esco_df = esco_df.rename(columns={'preferredLabel': 'skill', 'description': 'description', 'conceptUri': 'source_url','altLabels': 'addtional_notes' })
        osn_df = osn_df[['RSD Name','Keywords','Skill Statement', 'Canonical URL']].copy()
        osn_df = osn_df.rename(columns={'RSD Name': 'skill', 'Skill Statement': 'description', 'Canonical URL': 'source_url', 'Keywords': 'addtional_notes' })
        esco_df['source'] = 'esco' 
        osn_df['source'] = 'osn'    
        combined = pd.concat([esco_df, osn_df], ignore_index=True)

        for c in ['skill', 'description', 'source_url', 'addtional_notes']:
            combined[c] = combined[c].astype('string').str.strip()
        combined = combined.replace({'': pd.NA})

        combined['addtional_notes'] = combined['addtional_notes'].fillna('')

        combined = combined.dropna(subset=['skill'])
        combined.to_csv(local_combined_csv_path,index=True,encoding="utf-8")
        combined['text'] = (
            combined['skill'].astype('string').str.strip() + ' | ' +
            combined['description'].astype('string').str.strip() + ' | ' +
            combined['addtional_notes'].astype('string').str.strip()
        )

        self.index, self.embeddings = (self.data_access.build_faiss_index(combined['text'].tolist()))
        
        meta_df = combined[['skill', 'description', 'addtional_notes', 'source', 'source_url', 'text']].copy()
        self.metadata = meta_df

        self.data_access.save_skill_metadata_json(self.metadata, str(local_json_path))
        self.data_access.save_faiss_index(self.index, str(local_index_path))

        # Issue [GFI_OOPS]: config this not hardcode
        np.save(local_npy_path, self.embeddings)

        return self.index, self.metadata 

    def get_metadata(self):
        """
        Return loaded skill metadata.
        """
        if self.metadata is None:
            raise FAISSIndexError(
                "Metadata not initialized. Call initialize_index() first."
            )
        return self.metadata

    def search_similar_skills(self, query_embedding: np.ndarray, top_k: int = 25) -> List[Dict[str, Any]]:
        """Search for similar skills using FAISS index"""
        if self.index is None:
            raise FAISSIndexError("FAISS index not initialized. Call initialize_index() first.")

        if self.skill_names is None:
            try:
                esco_df = self.data_access.load_esco_skills()
                self.skill_names = esco_df["preferredLabel"].tolist()
            except Exception as e:
                raise FAISSIndexError(f"Failed to load skill names for index: {e}")

        try:
            # Ensure correct dtype/shape/layout
            q = np.asarray(query_embedding, dtype=np.float32)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            if not q.flags["C_CONTIGUOUS"]:
                q = np.ascontiguousarray(q)

            # Dimension check
            d_index = int(self.index.d)
            d_query = int(q.shape[1])
            if d_query != d_index:
                raise FAISSIndexError(
                    f"Embedding dimension mismatch: query={d_query}, index={d_index}. "
                    f"Ensure DEFAULT_EMBEDDING_MODEL matches the model used to build the index."
                )

            # Normalize and cap top_k
            faiss.normalize_L2(q)
            if getattr(self.index, "ntotal", 0) <= 0:
                return []
            top_k = max(1, min(int(top_k), int(self.index.ntotal)))

            # Search
            scores, indices = self.index.search(q, top_k)

            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx == -1:
                    continue  # FAISS may return -1 for padded results
                if 0 <= idx < len(self.skill_names):
                    results.append({
                        "Skill": self.skill_names[idx],
                        "Similarity": float(score),
                        "Rank": rank,
                        "Index": int(idx)
                    })

            return results
        except Exception as e:
            # Bubble up a helpful message
            raise FAISSIndexError(f"Failed to search similar skills: {repr(e)}")

