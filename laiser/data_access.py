"""
Data access layer for LAiSER project

This module handles all data loading and external API calls.
"""

import os
import requests
import pandas as pd
import faiss
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer

from laiser.config import (
    ESCO_SKILLS_URL, 
    COMBINED_SKILLS_URL, 
    FAISS_INDEX_URL, 
    DEFAULT_EMBEDDING_MODEL
)
from laiser.exceptions import FAISSIndexError, LAiSERError


class DataAccessLayer:
    """Handles data loading and external API calls"""
    
    def __init__(self):
        self.embedding_model = None
        self._esco_df = None
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
                print("Loading ESCO skill taxonomy data...")
                self._esco_df = pd.read_csv(ESCO_SKILLS_URL)
            except Exception as e:
                raise LAiSERError(f"Failed to load ESCO skills data: {e}")
        return self._esco_df
    
    def load_combined_skills(self) -> pd.DataFrame:
        """Load combined skills taxonomy data"""
        if self._combined_df is None:
            try:
                print("Loading combined skill taxonomy data...")
                self._combined_df = pd.read_csv(COMBINED_SKILLS_URL)
            except Exception as e:
                raise LAiSERError(f"Failed to load combined skills data: {e}")
        return self._combined_df
    
    def build_faiss_index(self, skill_names: List[str]) -> faiss.IndexFlatIP:
        """Build FAISS index for given skill names"""
        try:
            print("Building FAISS index...")
            model = self.get_embedding_model()
            embeddings = model.encode(skill_names, convert_to_numpy=True, show_progress_bar=True)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            return index
        except Exception as e:
            raise FAISSIndexError(f"Failed to build FAISS index: {e}")
    
    def save_faiss_index(self, index: faiss.IndexFlatIP, file_path: str) -> None:
        """Save FAISS index to file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            faiss.write_index(index, file_path)
            print(f"FAISS index saved to {file_path}")
        except Exception as e:
            raise FAISSIndexError(f"Failed to save FAISS index: {e}")
    
    def load_faiss_index(self, file_path: str) -> Optional[faiss.IndexFlatIP]:
        """Load FAISS index from file"""
        try:
            if os.path.exists(file_path):
                print(f"Loading FAISS index from {file_path}...")
                return faiss.read_index(file_path)
            return None
        except Exception as e:
            raise FAISSIndexError(f"Failed to load FAISS index: {e}")
    
    def download_faiss_index(self, url: str, local_path: str) -> bool:
        """Download FAISS index from URL"""
        try:
            print(f"Downloading FAISS index from {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if response.headers.get("Content-Type") != "application/octet-stream":
                raise ValueError(f"Unexpected content type: {response.headers.get('Content-Type')}")
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(response.content)
            print("Download complete.")
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
    
    def initialize_index(self, force_rebuild: bool = False) -> faiss.IndexFlatIP:
        """Initialize FAISS index (load or build)"""
        # Define paths
        script_dir = Path(__file__).parent
        local_index_path = script_dir / "public" / "esco_faiss_index.index"
        
        # Try to load existing index
        if not force_rebuild:
            self.index = self.data_access.load_faiss_index(str(local_index_path))
            
            if self.index is None:
                # Try to download from remote
                if self.data_access.download_faiss_index(FAISS_INDEX_URL, str(local_index_path)):
                    self.index = self.data_access.load_faiss_index(str(local_index_path))
        
        # Build new index if loading failed
        if self.index is None or force_rebuild:
            print("Building new FAISS index...")
            esco_df = self.data_access.load_esco_skills()
            self.skill_names = esco_df["preferredLabel"].tolist()
            self.index = self.data_access.build_faiss_index(self.skill_names)
            self.data_access.save_faiss_index(self.index, str(local_index_path))
        
        return self.index
    
    def search_similar_skills(self, query_embedding: np.ndarray, top_k: int = 25) -> List[Dict[str, Any]]:
        """Search for similar skills using FAISS index"""
        if self.index is None:
            raise FAISSIndexError("FAISS index not initialized")
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.skill_names):
                    results.append({
                        'Skill': self.skill_names[idx],
                        'Similarity': float(score),
                        'Rank': i + 1
                    })
            
            return results
        except Exception as e:
            raise FAISSIndexError(f"Failed to search similar skills: {e}")
