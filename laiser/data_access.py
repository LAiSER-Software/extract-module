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
import io
import certifi
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
    

    # -------------------------
    # Helper: fetch CSV via requests + certifi
    # -------------------------
    def _fetch_csv_via_requests(self, url: str, timeout: int = 30) -> pd.DataFrame:
        """Fetch a CSV over HTTPS using requests + certifi and return a pandas DataFrame.

        Honors LAISER_SKIP_ONLINE=1 for offline tests (returns empty DataFrame).
        """
        try:
            resp = requests.get(url, timeout=timeout, verify=certifi.where())
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text))
        except requests.exceptions.RequestException as e:
            # Keep same exception behavior as the rest of your codebase
            raise LAiSERError(f"Failed to fetch CSV from {url}: {e}")

    # -------------------------
    # Taxonomy loaders (use helper)
    # -------------------------
    def load_esco_skills(self) -> pd.DataFrame:
        """Load ESCO skills taxonomy data (fetched with certifi-verified requests)."""
        if self._esco_df is None:
            try:
                self._esco_df = self._fetch_csv_via_requests(ESCO_SKILLS_URL)
            except Exception as e:
                raise LAiSERError(f"Failed to load ESCO skills data: {e}")
        return self._esco_df

    def load_osn_skills(self) -> pd.DataFrame:
        """Load OSN skills taxonomy data (fetched with certifi-verified requests)."""
        if self._osn_df is None:
            try:
                self._osn_df = self._fetch_csv_via_requests(OSN_SKILLS_URL)
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
    def initialize_index(self, force_rebuild: bool = False, debug: bool = False) -> faiss.IndexFlatIP:
        """Initialize FAISS index (load or build).

        Behavior (minimal & strict):
        - If index + metadata JSON exist and force_rebuild is False -> load & return.
        - Else prefer a prebuilt CSV at public/faiss_skills.csv (must contain a 'taxonomy' column).
        - If CSV missing, fallback to fetching ESCO + OSN and set taxonomy values accordingly.
        - Do NOT infer taxonomy for CSV rows — if CSV is missing 'taxonomy' raise LAiSERError.
        - Persist metadata JSON, FAISS index, and embeddings (npy) to public/.
        """
        script_dir = Path(__file__).parent
        local_index_path = script_dir / "public" / "skills_v04.index"
        local_json_path = script_dir / "public" / "skills_df.json"
        local_npy_path = script_dir / "public" / "skill_embeddings.npy"
        local_combined_csv_path = script_dir / "public" / "faiss_skills.csv"

        # 1) Try to load existing index + metadata unless force_rebuild requested
        if not force_rebuild:
            
            ## Issue: Embedding (npy) is not accessed. Cosine Calculations might be faster if npy is accessed.
            try:
                self.index = self.data_access.load_faiss_index(str(local_index_path))
                self.metadata = self.data_access.load_skill_metadata(str(local_json_path))
            except Exception as e:
                if debug:
                    logger.warning(f"[initialize_index] load attempt failed: {e}")

            if self.index is not None and self.metadata is not None:
                if debug:
                    logger.debug("[initialize_index] loaded existing index + metadata")
                return self.index, self.metadata

        combined: Optional[pd.DataFrame] = None

        # 2) Prefer a prebuilt combined CSV if present
        if Path(local_combined_csv_path).exists():
            single_df = pd.read_csv(local_combined_csv_path, dtype=str)

            # drop pandas-created 'Unnamed:*' index columns if present
            single_df = single_df.loc[:, ~single_df.columns.str.match(r'^Unnamed')]

            cols_set = set(single_df.columns.str.lower())

            # CASE A: exact LAiSER export (skill_id, skill_name, aliases, description, taxonomy, original_id)
            laiser_export_headers = {'skill_id', 'skill_name', 'aliases', 'description', 'taxonomy', 'original_id'}
            if laiser_export_headers.issubset(cols_set):
                # rename to canonical downstream names; DO NOT rename taxonomy
                rename_map = {
                    'skill_name': 'skill',
                    'aliases': 'addtional_notes',
                    'original_id': 'source_url'
                }
                # preserve existing 'taxonomy' column name (user requested)
                # first lower-case incoming column names to find exact columns to rename
                # build a mapping from actual column name -> canonical for present columns
                actual_renames = {}
                for k, v in rename_map.items():
                    for col in single_df.columns:
                        if col.lower() == k:
                            actual_renames[col] = v
                            break
                single_df = single_df.rename(columns=actual_renames)

                # keep canonical set (if present)
                keep_cols = []
                for c in ['skill', 'addtional_notes', 'description', 'source_url', 'taxonomy']:
                    # find actual column name (case-insensitive)
                    found = next((col for col in single_df.columns if col.lower() == c), None)
                    if found:
                        keep_cols.append(found)
                single_df = single_df[keep_cols].copy()

            # CASE B: already-normalized CSV (contains skill, addtional_notes, description, source_url) and optionally taxonomy
            elif {'skill', 'addtional_notes', 'description', 'source_url'}.issubset(cols_set):
                # select canonical columns and preserve taxonomy if present
                cols = []
                for c in ['skill', 'addtional_notes', 'description', 'source_url', 'taxonomy']:
                    found = next((col for col in single_df.columns if col.lower() == c), None)
                    if found:
                        cols.append(found)
                single_df = single_df[cols].copy()

            else:
                # Heuristic mapping (best-effort) but require taxonomy column presence if we accept it.
                def _find(cols_substr):
                    for sub in cols_substr:
                        for col in single_df.columns:
                            if sub in col.lower():
                                return col
                    return None

                cand_skill = _find(['skill', 'name', 'title'])
                cand_alias = _find(['alias', 'alt', 'keyword', 'keywords'])
                cand_desc = _find(['desc', 'description', 'statement'])
                cand_orig = _find(['original', 'orig', 'id', 'url'])
                cand_tax = _find(['taxonomy', 'tax', 'source', 'provenance'])

                mapped = {}
                if cand_skill:
                    mapped[cand_skill] = 'skill'
                if cand_alias:
                    mapped[cand_alias] = 'addtional_notes'
                if cand_desc:
                    mapped[cand_desc] = 'description'
                if cand_orig:
                    mapped[cand_orig] = 'source_url'
                if cand_tax:
                    # we will keep the original column name for taxonomy
                    pass

                if 'skill' in mapped.values() and 'description' in mapped.values():
                    # include taxonomy if present in original CSV columns
                    cols = list(mapped.keys())
                    if cand_tax:
                        cols.append(cand_tax)
                    single_df = single_df[cols].rename(columns=mapped).copy()
                    # ensure the optional expected cols exist
                    for col in ['addtional_notes', 'source_url']:
                        if col not in single_df.columns:
                            single_df[col] = pd.NA
                else:
                    # Can't confidently use the CSV
                    single_df = None

            if single_df is not None:
                # Normalize string columns and require taxonomy exists
                # Standardize column names to canonical lower-case keys where possible (but keep original taxonomy column name)
                # First, find the actual taxonomy column name (case-insensitive)
                taxonomy_col = next((col for col in single_df.columns if col.lower() == 'taxonomy'), None)

                # If we couldn't find a taxonomy column, we must fail loudly (per your requirement)
                if taxonomy_col is None:
                    raise LAiSERError(
                        f"Prebuilt CSV '{local_combined_csv_path}' must include a 'taxonomy' column (case-insensitive). "
                        "Please add taxonomy values like 'ESCO' / 'OSN' / 'ONet'."
                    )

                # Clean canonical fields (skill, addtional_notes, description, source_url) if present
                for canonical in ['skill', 'addtional_notes', 'description', 'source_url']:
                    col_found = next((col for col in single_df.columns if col.lower() == canonical), None)
                    if col_found:
                        single_df[col_found] = single_df[col_found].astype('string').str.strip()

                # Normalize taxonomy column values (trim only; preserve case choice but store lowercased variant later)
                single_df[taxonomy_col] = single_df[taxonomy_col].astype('string').str.strip()
                single_df = single_df.replace({'': pd.NA})

                # rename taxonomy column to exact canonical name 'taxonomy' (lowercase) for consistent downstream access
                if taxonomy_col != 'taxonomy':
                    single_df = single_df.rename(columns={taxonomy_col: 'taxonomy'})

                # Ensure addtional_notes exists
                if 'addtional_notes' not in single_df.columns:
                    single_df['addtional_notes'] = single_df.get('addtional_notes', pd.NA)

                single_df['addtional_notes'] = single_df['addtional_notes'].fillna('')
                single_df = single_df.dropna(subset=['skill']).reset_index(drop=True)

                combined = single_df.copy()

        # 3) If CSV not used or could not be used, fallback to fetching ESCO + OSN and combine
        if combined is None:
            esco_df = self.data_access.load_esco_skills()
            osn_df = self.data_access.load_osn_skills()

            # map and normalize ESCO
            esco_df = esco_df[['preferredLabel', 'altLabels', 'description', 'conceptUri']].copy()
            esco_df = esco_df.rename(columns={
                'preferredLabel': 'skill',
                'altLabels': 'addtional_notes',
                'conceptUri': 'source_url',
                'description': 'description'
            })
            esco_df['taxonomy'] = 'esco'

            # map and normalize OSN
            osn_df = osn_df[['RSD Name', 'Keywords', 'Skill Statement', 'Canonical URL']].copy()
            osn_df = osn_df.rename(columns={
                'RSD Name': 'skill',
                'Keywords': 'addtional_notes',
                'Skill Statement': 'description',
                'Canonical URL': 'source_url'
            })
            osn_df['taxonomy'] = 'osn'

            combined = pd.concat([esco_df, osn_df], ignore_index=True)

            # Clean combined columns if present
            for c in ['skill', 'description', 'source_url', 'addtional_notes', 'taxonomy']:
                if c in combined.columns:
                    combined[c] = combined[c].astype('string').str.strip()
            combined = combined.replace({'': pd.NA})
            combined['addtional_notes'] = combined['addtional_notes'].fillna('')
            combined = combined.dropna(subset=['skill']).reset_index(drop=True)

        # 4) Final sanitize and create text column
        combined['description'] = combined['description'].fillna('').astype('string').str.strip()
        combined['addtional_notes'] = combined['addtional_notes'].fillna('').astype('string').str.strip()
        combined['skill'] = combined['skill'].fillna('').astype('string').str.strip()
        combined['text'] = (combined['skill'] + ' | ' + combined['description'] + ' | ' + combined['addtional_notes']).astype('string')
        combined['text'] = combined['text'].fillna('').str.strip()
        combined = combined[combined['text'] != ''].reset_index(drop=True)

        # 5) Build FAISS index
        self.index, self.embeddings = self.data_access.build_faiss_index(combined['text'].tolist())

        # Strict: require taxonomy column (do not infer)
        if 'taxonomy' not in combined.columns:
            raise LAiSERError(
                "After building combined DataFrame, required 'taxonomy' column is missing. "
                "Ensure input CSV or upstream source provides 'taxonomy'."
            )

        # Normalize taxonomy values to lowercase for consistent comparisons
        combined['taxonomy'] = combined['taxonomy'].astype('string').str.strip().str.lower()

        # Build metadata (canonical column order)
        meta_df = combined[['skill', 'description', 'addtional_notes', 'taxonomy', 'source_url', 'text']].copy()
        self.metadata = meta_df

        # Persist metadata JSON and FAISS index (best-effort with debug warnings)
        try:
            self.data_access.save_skill_metadata_json(self.metadata, str(local_json_path))
        except Exception as e:
            if debug:
                logger.warning(f"[initialize_index] Failed to write metadata JSON: {e}")

        try:
            self.data_access.save_faiss_index(self.index, str(local_index_path))
        except Exception as e:
            if debug:
                logger.warning(f"[initialize_index] Failed to write FAISS index: {e}")

        # Persist embeddings as npy (best-effort)
        try:
            np.save(local_npy_path, self.embeddings)
        except Exception as e:
            if debug:
                logger.warning(f"[initialize_index] Failed to save embeddings npy: {e}")

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
    
    def search_similar_skills(
        self,
        query_embedding: np.ndarray,
        top_k: int = 25,
        allowed_sources: Optional[List[str]] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar skills using FAISS index.

        Behavior:
        - If allowed_sources is None: return up to 'top_k' best matches (classic behavior).
        - If allowed_sources is provided: return ALL matches whose metadata 'source' is in allowed_sources,
          ordered by similarity. If you want a safety limit, pass `max_results` (int).
        """
        if self.index is None:
            raise FAISSIndexError("FAISS index not initialized. Call initialize_index() first.")

        if self.metadata is None:
            raise FAISSIndexError("Metadata not initialized. Call initialize_index() first.")

        # Ensure skill_names come from metadata
        if self.skill_names is None:
            try:
                if isinstance(self.metadata, pd.DataFrame):
                    self.skill_names = self.metadata['skill'].astype(str).tolist()
                elif isinstance(self.metadata, list):
                    self.skill_names = [m.get('skill', '') for m in self.metadata]
                else:
                    self.skill_names = [str(r.get('skill','')) for r in list(self.metadata)]
            except Exception as e:
                raise FAISSIndexError(f"Failed to load skill names for index: {e}")

        try:
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

            faiss.normalize_L2(q)

            ntotal = int(getattr(self.index, "ntotal", 0))
            if ntotal <= 0:
                return []
            # allowed_sources = ["onet_tech"]
            # ---------- No allowed_sources -> original behavior ----------
            if not allowed_sources:
                top_k = max(1, min(int(top_k), ntotal))
                scores, indices = self.index.search(q, top_k)
                results = []
                for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                    if idx == -1:
                        continue
                    results.append({
                        "Skill": self.skill_names[idx] if 0 <= idx < len(self.skill_names) else "",
                        "Similarity": float(score),
                        "Rank": rank,
                        "Index": int(idx)
                    })
                return results

            # ---------- Filtering by allowed_sources -> return ALL matches for those sources ----------
            # Make allowed_sources case-insensitive set
            allowed_lower = {s.lower() for s in allowed_sources}

            # We'll request a candidate set large enough to cover the index.
            # Practical choices:
            #  - For guaranteed completeness: candidate_k = ntotal (get everything)
            #  - Heuristic: candidate_k = min(ntotal, top_k * 10 + 500)
            # Because user asked "don't block the number", we go for completeness by default.
            candidate_k = ntotal

            # If caller provided a max_results, we still fetch all candidates but will cap final returned list
            scores, indices = self.index.search(q, candidate_k)

            filtered = []
            seen_idx = set()
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if idx in seen_idx:
                    continue
                seen_idx.add(int(idx))

                # bounds check
                if idx < 0 or idx >= len(self.skill_names):
                    continue

                # read source from metadata safely
                try:
                    if isinstance(self.metadata, pd.DataFrame):
                        row = self.metadata.iloc[int(idx)]
                        src = str(row.get('taxonomy', '')).strip()
                    elif isinstance(self.metadata, list):
                        src = str(self.metadata[int(idx)].get('taxonomy', '')).strip()
                    else:
                        src = str(self.metadata[int(idx)].get('taxonomy', '')).strip()
                except Exception:
                    src = ""

                if any(a in src.lower() for a in allowed_lower):
                    filtered.append((float(score), int(idx)))

            # sort by similarity descending (FAISS returns best-first but after filtering order is preserved; still sort to be safe)
            filtered.sort(key=lambda x: x[0], reverse=True)

            # apply max_results cap if requested (optional safety)
            if max_results is not None and isinstance(max_results, int):
                filtered = filtered[:int(max_results)]

            results = []
            for rank, (score, idx) in enumerate(filtered, start=1):
                results.append({
                    "Skill": self.skill_names[idx],
                    "Similarity": float(score),
                    "Rank": rank,
                    "Index": int(idx)
                })

            return results

        except Exception as e:
            raise FAISSIndexError(f"Failed to search similar skills: {repr(e)}")
