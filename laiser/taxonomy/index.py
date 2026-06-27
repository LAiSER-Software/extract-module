import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

from laiser.exceptions import FAISSIndexError, LAiSERError
from laiser.taxonomy.loader import DataAccessLayer

logger = logging.getLogger(__name__)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _search_index(
    *,
    index,
    metadata: pd.DataFrame,
    names: List[str],
    query_embedding: np.ndarray,
    top_k: int,
    allowed_sources: Optional[List[str]],
    name_key: str,
    error_prefix: str,
    max_results: Optional[int] = None,
    check_dimensions: bool = False,
) -> List[Dict[str, Any]]:
    try:
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if not q.flags["C_CONTIGUOUS"]:
            q = np.ascontiguousarray(q)

        if check_dimensions:
            d_index = int(index.d)
            d_query = int(q.shape[1])
            if d_query != d_index:
                raise FAISSIndexError(
                    f"Embedding dimension mismatch: query={d_query}, index={d_index}. "
                    "Ensure DEFAULT_EMBEDDING_MODEL matches the model used to build the index."
                )

        faiss.normalize_L2(q)
        ntotal = int(getattr(index, "ntotal", 0))
        if ntotal <= 0:
            return []

        if not allowed_sources:
            k = max(1, min(int(top_k), ntotal))
            scores, indices = index.search(q, k)
            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx == -1:
                    continue
                name = names[idx] if 0 <= idx < len(names) else ""
                results.append({name_key: name, "Similarity": float(score), "Rank": rank, "Index": int(idx)})
            return results

        allowed_lower = {s.lower() for s in allowed_sources}
        scores, indices = index.search(q, ntotal)
        filtered = []
        seen_idx = set()
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx == -1 or idx in seen_idx or idx >= len(names):
                continue
            seen_idx.add(idx)
            src = str(metadata.iloc[idx].get("taxonomy", "")).strip().lower()
            if any(source in src for source in allowed_lower):
                filtered.append((float(score), idx))

        filtered.sort(key=lambda x: x[0], reverse=True)
        limit = max_results if isinstance(max_results, int) else top_k
        filtered = filtered[: max(1, min(int(limit), len(filtered)))] if filtered else []

        return [
            {name_key: names[idx], "Similarity": float(score), "Rank": rank, "Index": int(idx)}
            for rank, (score, idx) in enumerate(filtered, start=1)
        ]
    except Exception as exc:
        raise FAISSIndexError(f"{error_prefix}: {repr(exc)}")


class FAISSIndexManager:
    """Manage the main skill taxonomy FAISS index."""

    def __init__(self, data_access: DataAccessLayer):
        self.data_access = data_access
        self.index = None
        self.skill_names = None
        self.embeddings = None
        self.metadata = None

    def initialize_index(self, force_rebuild: bool = False, debug: bool = False):
        local_index_path = ASSETS_DIR / "skills_v04.index"
        local_json_path = ASSETS_DIR / "skills_df.json"
        local_npy_path = ASSETS_DIR / "skill_embeddings.npy"
        local_combined_csv_path = ASSETS_DIR / "faiss_skills.csv"

        if not force_rebuild:
            try:
                self.index = self.data_access.load_faiss_index(str(local_index_path))
                self.metadata = self.data_access.load_skill_metadata(str(local_json_path))
            except Exception as exc:
                if debug:
                    logger.warning(f"[initialize_index] load attempt failed: {exc}")

            if self.index is not None and self.metadata is not None:
                if debug:
                    logger.debug("[initialize_index] loaded existing index + metadata")
                return self.index, self.metadata

        combined: Optional[pd.DataFrame] = None

        if Path(local_combined_csv_path).exists():
            single_df = pd.read_csv(local_combined_csv_path, dtype=str)
            single_df = single_df.loc[:, ~single_df.columns.str.match(r"^Unnamed")]
            cols_set = set(single_df.columns.str.lower())

            laiser_export_headers = {"skill_id", "skill_name", "aliases", "description", "taxonomy", "original_id"}
            if laiser_export_headers.issubset(cols_set):
                rename_map = {"skill_name": "skill", "aliases": "addtional_notes", "original_id": "source_url"}
                actual_renames = {}
                for key, value in rename_map.items():
                    for col in single_df.columns:
                        if col.lower() == key:
                            actual_renames[col] = value
                            break
                single_df = single_df.rename(columns=actual_renames)

                keep_cols = []
                for col_name in ["skill", "addtional_notes", "description", "source_url", "taxonomy"]:
                    found = next((col for col in single_df.columns if col.lower() == col_name), None)
                    if found:
                        keep_cols.append(found)
                single_df = single_df[keep_cols].copy()

            elif {"skill", "addtional_notes", "description", "source_url"}.issubset(cols_set):
                cols = []
                for col_name in ["skill", "addtional_notes", "description", "source_url", "taxonomy"]:
                    found = next((col for col in single_df.columns if col.lower() == col_name), None)
                    if found:
                        cols.append(found)
                single_df = single_df[cols].copy()

            else:

                def _find(substrings):
                    for sub in substrings:
                        for col in single_df.columns:
                            if sub in col.lower():
                                return col
                    return None

                cand_skill = _find(["skill", "name", "title"])
                cand_alias = _find(["alias", "alt", "keyword", "keywords"])
                cand_desc = _find(["desc", "description", "statement"])
                cand_orig = _find(["original", "orig", "id", "url"])
                cand_tax = _find(["taxonomy", "tax", "source", "provenance"])

                mapped = {}
                if cand_skill:
                    mapped[cand_skill] = "skill"
                if cand_alias:
                    mapped[cand_alias] = "addtional_notes"
                if cand_desc:
                    mapped[cand_desc] = "description"
                if cand_orig:
                    mapped[cand_orig] = "source_url"

                if "skill" in mapped.values() and "description" in mapped.values():
                    cols = list(mapped.keys())
                    if cand_tax:
                        cols.append(cand_tax)
                    single_df = single_df[cols].rename(columns=mapped).copy()
                    for col_name in ["addtional_notes", "source_url"]:
                        if col_name not in single_df.columns:
                            single_df[col_name] = pd.NA
                else:
                    single_df = None

            if single_df is not None:
                taxonomy_col = next((col for col in single_df.columns if col.lower() == "taxonomy"), None)
                if taxonomy_col is None:
                    raise LAiSERError(
                        f"Prebuilt CSV '{local_combined_csv_path}' must include a 'taxonomy' column (case-insensitive). "
                        "Please add taxonomy values like 'ESCO' / 'OSN' / 'ONet'."
                    )

                for canonical in ["skill", "addtional_notes", "description", "source_url"]:
                    col_found = next((col for col in single_df.columns if col.lower() == canonical), None)
                    if col_found:
                        single_df[col_found] = single_df[col_found].astype("string").str.strip()

                single_df[taxonomy_col] = single_df[taxonomy_col].astype("string").str.strip()
                single_df = single_df.replace({"": pd.NA})

                if taxonomy_col != "taxonomy":
                    single_df = single_df.rename(columns={taxonomy_col: "taxonomy"})

                if "addtional_notes" not in single_df.columns:
                    single_df["addtional_notes"] = single_df.get("addtional_notes", pd.NA)

                single_df["addtional_notes"] = single_df["addtional_notes"].fillna("")
                single_df = single_df.dropna(subset=["skill"]).reset_index(drop=True)
                combined = single_df.copy()

        if combined is None:
            esco_df = self.data_access.load_esco_skills()
            osn_df = self.data_access.load_osn_skills()

            esco_df = esco_df[["preferredLabel", "altLabels", "description", "conceptUri"]].copy()
            esco_df = esco_df.rename(
                columns={
                    "preferredLabel": "skill",
                    "altLabels": "addtional_notes",
                    "conceptUri": "source_url",
                    "description": "description",
                }
            )
            esco_df["taxonomy"] = "esco"

            osn_df = osn_df[["RSD Name", "Keywords", "Skill Statement", "Canonical URL"]].copy()
            osn_df = osn_df.rename(
                columns={
                    "RSD Name": "skill",
                    "Keywords": "addtional_notes",
                    "Skill Statement": "description",
                    "Canonical URL": "source_url",
                }
            )
            osn_df["taxonomy"] = "osn"

            combined = pd.concat([esco_df, osn_df], ignore_index=True)
            for col_name in ["skill", "description", "source_url", "addtional_notes", "taxonomy"]:
                if col_name in combined.columns:
                    combined[col_name] = combined[col_name].astype("string").str.strip()
            combined = combined.replace({"": pd.NA})
            combined["addtional_notes"] = combined["addtional_notes"].fillna("")
            combined = combined.dropna(subset=["skill"]).reset_index(drop=True)

        combined["description"] = combined["description"].fillna("").astype("string").str.strip()
        combined["addtional_notes"] = combined["addtional_notes"].fillna("").astype("string").str.strip()
        combined["skill"] = combined["skill"].fillna("").astype("string").str.strip()
        combined["text"] = (
            combined["skill"] + " | " + combined["description"] + " | " + combined["addtional_notes"]
        ).astype("string")
        combined["text"] = combined["text"].fillna("").str.strip()
        combined = combined[combined["text"] != ""].reset_index(drop=True)

        self.index, self.embeddings = self.data_access.build_faiss_index(combined["text"].tolist())

        if "taxonomy" not in combined.columns:
            raise LAiSERError(
                "After building combined DataFrame, required 'taxonomy' column is missing. "
                "Ensure input CSV or upstream source provides 'taxonomy'."
            )

        combined["taxonomy"] = combined["taxonomy"].astype("string").str.strip().str.lower()
        self.metadata = combined[["skill", "description", "addtional_notes", "taxonomy", "source_url", "text"]].copy()

        try:
            self.data_access.save_skill_metadata_json(self.metadata, str(local_json_path))
        except Exception as exc:
            if debug:
                logger.warning(f"[initialize_index] Failed to write metadata JSON: {exc}")

        try:
            self.data_access.save_faiss_index(self.index, str(local_index_path))
        except Exception as exc:
            if debug:
                logger.warning(f"[initialize_index] Failed to write FAISS index: {exc}")

        try:
            np.save(local_npy_path, self.embeddings)
        except Exception as exc:
            if debug:
                logger.warning(f"[initialize_index] Failed to save embeddings npy: {exc}")

        return self.index, self.metadata

    def get_metadata(self):
        if self.metadata is None:
            raise FAISSIndexError("Metadata not initialized. Call initialize_index() first.")
        return self.metadata

    def search_similar_skills(
        self,
        query_embedding: np.ndarray,
        top_k: int = 25,
        allowed_sources: Optional[List[str]] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            allowed_sources=allowed_sources,
            max_results=max_results,
        )

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 25,
        allowed_sources: Optional[List[str]] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if self.index is None:
            raise FAISSIndexError("FAISS index not initialized. Call initialize_index() first.")
        if self.metadata is None:
            raise FAISSIndexError("Metadata not initialized. Call initialize_index() first.")

        if self.skill_names is None:
            try:
                if isinstance(self.metadata, pd.DataFrame):
                    self.skill_names = self.metadata["skill"].astype(str).tolist()
                elif isinstance(self.metadata, list):
                    self.skill_names = [m.get("skill", "") for m in self.metadata]
                else:
                    self.skill_names = [str(r.get("skill", "")) for r in list(self.metadata)]
            except Exception as exc:
                raise FAISSIndexError(f"Failed to load skill names for index: {exc}")

        return _search_index(
            index=self.index,
            metadata=self.metadata,
            names=self.skill_names,
            query_embedding=query_embedding,
            top_k=top_k,
            allowed_sources=allowed_sources,
            name_key="Skill",
            error_prefix="Failed to search similar skills",
            max_results=max_results,
            check_dimensions=True,
        )


class _BaseTaxonomyFAISSIndexManager:
    INDEX_FILENAME: str = ""
    META_FILENAME: str = ""
    CSV_FILENAME: str = ""
    LABEL: str = ""

    def __init__(self, data_access: DataAccessLayer):
        self.data_access = data_access
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: Optional[pd.DataFrame] = None
        self._item_names: Optional[List[str]] = None

    def initialize_index(self, force_rebuild: bool = False) -> None:
        index_path = ASSETS_DIR / self.INDEX_FILENAME
        meta_path = ASSETS_DIR / self.META_FILENAME
        csv_path = ASSETS_DIR / self.CSV_FILENAME

        if not force_rebuild and index_path.exists() and meta_path.exists():
            try:
                self.index = self.data_access.load_faiss_index(str(index_path))
                self.metadata = pd.read_json(str(meta_path), orient="records")
                logger.info(f"[{self.LABEL}FAISSIndex] Loaded cached index ({len(self.metadata)} entries).")
                return
            except Exception as exc:
                logger.warning(f"[{self.LABEL}FAISSIndex] Cache load failed: {exc}. Rebuilding from CSV.")

        if not csv_path.exists():
            logger.warning(
                f"[{self.LABEL}FAISSIndex] Taxonomy CSV not found at {csv_path}. "
                f"Run scripts/build_{self.LABEL.lower()}_index.py to generate it. "
                f"Alignment for {self.LABEL} will be unavailable."
            )
            return

        df = pd.read_csv(str(csv_path), dtype=str)
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

        for col in ("name", "description", "taxonomy"):
            if col not in df.columns:
                raise LAiSERError(
                    f"[{self.LABEL}FAISSIndex] CSV '{csv_path}' missing required column '{col}'. Re-run the pipeline script."
                )
            df[col] = df[col].fillna("").astype(str).str.strip()

        df = df.dropna(subset=["name"]).reset_index(drop=True)
        df = df[df["name"] != ""].reset_index(drop=True)
        df["taxonomy"] = df["taxonomy"].str.lower()
        df["text"] = df["name"] + " | " + df["description"]

        logger.info(f"[{self.LABEL}FAISSIndex] Building index from {len(df)} entries…")
        self.index, _ = self.data_access.build_faiss_index(df["text"].tolist())
        self.metadata = df

        try:
            self.data_access.save_faiss_index(self.index, str(index_path))
            self.data_access.save_skill_metadata_json(self.metadata, str(meta_path))
            logger.info(f"[{self.LABEL}FAISSIndex] Index saved to {index_path}.")
        except Exception as exc:
            logger.warning(f"[{self.LABEL}FAISSIndex] Failed to persist index: {exc}")

    def get_metadata(self) -> pd.DataFrame:
        if self.metadata is None:
            raise FAISSIndexError(f"{self.LABEL} FAISS metadata not initialized. Call initialize_index() first.")
        return self.metadata

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 25,
        allowed_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.index is None or self.metadata is None:
            return []

        if self._item_names is None:
            self._item_names = self.metadata["name"].astype(str).tolist()

        return _search_index(
            index=self.index,
            metadata=self.metadata,
            names=self._item_names,
            query_embedding=query_embedding,
            top_k=top_k,
            allowed_sources=allowed_sources,
            name_key="Name",
            error_prefix=f"[{self.LABEL}FAISSIndex] Search failed",
        )


class KnowledgeFAISSIndexManager(_BaseTaxonomyFAISSIndexManager):
    INDEX_FILENAME = "knowledge_v05.index"
    META_FILENAME = "knowledge_df.json"
    CSV_FILENAME = "knowledge_taxonomy.csv"
    LABEL = "Knowledge"


class TaskFAISSIndexManager(_BaseTaxonomyFAISSIndexManager):
    INDEX_FILENAME = "tasks_v05.index"
    META_FILENAME = "tasks_df.json"
    CSV_FILENAME = "task_taxonomy.csv"
    LABEL = "Task"
