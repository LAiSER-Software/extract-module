import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from laiser.config import DEFAULT_TOP_K
from laiser.taxonomy.loader import DataAccessLayer

logger = logging.getLogger(__name__)


class AlignmentService:
    """Align extracted items to the active taxonomy index."""

    def __init__(self, data_access: DataAccessLayer, faiss_manager):
        self.data_access = data_access
        self.faiss_manager = faiss_manager

    def align(
        self,
        raw_items: List[str],
        document_id: str = "0",
        description: str = "",
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
        raw_col: str = "Raw Item",
        taxonomy_col: str = "Taxonomy Item",
        allowed_sources: Optional[List[str]] = None,
        debug: bool = False,
    ) -> pd.DataFrame:
        mapped_items: List[str] = []
        raw_matched: List[str] = []
        taxonomy_descriptions: List[str] = []
        taxonomy_sources: List[str] = []
        taxonomy_urls: List[str] = []
        correlations: List[float] = []

        def log_debug(msg: str):
            if debug:
                logger.debug(msg)

        log_debug(f"[align] raw_items={len(raw_items)} threshold={similarity_threshold} top_k={top_k}")

        model = self.data_access.get_embedding_model()
        try:
            metadata = self.faiss_manager.get_metadata()
        except Exception:
            return pd.DataFrame(
                {
                    "Research ID": pd.Series([], dtype=str),
                    raw_col: [],
                    taxonomy_col: [],
                    "Taxonomy Description": [],
                    "Taxonomy Source": [],
                    "Source Url": [],
                    "Correlation Coefficient": [],
                }
            )

        if not raw_items:
            return pd.DataFrame(
                {
                    "Research ID": pd.Series([], dtype=str),
                    raw_col: [],
                    taxonomy_col: [],
                    "Taxonomy Description": [],
                    "Taxonomy Source": [],
                    "Source Url": [],
                    "Correlation Coefficient": [],
                }
            )

        search_fn = getattr(self.faiss_manager, "search_similar", None)
        if search_fn is None:
            search_fn = self.faiss_manager.search_similar_skills

        query_vectors = model.encode(raw_items, normalize_embeddings=True)

        for i, (item, query_vec) in enumerate(zip(raw_items, query_vectors)):
            log_debug(f"[item {i}] raw='{item}'")
            results = search_fn(
                np.array(query_vec).astype("float32"),
                top_k=1,
                allowed_sources=allowed_sources,
            )

            if not results:
                log_debug(f"[item {i}] no results -> skip")
                continue

            best = results[0]
            similarity = float(best.get("Similarity", 0.0))
            meta_idx = best.get("Index")
            canonical = str(best.get("Name", best.get("Skill", ""))).strip()

            log_debug(f"[item {i}] best='{canonical}' sim={similarity:.4f}")

            if similarity < similarity_threshold or not canonical:
                continue

            meta: Dict = {}
            if meta_idx is not None and isinstance(metadata, pd.DataFrame):
                idx_int = int(meta_idx)
                if 0 <= idx_int < len(metadata):
                    meta = metadata.iloc[idx_int].to_dict()

            taxonomy_description = str(meta.get("description", meta.get("Description", "")))
            taxonomy_source = str(meta.get("taxonomy", ""))
            taxonomy_url = str(meta.get("source_url", meta.get("Source URL", meta.get("sourceUrl", ""))))

            mapped_items.append(canonical)
            raw_matched.append(item)
            taxonomy_descriptions.append(taxonomy_description)
            taxonomy_sources.append(taxonomy_source)
            taxonomy_urls.append(taxonomy_url)
            correlations.append(similarity)

        log_debug(f"[align] matched={len(mapped_items)} of {len(raw_items)}")

        if len(mapped_items) > top_k:
            combined = sorted(
                zip(correlations, raw_matched, mapped_items, taxonomy_descriptions, taxonomy_sources, taxonomy_urls),
                key=lambda x: x[0],
                reverse=True,
            )[:top_k]
            correlations, raw_matched, mapped_items, taxonomy_descriptions, taxonomy_sources, taxonomy_urls = map(
                list, zip(*combined)
            )

        return pd.DataFrame(
            {
                "Research ID": document_id,
                raw_col: raw_matched,
                taxonomy_col: mapped_items,
                "Taxonomy Description": taxonomy_descriptions,
                "Taxonomy Source": taxonomy_sources,
                "Source Url": taxonomy_urls,
                "Correlation Coefficient": correlations,
            }
        )

    def align_skills_to_taxonomy(
        self,
        raw_skills: List[str],
        document_id: str = "0",
        description: str = "",
        similarity_threshold: float = 0.20,
        top_k: int = DEFAULT_TOP_K,
        debug: bool = False,
        allowed_sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        return self.align(
            raw_items=raw_skills,
            document_id=document_id,
            description=description,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            raw_col="Raw Skill",
            taxonomy_col="Taxonomy Skill",
            allowed_sources=allowed_sources,
            debug=debug,
        )


SkillAlignmentService = AlignmentService
