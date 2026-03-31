"""
Build the Knowledge FAISS index for LAiSER v0.5.

Sources:
  1. O*NET Knowledge (Knowledge.txt + Content Model Reference.txt)
     Downloaded from: https://www.onetcenter.org/database.html#individual-files
  2. ESCO Knowledge entries filtered from the ESCO skills CSV

Output:
  laiser/public/knowledge_taxonomy.csv  — unified knowledge taxonomy
  laiser/public/knowledge_v05.index     — FAISS IndexFlatIP
  laiser/public/knowledge_df.json       — metadata JSON

Usage:
  python scripts/build_knowledge_index.py
  python scripts/build_knowledge_index.py --force-rebuild
  python scripts/build_knowledge_index.py --onet-dir /path/to/onet_db_28_0/
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import certifi
import pandas as pd
import requests

# Ensure the repo root is on sys.path so `laiser` can be imported
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from laiser.data_access import DataAccessLayer, KnowledgeFAISSIndexManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote source URLs
# ---------------------------------------------------------------------------
# O*NET 28.0 individual files (plain-text tab-separated, UTF-8)
ONET_KNOWLEDGE_URL = "https://www.onetcenter.org/dl_files/database/db_28_0_text/Knowledge.txt"
ONET_CONTENT_MODEL_URL = "https://www.onetcenter.org/dl_files/database/db_28_0_text/Content%20Model%20Reference.txt"

# ESCO skills CSV from LAiSER datasets repo (already used for skills index)
ESCO_SKILLS_URL = (
    "https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master"
    "/taxonomies/ESCO_skills_Taxonomy.csv"
)

# Minimum O*NET importance score to include (0–5 scale; 3.0 = "important")
ONET_MIN_IMPORTANCE = 3.0

OUTPUT_DIR = REPO_ROOT / "laiser" / "public"


def _fetch(url: str, timeout: int = 60) -> bytes:
    logger.info(f"Fetching {url} …")
    resp = requests.get(url, timeout=timeout, verify=certifi.where())
    resp.raise_for_status()
    return resp.content


def load_onet_knowledge(onet_dir: Path = None) -> pd.DataFrame:
    """
    Load O*NET Knowledge entries.

    If onet_dir is provided, reads Knowledge.txt and Content Model Reference.txt
    from that local directory.  Otherwise downloads them from onetcenter.org.

    Returns DataFrame with columns: name, description, taxonomy
    """
    if onet_dir:
        knowledge_path = onet_dir / "Knowledge.txt"
        ref_path = onet_dir / "Content Model Reference.txt"
        if not knowledge_path.exists():
            raise FileNotFoundError(f"Knowledge.txt not found in {onet_dir}")
        knowledge_raw = pd.read_csv(str(knowledge_path), sep="\t", dtype=str)
        ref_raw = pd.read_csv(str(ref_path), sep="\t", dtype=str) if ref_path.exists() else None
    else:
        knowledge_raw = pd.read_csv(io.StringIO(_fetch(ONET_KNOWLEDGE_URL).decode("utf-8")), sep="\t", dtype=str)
        try:
            ref_raw = pd.read_csv(io.StringIO(_fetch(ONET_CONTENT_MODEL_URL).decode("utf-8")), sep="\t", dtype=str)
        except Exception:
            ref_raw = None

    # O*NET Knowledge.txt columns:
    # O*NET-SOC Code | Title | Element ID | Element Name | Scale ID | Data Value | N | SE | Lower CI Bound | Upper CI Bound | Recommend Suppress | Not Relevant
    knowledge_raw.columns = [c.strip() for c in knowledge_raw.columns]

    # Keep only importance scale rows above threshold
    if "Scale ID" in knowledge_raw.columns and "Data Value" in knowledge_raw.columns:
        knowledge_raw = knowledge_raw[knowledge_raw["Scale ID"].str.strip() == "IM"].copy()
        knowledge_raw["Data Value"] = pd.to_numeric(knowledge_raw["Data Value"], errors="coerce")
        knowledge_raw = knowledge_raw[knowledge_raw["Data Value"] >= ONET_MIN_IMPORTANCE]

    # Get unique knowledge elements with descriptions from Content Model Reference
    if ref_raw is not None:
        ref_raw.columns = [c.strip() for c in ref_raw.columns]
        # Filter to Knowledge domain elements
        if "Element ID" in ref_raw.columns and "Description" in ref_raw.columns:
            ref_knowledge = ref_raw[ref_raw["Element ID"].str.startswith("2.C", na=False)][
                ["Element ID", "Element Name", "Description"]
            ].drop_duplicates(subset=["Element ID"])
        else:
            ref_knowledge = None
    else:
        ref_knowledge = None

    # Unique knowledge elements from the importance rows
    if "Element Name" in knowledge_raw.columns:
        elements = knowledge_raw[["Element Name"]].drop_duplicates().rename(columns={"Element Name": "name"})
    else:
        logger.warning("O*NET Knowledge.txt missing 'Element Name' column — skipping O*NET source.")
        return pd.DataFrame(columns=["name", "description", "taxonomy"])

    elements["name"] = elements["name"].str.strip()

    # Merge in descriptions from Content Model Reference if available
    if ref_knowledge is not None and "Element Name" in ref_knowledge.columns:
        ref_knowledge = ref_knowledge.rename(columns={"Element Name": "name", "Description": "description"})
        ref_knowledge["name"] = ref_knowledge["name"].str.strip()
        elements = elements.merge(ref_knowledge[["name", "description"]], on="name", how="left")
    else:
        elements["description"] = ""

    elements["description"] = elements["description"].fillna("").str.strip()
    elements["taxonomy"] = "onet_knowledge"
    elements["field"] = "workforce"

    logger.info(f"O*NET Knowledge: {len(elements)} unique entries")
    return elements[["name", "description", "taxonomy", "field"]]


def load_esco_knowledge() -> pd.DataFrame:
    """
    Extract Knowledge entries from the ESCO skills CSV.

    ESCO marks knowledge entries with conceptType containing 'knowledge'.
    Returns DataFrame with columns: name, description, taxonomy, field
    """
    logger.info("Loading ESCO skills CSV for knowledge entries…")
    try:
        raw = pd.read_csv(io.StringIO(_fetch(ESCO_SKILLS_URL).decode("utf-8")), dtype=str)
    except Exception as e:
        logger.warning(f"Failed to fetch ESCO CSV: {e}. Skipping ESCO knowledge.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "field"])

    raw.columns = [c.strip() for c in raw.columns]

    # ESCO marks knowledge with conceptType = "KnowledgeSkillCompetence" or similar
    # Fall back to filtering by preferredLabel + description heuristics if no type column
    if "conceptType" in raw.columns:
        knowledge_df = raw[raw["conceptType"].str.lower().str.contains("knowledge", na=False)].copy()
    elif "skillType" in raw.columns:
        knowledge_df = raw[raw["skillType"].str.lower().str.contains("knowledge", na=False)].copy()
    else:
        logger.warning("ESCO CSV has no conceptType/skillType column — skipping ESCO knowledge.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "field"])

    if knowledge_df.empty:
        logger.warning("No ESCO knowledge entries found after filtering.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "field"])

    # Map to canonical columns
    name_col = next(
        (c for c in knowledge_df.columns if c.lower() in ("preferredlabel", "preferred_label", "label", "name")), None
    )
    desc_col = next((c for c in knowledge_df.columns if "description" in c.lower()), None)

    if not name_col:
        logger.warning("ESCO CSV missing name column — skipping.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "field"])

    result = pd.DataFrame()
    result["name"] = knowledge_df[name_col].str.strip()
    result["description"] = knowledge_df[desc_col].str.strip() if desc_col else ""
    result["description"] = result["description"].fillna("")
    result["taxonomy"] = "esco_knowledge"
    result["field"] = "workforce"

    result = result.dropna(subset=["name"]).drop_duplicates(subset=["name"]).reset_index(drop=True)
    logger.info(f"ESCO Knowledge: {len(result)} entries")
    return result[["name", "description", "taxonomy", "field"]]


def build_knowledge_taxonomy(onet_dir: Path = None) -> pd.DataFrame:
    """Combine O*NET and ESCO knowledge into a single deduplicated taxonomy."""
    onet_df = load_onet_knowledge(onet_dir)
    esco_df = load_esco_knowledge()

    combined = pd.concat([onet_df, esco_df], ignore_index=True)
    combined["name"] = combined["name"].str.strip()
    combined = combined[combined["name"] != ""].drop_duplicates(subset=["name"]).reset_index(drop=True)

    logger.info(f"Combined knowledge taxonomy: {len(combined)} unique entries (O*NET + ESCO)")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Build LAiSER v0.5 Knowledge FAISS index")
    parser.add_argument(
        "--onet-dir",
        type=Path,
        default=None,
        help="Path to local O*NET database directory (optional; downloads from onetcenter.org if not provided)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild index even if cached files already exist",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "knowledge_taxonomy.csv"

    # Build taxonomy CSV
    logger.info("Building knowledge taxonomy…")
    df = build_knowledge_taxonomy(onet_dir=args.onet_dir)

    if df.empty:
        logger.error("Knowledge taxonomy is empty — aborting.")
        sys.exit(1)

    df.to_csv(str(csv_path), index=False, encoding="utf-8")
    logger.info(f"Saved {len(df)} knowledge entries to {csv_path}")

    # Build FAISS index via KnowledgeFAISSIndexManager
    dal = DataAccessLayer()
    manager = KnowledgeFAISSIndexManager(dal)
    manager.initialize_index(force_rebuild=True)

    index_path = OUTPUT_DIR / "knowledge_v05.index"
    meta_path = OUTPUT_DIR / "knowledge_df.json"

    if index_path.exists() and meta_path.exists():
        logger.info(f"Knowledge FAISS index built successfully ({len(df)} entries).")
        logger.info(f"  Index:    {index_path}")
        logger.info(f"  Metadata: {meta_path}")
    else:
        logger.error("Index files not found after build — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
