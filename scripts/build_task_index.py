"""
Build the Task FAISS index for LAiSER v0.5.

Sources:
  1. O*NET Task Statements (Task Statements.txt)
     Downloaded from: https://www.onetcenter.org/database.html#individual-files
  2. ESCO Occupation Tasks filtered from the ESCO occupations CSV

Output:
  laiser/public/task_taxonomy.csv   — unified task taxonomy
  laiser/public/tasks_v05.index     — FAISS IndexFlatIP
  laiser/public/tasks_df.json       — metadata JSON

Usage:
  python scripts/build_task_index.py
  python scripts/build_task_index.py --force-rebuild
  python scripts/build_task_index.py --onet-dir /path/to/onet_db_28_0/
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import certifi
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from laiser.data_access import DataAccessLayer, TaskFAISSIndexManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote source URLs
# ---------------------------------------------------------------------------
ONET_TASKS_URL = "https://www.onetcenter.org/dl_files/database/db_28_0_text/Task%20Statements.txt"
ONET_OCCUPATION_URL = "https://www.onetcenter.org/dl_files/database/db_28_0_text/Occupation%20Data.txt"

# ESCO occupations CSV — tasks are embedded as "essential" / "optional" skill statements
# We use the ESCO skills CSV which includes occupational context via altLabels
ESCO_OCCUPATIONS_URL = (
    "https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master"
    "/taxonomies/ESCO_skills_Taxonomy.csv"
)

# Minimum O*NET incumbents responding % to include a task (0–100; 50 = majority reported it)
ONET_MIN_INCUMBENTS_PCT = 50.0

OUTPUT_DIR = REPO_ROOT / "laiser" / "public"


def _fetch(url: str, timeout: int = 60) -> bytes:
    logger.info(f"Fetching {url} …")
    resp = requests.get(url, timeout=timeout, verify=certifi.where())
    resp.raise_for_status()
    return resp.content


def _extract_action_verb(task_text: str) -> str:
    """Return the first word of a task statement (the action verb)."""
    if not task_text:
        return ""
    return task_text.strip().split()[0].rstrip(".,;:")


def load_onet_tasks(onet_dir: Path = None) -> pd.DataFrame:
    """
    Load O*NET Task Statements.

    Filters to Core tasks with >= ONET_MIN_INCUMBENTS_PCT incumbents responding.
    Joins with Occupation Data for occupation title context.

    Returns DataFrame with columns: name, description, taxonomy, action_verb
    """
    if onet_dir:
        tasks_path = onet_dir / "Task Statements.txt"
        occ_path = onet_dir / "Occupation Data.txt"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Task Statements.txt not found in {onet_dir}")
        tasks_raw = pd.read_csv(str(tasks_path), sep="\t", dtype=str)
        occ_raw = pd.read_csv(str(occ_path), sep="\t", dtype=str) if occ_path.exists() else None
    else:
        tasks_raw = pd.read_csv(io.StringIO(_fetch(ONET_TASKS_URL).decode("utf-8")), sep="\t", dtype=str)
        try:
            occ_raw = pd.read_csv(io.StringIO(_fetch(ONET_OCCUPATION_URL).decode("utf-8")), sep="\t", dtype=str)
        except Exception:
            occ_raw = None

    tasks_raw.columns = [c.strip() for c in tasks_raw.columns]

    # O*NET Task Statements.txt columns:
    # O*NET-SOC Code | Title | Task ID | Task | Task Type | Incumbents Responding | Date | Domain Source
    if "Task Type" in tasks_raw.columns:
        tasks_raw = tasks_raw[tasks_raw["Task Type"].str.strip() == "Core"].copy()

    if "Incumbents Responding" in tasks_raw.columns:
        tasks_raw["Incumbents Responding"] = pd.to_numeric(tasks_raw["Incumbents Responding"], errors="coerce")
        tasks_raw = tasks_raw[tasks_raw["Incumbents Responding"] >= ONET_MIN_INCUMBENTS_PCT]

    if "Task" not in tasks_raw.columns:
        logger.warning("O*NET Task Statements.txt missing 'Task' column — skipping O*NET source.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "action_verb"])

    tasks_raw["Task"] = tasks_raw["Task"].str.strip()

    # Join with occupation titles for description context
    if occ_raw is not None:
        occ_raw.columns = [c.strip() for c in occ_raw.columns]
        title_col = next((c for c in occ_raw.columns if c.lower() in ("title", "occupation title", "occ title")), None)
        code_col = next((c for c in occ_raw.columns if "code" in c.lower() or "soc" in c.lower()), None)
        if title_col and code_col:
            occ_map = occ_raw.set_index(code_col)[title_col].to_dict()
            soc_col = next((c for c in tasks_raw.columns if "code" in c.lower() or "soc" in c.lower()), None)
            if soc_col:
                tasks_raw["_occ_title"] = tasks_raw[soc_col].map(occ_map).fillna("")
            else:
                tasks_raw["_occ_title"] = ""
        else:
            tasks_raw["_occ_title"] = ""
    else:
        # Use the Title column if present in Task Statements.txt
        title_col = next((c for c in tasks_raw.columns if c.lower() == "title"), None)
        tasks_raw["_occ_title"] = tasks_raw[title_col].fillna("").str.strip() if title_col else ""

    # Deduplicate by task text (same task appears across many occupations)
    tasks_dedup = tasks_raw.drop_duplicates(subset=["Task"]).reset_index(drop=True)

    result = pd.DataFrame()
    result["name"] = tasks_dedup["Task"]
    result["description"] = (tasks_dedup["_occ_title"] + " | " + tasks_dedup["Task"]).str.strip(" |")
    result["taxonomy"] = "onet_task"
    result["action_verb"] = tasks_dedup["Task"].apply(_extract_action_verb)

    result = result[result["name"] != ""].reset_index(drop=True)
    logger.info(f"O*NET Tasks: {len(result)} unique core task statements")
    return result[["name", "description", "taxonomy", "action_verb"]]


def load_esco_tasks() -> pd.DataFrame:
    """
    Extract task-like entries from the ESCO skills CSV.

    ESCO marks task competences with skillType containing 'task' or 'activity'.
    Returns DataFrame with columns: name, description, taxonomy, action_verb
    """
    logger.info("Loading ESCO skills CSV for task entries…")
    try:
        raw = pd.read_csv(io.StringIO(_fetch(ESCO_OCCUPATIONS_URL).decode("utf-8")), dtype=str)
    except Exception as e:
        logger.warning(f"Failed to fetch ESCO CSV: {e}. Skipping ESCO tasks.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "action_verb"])

    raw.columns = [c.strip() for c in raw.columns]

    # Filter for task/activity skill type
    type_col = next((c for c in raw.columns if c.lower() in ("skilltype", "concepttype", "skill_type")), None)
    if type_col:
        task_df = raw[raw[type_col].str.lower().str.contains("task|activit", na=False, regex=True)].copy()
    else:
        logger.warning("ESCO CSV has no skillType/conceptType column — skipping ESCO tasks.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "action_verb"])

    if task_df.empty:
        logger.warning("No ESCO task entries found after filtering.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "action_verb"])

    name_col = next(
        (c for c in task_df.columns if c.lower() in ("preferredlabel", "preferred_label", "label", "name")), None
    )
    desc_col = next((c for c in task_df.columns if "description" in c.lower()), None)

    if not name_col:
        logger.warning("ESCO CSV missing name column — skipping ESCO tasks.")
        return pd.DataFrame(columns=["name", "description", "taxonomy", "action_verb"])

    result = pd.DataFrame()
    result["name"] = task_df[name_col].str.strip()
    result["description"] = task_df[desc_col].str.strip() if desc_col else result["name"]
    result["description"] = result["description"].fillna(result["name"])
    result["taxonomy"] = "esco_task"
    result["action_verb"] = result["name"].apply(_extract_action_verb)

    result = result.dropna(subset=["name"]).drop_duplicates(subset=["name"]).reset_index(drop=True)
    logger.info(f"ESCO Tasks: {len(result)} entries")
    return result[["name", "description", "taxonomy", "action_verb"]]


def build_task_taxonomy(onet_dir: Path = None) -> pd.DataFrame:
    """Combine O*NET and ESCO tasks into a single deduplicated taxonomy."""
    onet_df = load_onet_tasks(onet_dir)
    esco_df = load_esco_tasks()

    combined = pd.concat([onet_df, esco_df], ignore_index=True)
    combined["name"] = combined["name"].str.strip()
    combined = combined[combined["name"] != ""].drop_duplicates(subset=["name"]).reset_index(drop=True)

    logger.info(f"Combined task taxonomy: {len(combined)} unique entries (O*NET + ESCO)")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Build LAiSER v0.5 Task FAISS index")
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
    csv_path = OUTPUT_DIR / "task_taxonomy.csv"

    logger.info("Building task taxonomy…")
    df = build_task_taxonomy(onet_dir=args.onet_dir)

    if df.empty:
        logger.error("Task taxonomy is empty — aborting.")
        sys.exit(1)

    df.to_csv(str(csv_path), index=False, encoding="utf-8")
    logger.info(f"Saved {len(df)} task entries to {csv_path}")

    # Build FAISS index via TaskFAISSIndexManager
    dal = DataAccessLayer()
    manager = TaskFAISSIndexManager(dal)
    manager.initialize_index(force_rebuild=True)

    index_path = OUTPUT_DIR / "tasks_v05.index"
    meta_path = OUTPUT_DIR / "tasks_df.json"

    if index_path.exists() and meta_path.exists():
        logger.info(f"Task FAISS index built successfully ({len(df)} entries).")
        logger.info(f"  Index:    {index_path}")
        logger.info(f"  Metadata: {meta_path}")
    else:
        logger.error("Index files not found after build — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
