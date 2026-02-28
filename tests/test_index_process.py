import faiss
import pytest
import pytest_check as check
from pathlib import Path

from laiser.data_access import DataAccessLayer, FAISSIndexManager


def get_public_paths():
    laiser_dir = Path(__file__).parents[1] / "laiser"
    public_dir = laiser_dir / "public"
    return {
        "dir": public_dir,
        "index": public_dir / "skills_v04.index",
        "json": public_dir / "skills_df.json",
        "npy": public_dir / "skill_embeddings.npy",
    }

@pytest.mark.index
def test_initialize_index_full_flow_subtests():
    da = DataAccessLayer()
    manager = FAISSIndexManager(da)

    paths = get_public_paths()
    public_dir = paths["dir"]
    public_dir.mkdir(exist_ok=True)

    artifacts = [paths["index"], paths["json"], paths["npy"]]

    # ---- Step 1: init using existing files (or build if missing) ----
    index1,metadata = manager.initialize_index(force_rebuild=False)
    check.is_not_none(index1, "step1: index1 should not be None")
    check.is_true(isinstance(index1, faiss.Index), "step1: index1 should be a FAISS Index")
    for p in artifacts:
        check.is_true(p.exists(), f"step1: artifact should exist: {p.name}")

    # ---- Step 2: init again without rebuild ----
    index2,metadata = manager.initialize_index(force_rebuild=False)
    check.is_not_none(index2, "step2: index2 should not be None")
    check.is_true(index2.ntotal > 0, "step2: index2.ntotal should be > 0")

    # ---- Step 3: force rebuild ----
    index3,metadata = manager.initialize_index(force_rebuild=True)
    check.is_not_none(index3, "step3: index3 should not be None")
    check.is_true(index3.ntotal > 0, "step3: index3.ntotal should be > 0")
    for p in artifacts:
        check.is_true(p.exists(), f"step3: artifact should exist after rebuild: {p.name}")

    # ---- Step 4: delete artifacts ----
    for p in artifacts:
        if p.exists():
            p.unlink()
    for p in artifacts:
        check.is_false(p.exists(), f"step4: artifact should be deleted: {p.name}")

    # ---- Step 5: init again (should rebuild because files missing) ----
    index4,metadata = manager.initialize_index(force_rebuild=False)
    check.is_not_none(index4, "step5: index4 should not be None")
    check.is_true(index4.ntotal > 0, "step5: index4.ntotal should be > 0")
    for p in artifacts:
        check.is_true(p.exists(), f"step5: artifact should exist after rebuild: {p.name}")

