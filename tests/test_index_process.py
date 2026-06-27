from pathlib import Path

import faiss
import numpy as np
import pytest
import pytest_check as check

from laiser.taxonomy import DataAccessLayer, FAISSIndexManager
from laiser.taxonomy import index as taxonomy_index_module


def get_asset_paths():
    laiser_dir = Path(__file__).parents[1] / "laiser"
    assets_dir = laiser_dir / "assets"
    return {
        "dir": assets_dir,
        "index": assets_dir / "skills_v04.index",
        "json": assets_dir / "skills_df.json",
        "npy": assets_dir / "skill_embeddings.npy",
    }


class _FakeEmbeddingModel:
    def encode(self, text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        rows = len(text)
        return np.ones((rows, 4), dtype="float32")


@pytest.mark.index
def test_initialize_index_full_flow_subtests(tmp_path, monkeypatch):
    da = DataAccessLayer()
    manager = FAISSIndexManager(da)

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    monkeypatch.setattr(taxonomy_index_module, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(da, "get_embedding_model", lambda: _FakeEmbeddingModel())

    source_assets = get_asset_paths()["dir"]
    (assets_dir / "faiss_skills.csv").write_bytes((source_assets / "faiss_skills.csv").read_bytes())

    paths = {
        "dir": assets_dir,
        "index": assets_dir / "skills_v04.index",
        "json": assets_dir / "skills_df.json",
        "npy": assets_dir / "skill_embeddings.npy",
    }
    artifacts = [paths["index"], paths["json"], paths["npy"]]

    # ---- Step 1: init using existing files (or build if missing) ----
    index1, metadata = manager.initialize_index(force_rebuild=False)
    check.is_not_none(index1, "step1: index1 should not be None")
    check.is_true(isinstance(index1, faiss.Index), "step1: index1 should be a FAISS Index")
    for p in artifacts:
        check.is_true(p.exists(), f"step1: artifact should exist: {p.name}")

    # ---- Step 2: init again without rebuild ----
    index2, metadata = manager.initialize_index(force_rebuild=False)
    check.is_not_none(index2, "step2: index2 should not be None")
    check.is_true(index2.ntotal > 0, "step2: index2.ntotal should be > 0")

    # ---- Step 3: force rebuild ----
    index3, metadata = manager.initialize_index(force_rebuild=True)
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
    index4, metadata = manager.initialize_index(force_rebuild=False)
    check.is_not_none(index4, "step5: index4 should not be None")
    check.is_true(index4.ntotal > 0, "step5: index4.ntotal should be > 0")
    for p in artifacts:
        check.is_true(p.exists(), f"step5: artifact should exist after rebuild: {p.name}")
