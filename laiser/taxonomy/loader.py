import io
import json
import logging
import os

import certifi
import faiss
import numpy as np
import pandas as pd
import requests
from huggingface_hub.utils import disable_progress_bars
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

from laiser.config import COMBINED_SKILLS_URL, DEFAULT_EMBEDDING_MODEL, ESCO_SKILLS_URL, OSN_SKILLS_URL
from laiser.exceptions import FAISSIndexError, LAiSERError

logger = logging.getLogger(__name__)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
disable_progress_bars()
transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


class DataAccessLayer:
    """Handle taxonomy loading and embedding model access."""

    def __init__(self):
        self.embedding_model = None
        self._esco_df = None
        self._osn_df = None
        self._combined_df = None

    def get_embedding_model(self) -> SentenceTransformer:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        return self.embedding_model

    def _fetch_csv_via_requests(self, url: str, timeout: int = 30) -> pd.DataFrame:
        try:
            resp = requests.get(url, timeout=timeout, verify=certifi.where())
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text))
        except requests.exceptions.RequestException as exc:
            raise LAiSERError(f"Failed to fetch CSV from {url}: {exc}")

    def load_esco_skills(self) -> pd.DataFrame:
        if self._esco_df is None:
            try:
                self._esco_df = self._fetch_csv_via_requests(ESCO_SKILLS_URL)
            except Exception as exc:
                raise LAiSERError(f"Failed to load ESCO skills data: {exc}")
        return self._esco_df

    def load_osn_skills(self) -> pd.DataFrame:
        if self._osn_df is None:
            try:
                self._osn_df = self._fetch_csv_via_requests(OSN_SKILLS_URL)
            except Exception as exc:
                raise LAiSERError(f"Failed to load OSN skills data: {exc}")
        return self._osn_df

    def load_skill_metadata(self, file_path: str) -> pd.DataFrame:
        if self._combined_df is None:
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Skills metadata file not found at {file_path}. Build or download the FAISS index first."
                    )
                self._combined_df = pd.read_json(file_path, orient="records")
            except Exception as exc:
                raise LAiSERError(f"Failed to load skills metadata: {exc}")
        return self._combined_df

    def load_combined_skills(self) -> pd.DataFrame:
        if self._combined_df is None:
            try:
                self._combined_df = pd.read_csv(COMBINED_SKILLS_URL)
            except Exception as exc:
                raise LAiSERError(f"Failed to load combined skills data: {exc}")
        return self._combined_df

    def build_faiss_index(self, text):
        try:
            model = self.get_embedding_model()
            embeddings = model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(np.asarray(embeddings, dtype="float32"))
            return index, embeddings
        except Exception as exc:
            raise FAISSIndexError(f"Failed to build FAISS index: {exc}")

    def save_faiss_index(self, index: faiss.IndexFlatIP, file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            faiss.write_index(index, file_path)
        except Exception as exc:
            raise FAISSIndexError(f"Failed to save FAISS index: {exc}")

    def save_skill_metadata_json(self, metadata_df: pd.DataFrame, file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            records = metadata_df.to_dict(orient="records")
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to save skill metadata JSON: {exc}")

    def load_faiss_index(self, file_path: str):
        try:
            if os.path.exists(file_path):
                return faiss.read_index(file_path)
            return None
        except Exception as exc:
            raise FAISSIndexError(f"Failed to load FAISS index: {exc}")

    def download_faiss_index(self, url: str, local_path: str) -> bool:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            if response.headers.get("Content-Type") != "application/octet-stream":
                raise ValueError(f"Unexpected content type: {response.headers.get('Content-Type')}")

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as handle:
                handle.write(response.content)
            return True
        except Exception as exc:
            print(f"Failed to download FAISS index: {exc}")
            return False
