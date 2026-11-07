"""Install and locate versioned taxonomy assets."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import requests

from laiser.exceptions import LAiSERError

MANIFEST = json.loads(Path(__file__).with_name("taxonomy-assets.json").read_text(encoding="utf-8"))
SOURCE_DATA = Path(__file__).with_name("public")
CHUNK_SIZE = 1024 * 1024


def is_source_checkout():
    return (SOURCE_DATA.parent.parent / "scripts" / "build_task_index.py").is_file()


def cache_dir():
    base = os.getenv("LAISER_CACHE_DIR") or Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "laiser"
    return Path(base).expanduser() / MANIFEST["version"]


def _bundle_ready(directory, files):
    try:
        return all((directory / name).stat().st_size == info["size"] for name, info in files.items())
    except FileNotFoundError:
        return False


def prepared_bundle(kind):
    directory = cache_dir() / kind
    if not _bundle_ready(directory, MANIFEST["bundles"][kind]):
        raise LAiSERError(f"Taxonomy data is missing or incomplete at {directory}. Run 'laiser setup'.")
    return directory


def _download_bundle(kind):
    root = cache_dir()
    destination = root / kind
    files = MANIFEST["bundles"][kind]
    if destination.exists():
        if _bundle_ready(destination, files):
            return f"{kind}: already prepared"
        # shortcut: if a prepared directory was modified externally, move it aside before repairing it.
        raise LAiSERError(f"Taxonomy data at {destination} is incomplete. Move it aside and rerun 'laiser setup'.")

    url = f"{MANIFEST['release_url']}/{MANIFEST['version']}/{kind}.zip"
    with TemporaryDirectory(dir=root) as temp:
        staging = Path(temp)
        archive = staging / "bundle.zip"
        unpacked = staging / kind
        unpacked.mkdir()

        # Allow ZIP overhead while bounding the size of an unexpected response.
        remaining = sum(info["size"] for info in files.values()) + CHUNK_SIZE
        with requests.get(url, stream=True, timeout=(10, 60)) as response, archive.open("wb") as output:
            response.raise_for_status()
            for chunk in response.iter_content(CHUNK_SIZE):
                remaining -= len(chunk)
                if remaining < 0:
                    raise LAiSERError(f"{kind}: taxonomy download exceeds the expected size")
                output.write(chunk)

        with ZipFile(archive) as bundle:
            if sorted(bundle.namelist()) != sorted(files):
                raise LAiSERError(f"{kind}: unexpected files in taxonomy archive")
            for name, info in files.items():
                if bundle.getinfo(name).file_size != info["size"]:
                    raise LAiSERError(f"{kind}: unexpected size for {name}")
                digest = hashlib.sha256()
                with bundle.open(name) as source, (unpacked / name).open("wb") as output:
                    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                        digest.update(chunk)
                        output.write(chunk)
                if digest.hexdigest() != info["sha256"]:
                    raise LAiSERError(f"{kind}: checksum mismatch for {name}")

        try:
            unpacked.rename(destination)
        except OSError:
            if not _bundle_ready(destination, files):
                raise

    return f"{kind}: downloaded and verified"


def main():
    parser = argparse.ArgumentParser(prog="laiser")
    parser.add_argument("command", choices=["setup"], help="Prepare taxonomy data and the embedding model")
    parser.parse_args()

    from laiser.data_access import load_embedding_model

    try:
        if is_source_checkout():
            print(f"Taxonomy data: using source checkout at {SOURCE_DATA}")
        else:
            cache_dir().mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=3) as executor:
                for status in executor.map(_download_bundle, MANIFEST["bundles"]):
                    print(status)
            print(f"Taxonomy data: {cache_dir()}")

        try:
            load_embedding_model(local_files_only=True)
            print("Embedding model: already prepared")
        except OSError:
            load_embedding_model(local_files_only=False)
            print("Embedding model: downloaded")
    except (LAiSERError, OSError, BadZipFile, requests.RequestException) as exc:
        parser.exit(1, f"LAiSER setup failed: {exc}\n")
