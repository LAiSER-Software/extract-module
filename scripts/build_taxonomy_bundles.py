"""Build GitHub Release archives from the committed taxonomy files."""

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "laiser" / "taxonomy-assets.json").read_text(encoding="utf-8"))
if len(sys.argv) != 3 or sys.argv[1] != manifest["version"]:
    raise SystemExit(f"Usage: {sys.argv[0]} {manifest['version']} OUTPUT_DIR")

output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
for kind, files in manifest["bundles"].items():
    with ZipFile(output / f"{kind}.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for name, expected in files.items():
            path = root / "laiser" / "public" / name
            digest = hashlib.sha256()
            with path.open("rb") as source, bundle.open(name, "w") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    target.write(chunk)
            if path.stat().st_size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise SystemExit(f"{name} does not match the manifest. Update the taxonomy version and checksums.")
