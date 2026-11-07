import hashlib
import io
from zipfile import ZipFile

import pytest
import requests

from laiser import data_access, taxonomy_assets
from laiser.exceptions import LAiSERError


@pytest.mark.parametrize("valid", [True, False])
def test_setup_checks_bundle_before_publishing(tmp_path, monkeypatch, valid):
    contents = b"expected"
    files = {
        "skills.index": {"size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()},
        "skills.json": {"size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()},
    }
    monkeypatch.setenv("LAISER_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(taxonomy_assets.MANIFEST, "bundles", {"skills": files})
    root = taxonomy_assets.cache_dir()
    root.mkdir()

    archive = io.BytesIO()
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("skills.index", contents)
        bundle.writestr("skills.json", contents if valid else b"modified")

    def response(*args, **kwargs):
        result = requests.Response()
        result.status_code = 200
        result.raw = io.BytesIO(archive.getvalue())
        return result

    monkeypatch.setattr(taxonomy_assets.requests, "get", response)
    with pytest.raises(LAiSERError, match="laiser setup"):
        taxonomy_assets.prepared_bundle("skills")

    if valid:
        taxonomy_assets._download_bundle("skills")
        directory = taxonomy_assets.prepared_bundle("skills")
        assert all((directory / name).read_bytes() == contents for name in files)
        monkeypatch.setattr(taxonomy_assets.requests, "get", lambda *args, **kwargs: pytest.fail("Unexpected download"))
        assert taxonomy_assets._download_bundle("skills") == "skills: already prepared"
    else:
        with pytest.raises(LAiSERError, match="checksum mismatch"):
            taxonomy_assets._download_bundle("skills")
        assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    ("manager", "kind"),
    [
        (data_access.FAISSIndexManager, "skills"),
        (data_access.KnowledgeFAISSIndexManager, "knowledge"),
        (data_access.TaskFAISSIndexManager, "tasks"),
    ],
)
def test_installed_index_managers_require_setup(tmp_path, monkeypatch, manager, kind):
    monkeypatch.setenv("LAISER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data_access, "is_source_checkout", lambda: False)
    with pytest.raises(LAiSERError, match=rf"{kind}\. Run 'laiser setup'"):
        manager(data_access.DataAccessLayer()).initialize_index()
