import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Relative defaults (models/, data/) must not touch the repository."""
    monkeypatch.chdir(tmp_path)
