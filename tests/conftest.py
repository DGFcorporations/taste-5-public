import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """Fresh app on a throwaway db per test — deterministic, offline-fast."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("TASTE5_DB", str(db))
    monkeypatch.setenv("TASTE_EMBED", "hash")
    monkeypatch.delenv("TASTE_JUDGE_MODEL", raising=False)
    from taste.embed import get_embedder
    get_embedder.cache_clear()
    from taste.api import create_app
    return create_app(db)


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
