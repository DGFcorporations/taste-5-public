"""taste-5 unit smoke — the loop end to end, in-process, offline.

Runs on the hashing embedder (TASTE_EMBED=hash via conftest) so it's
deterministic and fast; Qwen3 semantic quality is exercised separately.
"""

import pytest
from pydantic import ValidationError

from taste.models import (
    Category,
    DecisionRecord,
    LayerState,
    Polarity,
    Provenance,
    Standard,
)

PERSONA = """# Persona: Owner
- Never use generic openers like "In today's world".
- Lead with the angle, then the structure, then the build.
- Prefer short declarative sentences because they carry authority.
"""


def _confirm_all(client):
    rows = client.get("/standards").json()["standards"]
    for s in rows:
        if s["state"] == "provisional":
            client.post(f"/standards/{s['id']}/confirm")


def test_full_loop(client):
    # 1. ingest persona → provisional candidates
    r = client.post("/ingest", json={
        "profile_id": "owner",
        "docs": [{"id": "p", "text": PERSONA, "format": "md"}]})
    assert r.status_code == 200
    assert len(r.json()["candidates"]) >= 3

    # 2. owner confirms via comprehension gate
    _confirm_all(client)
    active = [s for s in client.get("/standards").json()["standards"]
              if s["state"] == "active"]
    assert len(active) >= 3

    # 3. packet compiles directives from active taste
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "ship notes"}}).json()
    never_texts = [d["text"] for d in pkt["directives"]
                   if d["polarity"] == "never"]
    assert any("generic openers" in t for t in never_texts)

    # 4. scorer catches the violation taste forbids
    r = client.post("/score", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "candidates": [
            {"label": "A", "content": "In today's world, shipping matters. We shipped evidence capture; here's what changed."},
            {"label": "B", "content": "Shipped: evidence capture. What changed, why it matters."}]})
    scores = {s["ref"]: s for s in r.json()["scores"]}
    assert scores["A"]["violations"], "generic opener must be caught"
    assert scores["A"]["score"] < scores["B"]["score"]

    # 5. hone picks the better candidate and logs the episode
    h = client.post("/hone", json={
        "profile_id": "owner",
        "segment_ctx": {"type": "write_doc", "segment": "lead"},
        "draft": "In today's world, things are hard. We shipped a fix."}).json()
    assert h["pick"] and h["candidates"]

    # 6. feedback lands + extracts a provisional why-layer (+ pair when
    #    both sides resolve)
    before = client.get("/standards").json()["count"]
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "pick": "B", "rationale": "verdict-first beats throat-clearing"})
    assert r.status_code == 200
    assert client.get("/standards").json()["count"] == before + 1


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_rationale_required_422(client, bad):
    """NEVER rule 1 — empty AND whitespace-only rationales are rejected
    with the contract §1 error body; no decision record is written."""
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {}, "pick": "A",
        "rationale": bad})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body and "code" in body["error"] \
        and "message" in body["error"], "contract §1 error shape"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_decision_record_rejects_blank_rationale(bad):
    with pytest.raises(ValidationError):
        DecisionRecord(pick="A", rationale=bad)


def test_never_rule_in_packet_without_retrieval_hit(client, app):
    """Guardrails are not relevance picks — an active never-rule reaches
    the packet even when retrieval never surfaces it."""
    s = Standard(category=Category.ANTI, polarity=Polarity.NEVER,
                 directive="Never use emoji in output",
                 rationale="owner hates emoji",
                 provenance=Provenance.PERSONA)
    app.state.engine.propose(s)
    app.state.engine.confirm(s.id)

    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "database_migration_planning"}}).json()
    texts = [d["text"] for d in pkt["directives"]
             if d["polarity"] == "never"]
    assert "Never use emoji in output" in texts, \
        "active never-rule must reach the packet without a retrieval hit"


def test_mirror_veto_long_rule_short_task(client, app):
    """Long rule + short task: ≥3 hits covering ≥half the task fires."""
    s = Standard(category=Category.ANTI, polarity=Polarity.NEVER,
                 directive="never alpha beta gamma delta epsilon zeta theta",
                 rationale="long forbidden list",
                 provenance=Provenance.PERSONA)
    app.state.engine.propose(s)
    app.state.engine.confirm(s.id)

    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"brief": "alpha beta gamma", "attended": True}}).json()
    assert r["vetoes_hit"], "3 hits covering the whole task must veto"


def test_mirror_veto_short_rule_fires(client, app):
    s = Standard(category=Category.ANTI, polarity=Polarity.NEVER,
                 directive="No emoji", rationale="owner hates emoji",
                 provenance=Provenance.PERSONA)
    app.state.engine.propose(s)
    app.state.engine.confirm(s.id)

    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"brief": "emoji everywhere", "attended": True}}).json()
    assert r["vetoes_hit"], "rule-side coverage must fire a short rule"


def test_mirror_veto_unrelated_task_clean(client, app):
    s = Standard(category=Category.ANTI, polarity=Polarity.NEVER,
                 directive="No emoji", rationale="owner hates emoji",
                 provenance=Provenance.PERSONA)
    app.state.engine.propose(s)
    app.state.engine.confirm(s.id)

    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"brief": "database migration planning",
                     "attended": True}}).json()
    assert r["vetoes_hit"] == []


def test_import_active_blank_rationale_lands_provisional(client):
    """NEVER rule 1 — an imported 'active' standard with a blank rationale
    is demoted to provisional instead of landing active with no 'because'."""
    r = client.post("/standards/import", json={
        "profile_id": "owner",
        "standards": [{
            "category": "doctrine", "polarity": "do",
            "directive": "lead with the verdict",
            "rationale": "   ", "provenance": "persona",
            "state": "active"}]})
    assert r.status_code == 200 and r.json()["imported"] == 1
    rows = client.get("/standards").json()["standards"]
    assert rows[0]["state"] == "provisional"


def test_supersede_missing_id_404(client):
    r = client.post("/standards/std_nonexistent/supersede", json={
        "category": "doctrine", "polarity": "do",
        "directive": "lead with the verdict", "rationale": "faster"})
    assert r.status_code == 404
    body = r.json()
    assert "error" in body and "code" in body["error"]


def test_standards_bad_state_422(client):
    r = client.get("/standards?state=bad")
    assert r.status_code == 422
    body = r.json()
    assert "error" in body and "code" in body["error"] \
        and "message" in body["error"]


def test_scorer_fires_on_short_never_rule(client, app):
    """'No emoji' — a sub-4-word never-rule — must still flag violations."""
    s = Standard(category=Category.ANTI, polarity=Polarity.NEVER,
                 directive="No emoji", rationale="owner hates emoji",
                 provenance=Provenance.PERSONA)
    app.state.engine.propose(s)
    app.state.engine.confirm(s.id)

    r = client.post("/score", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "candidates": [{"label": "A", "content": "an emoji here 😀"}]})
    scores = r.json()["scores"]
    assert scores[0]["violations"], \
        "short never-rule must produce a violation"


def test_provisional_layers_never_apply(client):
    client.post("/ingest", json={
        "profile_id": "owner",
        "docs": [{"id": "p", "text": PERSONA}]})
    pkt = client.post("/packet", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"}}).json()
    assert pkt["directives"] == [], \
        "provisional layers must not reach packets"


def test_supersede_keeps_history(app):
    from taste.embed import get_embedder
    get_embedder.cache_clear()
    s = Standard(category=Category.DOCTRINE, polarity=Polarity.DO,
                 directive="Lead with the angle", rationale="attention",
                 provenance=Provenance.PERSONA)
    app.state.engine.propose(s)
    app.state.engine.confirm(s.id)

    newer = Standard(category=Category.DOCTRINE, polarity=Polarity.DO,
                     directive="Lead with the verdict", rationale="faster",
                     provenance=Provenance.CORRECTION)
    app.state.engine.supersede(s.id, newer)

    old = app.state.store.get_standard(s.id)
    assert old.state == LayerState.SUPERSEDED  # NEVER deleted
    assert newer.supersedes == [s.id]
