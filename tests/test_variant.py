"""taste-5 variant checks — the blueprint's own acceptance criteria.

- hone_events row written per /hone call
- escalating hone: strong draft → exactly 1 candidate generated
- weak draft → escalates to 3, iterate=True
- contrastive pairs appear in packets iff pair data exists
- feedback writes pairs when both sides resolve
- distiller produces provisional (never active) standards + pairs
- D4: judge called only on borderline; failure degrades to rules
"""

from taste.models import (
    Candidate,
    Category,
    HoneEvent,
    LayerState,
    Polarity,
    Provenance,
    Standard,
)


def _engine(app):
    return app.state.engine


def _store(app):
    return app.state.store


def _activate(app, **kw):
    s = Standard(profile_id="owner", **kw)
    app.state.engine.propose(s)
    app.state.engine.confirm(s.id)
    return s


# -------------------------------------------------------------- hone / F4

def test_hone_event_row_written_per_call(client, app):
    client.post("/hone", json={
        "profile_id": "owner",
        "segment_ctx": {"type": "write_doc", "segment": "lead"},
        "draft": "Some draft."})
    assert _store(app).count_hone_events("owner") == 1


def test_strong_draft_generates_exactly_one(app):
    """F4 proof: a first shot that clears the bar never pays for more."""
    _activate(app, category=Category.DOCTRINE, polarity=Polarity.DO,
              directive="write_doc opening shipped evidence capture matters",
              rationale="angle-first wins",
              provenance=Provenance.PERSONA)

    from taste.hone import hone_segment
    from taste.models import HoneRequest

    calls = []

    def strong_driver(draft, rules):
        def gen(n):
            calls.append(n)
            return ["shipped evidence capture matters"] * n
        return gen

    req = HoneRequest(profile_id="owner",
                      segment_ctx={"type": "write_doc", "segment": "opening"},
                      draft="x")
    resp = hone_segment(_engine(app), req, driver=strong_driver)

    assert calls == [1], "escalating hone must stop after a winning shot"
    assert resp.iterate is False
    events = _store(app).list_hone_events("owner")
    assert events[-1].n_generated == 1


def test_weak_draft_escalates_to_three(client, app):
    r = client.post("/hone", json={
        "profile_id": "owner",
        "segment_ctx": {"type": "write_doc", "segment": "lead"},
        "draft": "In today's world, things are hard."}).json()
    assert len(r["candidates"]) == 3
    assert r["iterate"] is True
    assert _store(app).list_hone_events("owner")[-1].n_generated == 3


# ---------------------------------------------------- contrast pairs / I3

def test_feedback_writes_contrast_pair(client, app):
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "pick": "B", "rationale": "verdict-first beats throat-clearing",
        "chosen": "write_doc verdict first always",
        "rejected": ["write_doc in today's world filler"]})
    assert r.json()["extracted"]["pairs"] == 1
    pair = _store(app).list_pairs("owner")[0]
    assert pair.chosen == "write_doc verdict first always"
    assert pair.source == "feedback"


def test_feedback_resolves_pick_label_via_candidates(client, app):
    client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "pick": "B", "rationale": "B had the angle",
        "candidates": [
            {"label": "A", "content": "write_doc bland opener"},
            {"label": "B", "content": "write_doc verdict leads"}],
        "rejected": ["A"]})
    pair = _store(app).list_pairs("owner")[0]
    assert pair.chosen == "write_doc verdict leads"
    assert pair.rejected == "write_doc bland opener"


def test_pair_appears_in_packet(client, app):
    client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "pick": "B", "rationale": "verdict-first wins",
        "chosen": "write_doc verdict leads",
        "rejected": ["write_doc bland opener"]})
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc"}}).json()
    contrast = [e for e in pkt["exemplars"] if e.get("kind") == "contrast"]
    assert contrast, "decision evidence must reach the packet"
    assert contrast[0]["rejected"] and contrast[0]["text"] and contrast[0]["why"]


def test_no_pairs_means_graceful_empty(client):
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc"}}).json()
    assert pkt["exemplars"] == []
    assert len(pkt["free_degrees"]) >= 1  # NEVER rule 6 holds regardless


def test_exemplar_standard_gets_retrieval_bias(app):
    from taste.embed import embed
    store = _store(app)
    plain = Standard(
        category=Category.DOCTRINE, polarity=Polarity.WHY,
        directive="write_doc keep the rhythm declarative always",
        rationale="rhythm", provenance=Provenance.PERSONA,
        variant_meta={})
    exemplar = Standard(
        category=Category.DOCTRINE, polarity=Polarity.WHY,
        directive="write_doc keep the rhythm declarative always",
        rationale="won before", provenance=Provenance.DECISION,
        variant_meta={"kind": "exemplar"})
    store.put_standard(plain, embed([plain.directive])[0])
    store.put_standard(exemplar, embed([exemplar.directive])[0])
    store.set_state(plain.id, LayerState.ACTIVE)
    store.set_state(exemplar.id, LayerState.ACTIVE)

    hits = _engine(app).retrieve("write_doc rhythm", "owner")
    by_id = {s.id: sim for s, sim in hits}
    assert by_id[exemplar.id] > by_id[plain.id], \
        "exemplar-kind standards must outrank identical non-exemplar text"


# -------------------------------------------------------------- distill / H3

def _seed_events(store, profile_id="owner"):
    """3 wins + 3 fails — the minimum clusters the distiller acts on."""
    for i in range(3):
        store.put_hone_event(HoneEvent(
            profile_id=profile_id, task_type="write_doc", segment="lead",
            draft="d",
            candidates=[Candidate(label="A", content="write_doc verdict "
                                  f"first draft {i}", score=0.8),
                        Candidate(label="B", content="write_doc slow "
                                  f"warmup {i}", score=0.2)],
            pick="A", score=0.8, n_generated=3))
        store.put_hone_event(HoneEvent(
            profile_id=profile_id, task_type="write_doc", segment="close",
            draft="d",
            candidates=[Candidate(label="A", content="write_doc in today's "
                                  f"world ending {i}", score=0.1),
                        Candidate(label="B", content="write_doc clean "
                                  f"ending {i}", score=0.2)],
            pick="B", score=0.2,
            violations=["violates std_x: 'in today's world'"],
            n_generated=3))


def test_distill_produces_provisional_only(client, app):
    _seed_events(_store(app))
    r = client.post("/distill", json={"profile_id": "owner"})
    body = r.json()
    assert body["events_seen"] == 6
    assert body["proposed"] >= 1
    assert body["pairs_written"] == 3

    rows = client.get("/standards").json()["standards"]
    assert all(s["state"] == "provisional" for s in rows), \
        "distilled standards must NEVER arrive active"
    exemplars = [s for s in rows
                 if s["variant_meta"].get("kind") == "exemplar"]
    antis = [s for s in rows if s["polarity"] == "never"]
    assert exemplars, "win cluster must produce an exemplar standard"
    assert antis, "fail cluster must produce an anti standard"
    assert exemplars[0]["provenance"] == "inference"


def test_distill_pairs_reach_packet(client, app):
    _seed_events(_store(app))
    client.post("/distill", json={"profile_id": "owner"})
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc"}}).json()
    assert any(e.get("kind") == "contrast" for e in pkt["exemplars"])


def test_distill_idempotent(client, app):
    _seed_events(_store(app))
    client.post("/distill", json={"profile_id": "owner"})
    again = client.post("/distill", json={"profile_id": "owner"}).json()
    assert again["proposed"] == 0, "re-distill must not re-propose"
    assert again["pairs_written"] == 0, "re-distill must not duplicate pairs"
    assert len(_store(app).list_pairs("owner")) == 3


# -------------------------------------------------------------- score / D4

def test_judge_not_called_without_model(client):
    r = client.post("/score", json={
        "profile_id": "owner", "task_ctx": {"type": "x"},
        "candidates": [{"label": "A", "content": "same words here"},
                       {"label": "B", "content": "same words here"}]})
    assert r.status_code == 200
    assert all(s["breakdown"]["borderline"] == 0.0
               for s in r.json()["scores"])


def test_judge_tiebreaks_borderline(client, app, monkeypatch):
    monkeypatch.setenv("TASTE_JUDGE_MODEL", "mock-judge")
    seen = {}

    def fake_chat(model, system, prompt):
        seen["called"] = True
        assert "write_doc" in prompt or "CANDIDATE" in prompt
        return {"scores": [{"ref": "A", "score": 0.9},
                           {"ref": "B", "score": 0.4}],
                "judge_conf": 0.95}

    monkeypatch.setattr("taste.llm.chat_json", fake_chat)

    r = client.post("/score", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "candidates": [{"label": "A", "content": "identical text"},
                       {"label": "B", "content": "identical text"}]})
    scores = {s["ref"]: s for s in r.json()["scores"]}
    assert seen["called"], "borderline Δ<0.15 must invoke the judge"
    assert scores["A"]["score"] == 0.9 and scores["B"]["score"] == 0.4
    assert scores["A"]["breakdown"]["borderline"] == 1.0


def test_judge_failure_degrades_to_rules(client, app, monkeypatch):
    monkeypatch.setenv("TASTE_JUDGE_MODEL", "mock-judge")

    def boom(model, system, prompt):
        raise RuntimeError("provider down")

    monkeypatch.setattr("taste.llm.chat_json", boom)
    r = client.post("/score", json={
        "profile_id": "owner", "task_ctx": {"type": "x"},
        "candidates": [{"label": "A", "content": "same words"},
                       {"label": "B", "content": "same words"}]})
    assert r.status_code == 200, "judge failure must degrade, not 500"
