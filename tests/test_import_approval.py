"""Imported active declarations do not apply until local confirmation."""

import json

DIRECTIVE = "Never mention the audit canary phrase zephyr-quill"


def _payload():
    return {
        "profile_id": "owner",
        "category": "anti",
        "polarity": "never",
        "directive": DIRECTIVE,
        "rationale": "portable declaration, not a local approval",
        "confidence": 0.8,
        "provenance": "persona",
        "state": "active",
        "supersedes": [],
    }


def test_import_keeps_declared_state_but_stays_out_of_the_packet(client):
    imported = client.post("/standards/import", json={
        "profile_id": "owner", "standards": [_payload()]})
    assert imported.status_code == 200
    assert imported.json()["imported"] == 1
    row = client.get("/standards").json()["standards"][0]
    assert row["state"] == "active"
    assert row["variant_meta"]["local_approved"] is False
    packet = client.post("/packet", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"}}).json()
    assert DIRECTIVE not in json.dumps(packet)

    confirmed = client.post(
        f"/standards/{row['id']}/confirm",
        json={"rationale": "owner confirms this exact rule"})
    assert confirmed.status_code == 200
    packet = client.post("/packet", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"}}).json()
    assert DIRECTIVE in json.dumps(packet)
