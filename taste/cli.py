"""taste CLI — operator surface for taste-5.

    taste ingest <file.md>                — extract candidate layers
    taste standards [--state=..]          — list corpus
    taste confirm <std_id>                — activate a provisional layer
    taste packet --type=write_doc         — compile the packet for a task
    taste feedback --pick=A --why="..."   — log an owner decision
    taste pairs                           — contrast-pair evidence store
    taste distill [--since ISO]           — H3: distill hone episodes → layers
    taste queue                           — pending inferred items
    taste serve [--sock PATH]             — print the serve command
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api import create_app
from .distill import distill
from .feedback import record_feedback
from .ingest import ingest_docs
from .inject import compile_packet
from .models import FeedbackRequest, PacketRequest


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="taste")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest")
    pi.add_argument("file")
    ps = sub.add_parser("standards")
    ps.add_argument("--state", default=None)
    pc = sub.add_parser("confirm")
    pc.add_argument("std_id")
    pp = sub.add_parser("packet")
    pp.add_argument("--type", default="default")
    pf = sub.add_parser("feedback")
    pf.add_argument("--pick", required=True)
    pf.add_argument("--why", required=True)
    pf.add_argument("--chosen", default=None)
    pf.add_argument("--rejected", action="append", default=[])
    sub.add_parser("pairs")
    pd = sub.add_parser("distill")
    pd.add_argument("--since", default=None)
    pd.add_argument("--min-cluster", type=int, default=3)
    sub.add_parser("queue")
    pv = sub.add_parser("serve")
    pv.add_argument("--sock", default="./taste5.sock")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        print(f"  granian --interface asgi --uds {args.sock} taste.api:app")
        print(f"  # or: uvicorn --uds {args.sock} taste.api:app")
        return 0

    app = create_app()
    engine, store = app.state.engine, app.state.store

    if args.cmd == "ingest":
        text = Path(args.file).read_text()
        resp = ingest_docs("owner", [{"id": args.file, "text": text}])
        for c in resp.candidates:
            engine.propose(c)
        print(f"{len(resp.candidates)} candidate layers (provisional):")
        for r in resp.restatements:
            print(f"  [{r.layer.polarity.value}] {r.layer.directive}")
            for imp in r.implications[:2]:
                print(f"      → {imp}")

    elif args.cmd == "standards":
        for s in store.list_standards("owner"):
            kind = "·exemplar" if s.is_exemplar else ""
            print(f"[{s.state.value:11}] [{s.polarity.value:5}]{kind} "
                  f"{s.id}  {s.directive[:70]}")

    elif args.cmd == "confirm":
        print("activated" if engine.confirm(args.std_id) else "not found")

    elif args.cmd == "packet":
        pkt = compile_packet(engine, PacketRequest(
            task_ctx={"type": args.type}))
        print(pkt.model_dump_json(indent=2))

    elif args.cmd == "feedback":
        resp = record_feedback(engine, store, FeedbackRequest(
            pick=args.pick, rationale=args.why, chosen=args.chosen,
            rejected=args.rejected))
        print(f"decision {resp.record_id} logged; "
              f"{resp.extracted['candidates']} provisional layer(s), "
              f"{resp.extracted['pairs']} contrast pair(s)")

    elif args.cmd == "pairs":
        for pr in store.list_pairs("owner"):
            print(f"[{pr.source:8}] {pr.id}  WON: {pr.chosen[:50]!r}  "
                  f"LOST: {pr.rejected[:50]!r}  WHY: {pr.why[:40]}")

    elif args.cmd == "distill":
        r = distill(engine, store, "owner",
                    since=args.since, min_cluster=args.min_cluster)
        print(f"{r.events_seen} episodes → {r.proposed} provisional layers, "
              f"{r.pairs_written} contrast pairs")
        for k, n in sorted(r.clusters.items()):
            print(f"  {k}: {n}")

    elif args.cmd == "queue":
        for item in store.pending_inferred("owner"):
            print(f"[{item['kind']}] {item['id']}  {item['ts']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
