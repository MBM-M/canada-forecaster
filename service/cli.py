"""Command line: python -m service {replay,work,status,brief}."""
from __future__ import annotations

import argparse
import logging

from .data import TARGETS, load_raw, release_calendar
from .store import connect, db_path, record_release
from .worker import Worker

DEFAULT_MIN_YEAR = 2021  # the first release with enough history for a 3-year gate window


def replay(conn, min_year: int = DEFAULT_MIN_YEAR, targets: list[str] | None = None, raw=None) -> tuple[int, int]:
    """Record every simulated release from `min_year` on. Safe to run twice: the second run inserts nothing."""
    cal = release_calendar(raw or load_raw(), min_year, targets)
    new = dup = 0
    for row in cal.itertuples(index=False):
        if record_release(conn, row.target, row.ref_year, row.ready_at, "replay"):
            new += 1
        else:
            dup += 1
    return new, dup


def status(conn) -> str:
    lines = []
    runs = conn.execute("SELECT status, count(*) n FROM runs GROUP BY status ORDER BY status").fetchall()
    lines.append("runs: " + (", ".join(f"{r['status']}={r['n']}" for r in runs) or "none"))
    rows = conn.execute(
        """SELECT l.target, l.ref_year, r.status, m.decision, m.spec, m.rmspe_candidate, m.rmspe_incumbent, m.rmspe_naive
             FROM releases l JOIN runs r ON r.release_id = l.id LEFT JOIN model_versions m ON m.created_from_run = r.id
            ORDER BY l.target, l.ref_year""").fetchall()
    for r in rows:
        gate = "" if r["decision"] is None else (
            f" {r['decision']:9s} cand {r['rmspe_candidate']:.2f}% inc {r['rmspe_incumbent']:.2f}% naive {r['rmspe_naive']:.2f}%")
        lines.append(f"  {r['target']:10s} {r['ref_year']}  {r['status']:11s}{gate}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="record simulated releases (idempotent)")
    rp.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)
    rp.add_argument("--targets", nargs="*", choices=list(TARGETS))
    wk = sub.add_parser("work", help="process runs")
    wk.add_argument("--exit-when-idle", action="store_true")
    sub.add_parser("status", help="show releases, runs and gate decisions")
    bf = sub.add_parser("brief", help="print the latest brief per target")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    conn = connect()
    if args.cmd == "replay":
        new, dup = replay(conn, args.min_year, args.targets)
        print(f"replay: {new} new releases, {dup} already recorded ({db_path()})")
    elif args.cmd == "work":
        Worker(conn).run_forever(exit_when_idle=args.exit_when_idle)
    elif args.cmd == "status":
        print(status(conn))
    elif args.cmd == "brief":
        for r in conn.execute(
                """SELECT n.target, n.forecast_year, b.text FROM nowcasts n JOIN briefs b ON b.nowcast_id = n.id
                    WHERE n.id IN (SELECT max(id) FROM nowcasts GROUP BY target) ORDER BY n.target"""):
            print(f"[{r['target']} {r['forecast_year']}] {r['text']}\n")
    return 0

