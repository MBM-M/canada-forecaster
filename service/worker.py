"""Worker: claims runs from the SQLite queue and drives each through fitting, validating and publishing.

Delivery is at-least-once: a run can be retried after an error, or picked up again after its worker dies (the
lease expires). Publishing is still exactly-once, because the nowcast, its values and its brief are written in ONE
transaction guarded by nowcasts(run_id) UNIQUE, and every write is fenced on worker_id so a worker that lost its
lease cannot overwrite the new owner's progress. The gate decision is stored before publishing, so a retry reuses it.

Runs of the same target go in order (an earlier release must finish first, so the incumbent evolves
chronologically); different targets can run in parallel.
"""
from __future__ import annotations

import json
import logging
import os
import random
import socket
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

import pandas as pd

from . import brief as brief_mod
from .backtest import evaluate, select_candidate
from .data import TARGETS, load_raw, visible
from .model import DEFAULT_SPEC, Spec, build_features, fit, predict
from .store import connect, log_event, tx

log = logging.getLogger("service.worker")
ACTIVE = ("fitting", "validating", "publishing")


def _env_f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    lease_seconds: float = field(default_factory=lambda: _env_f("LEASE_SECONDS", 120))
    max_attempts: int = field(default_factory=lambda: int(_env_f("MAX_ATTEMPTS", 3)))
    backoff_base_seconds: float = field(default_factory=lambda: _env_f("BACKOFF_BASE_SECONDS", 5))
    comparison_years: int = field(default_factory=lambda: int(_env_f("GATE_YEARS", 3)))
    tuning_years: int = field(default_factory=lambda: int(_env_f("TUNING_YEARS", 2)))
    candidate_spec_json: str = field(default_factory=lambda: os.environ.get("CANDIDATE_SPEC_JSON", ""))
    chaos_crash_prob: float = field(default_factory=lambda: _env_f("CHAOS_CRASH_PROB", 0))
    chaos_seed: str = field(default_factory=lambda: os.environ.get("CHAOS_SEED", ""))


class LeaseLost(RuntimeError):
    """Another worker owns this run now; stop without writing."""


@dataclass
class Claimed:
    id: int
    target: str
    ref_year: int
    ready_at: pd.Timestamp
    attempt: int


CLAIM_SQL = f"""
SELECT r.id, r.status, l.target, l.ref_year, l.ready_at
  FROM runs r JOIN releases l ON l.id = r.release_id
 WHERE ((r.status = 'pending' AND r.next_attempt_at <= :now)
        OR (r.status IN {ACTIVE!r} AND r.lease_until < :now))
   AND NOT EXISTS (SELECT 1 FROM runs r2 JOIN releases l2 ON l2.id = r2.release_id
                    WHERE l2.target = l.target AND l2.ref_year < l.ref_year
                      AND r2.status NOT IN ('done', 'dead_letter'))
 ORDER BY l.ready_at, r.id
 LIMIT 1
"""


class Worker:
    def __init__(self, conn: sqlite3.Connection | None = None, settings: Settings | None = None,
                 worker_id: str | None = None, raw: dict | None = None) -> None:
        self.conn = conn or connect()
        self.s = settings or Settings()
        self.id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.raw = raw or load_raw()
        self._chaos = random.Random(f"{self.s.chaos_seed}-{self.id}" if self.s.chaos_seed else None)

    # ---------------------------------------------------------------- queue
    def claim(self) -> Claimed | None:
        now = time.time()
        with tx(self.conn):
            row = self.conn.execute(CLAIM_SQL, {"now": now}).fetchone()
            if row is None:
                return None
            attempt = self.conn.execute("SELECT attempt FROM runs WHERE id=?", (row["id"],)).fetchone()[0] + 1
            if attempt > self.s.max_attempts:
                self.conn.execute("UPDATE runs SET status='dead_letter', attempt=?, finished_at=?, lease_until=NULL, "
                                  "error=coalesce(error,'') || ' [max attempts exceeded]' WHERE id=?", (attempt, now, row["id"]))
                log_event(self.conn, row["id"], self.id, attempt, "dead_letter", {"reason": "max attempts exceeded"})
                log.error("ALERT run %s moved to dead_letter", row["id"])
                return None
            status = "fitting" if row["status"] == "pending" else row["status"]
            self.conn.execute("UPDATE runs SET status=?, worker_id=?, lease_until=?, attempt=?, "
                              "started_at=coalesce(started_at, ?) WHERE id=?",
                              (status, self.id, now + self.s.lease_seconds, attempt, now, row["id"]))
            log_event(self.conn, row["id"], self.id, attempt, "claimed", {"from": row["status"]})
        return Claimed(row["id"], row["target"], row["ref_year"], pd.Timestamp(row["ready_at"]), attempt)

    def _fence(self, run: Claimed) -> None:
        owner = self.conn.execute("SELECT worker_id FROM runs WHERE id=?", (run.id,)).fetchone()[0]
        if owner != self.id:
            raise LeaseLost(run.id)

    def _set_stage(self, run: Claimed, status: str) -> None:
        with tx(self.conn):
            self._fence(run)
            self.conn.execute("UPDATE runs SET status=?, lease_until=? WHERE id=?",
                              (status, time.time() + self.s.lease_seconds, run.id))
            log_event(self.conn, run.id, self.id, run.attempt, f"stage:{status}")

    def _chaos_point(self, label: str) -> None:
        if self.s.chaos_crash_prob > 0 and self._chaos.random() < self.s.chaos_crash_prob:
            log.warning("CHAOS: crashing at %s", label)
            os._exit(137)

    def fail(self, run: Claimed, stage: str, exc: Exception) -> None:
        dead = run.attempt >= self.s.max_attempts
        backoff = self.s.backoff_base_seconds * (2 ** (run.attempt - 1))
        with tx(self.conn):
            self.conn.execute(
                "UPDATE runs SET status=?, error=?, worker_id=NULL, lease_until=NULL, next_attempt_at=?, finished_at=? "
                "WHERE id=? AND worker_id=?",
                ("dead_letter" if dead else "pending", f"{stage}: {type(exc).__name__}: {exc}"[:2000],
                 time.time() + backoff, time.time() if dead else None, run.id, self.id))
            log_event(self.conn, run.id, self.id, run.attempt, "dead_letter" if dead else "retry",
                      {"stage": stage, "error": str(exc)[:500], "backoff_s": backoff})
        if dead:
            log.error("ALERT run %s moved to dead_letter: %s", run.id, exc)

    # ---------------------------------------------------------------- stages
    def _incumbent(self, target: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM model_versions WHERE target=? AND is_incumbent=1", (target,)).fetchone()

    def _candidate(self, run: Claimed) -> Spec:
        if self.s.candidate_spec_json:
            return Spec.from_dict(json.loads(self.s.candidate_spec_json))
        return select_candidate(self.raw, run.target, run.ready_at, self.s.comparison_years, self.s.tuning_years)

    def _validate(self, run: Claimed, candidate: Spec) -> sqlite3.Row:
        existing = self.conn.execute("SELECT * FROM model_versions WHERE created_from_run=?", (run.id,)).fetchone()
        if existing:  # retry after a crash: the decision is already durable
            return existing
        incumbent = self._incumbent(run.target)
        inc_spec = Spec.from_dict(json.loads(incumbent["spec"])) if incumbent else DEFAULT_SPEC
        gate = evaluate(self.raw, run.target, run.ready_at, candidate, inc_spec, self.s.comparison_years)
        decision, reason = gate.decision, gate.reason
        if incumbent is None:
            decision, reason = "bootstrap", f"no incumbent yet; {reason}; adopting the candidate"
        self._chaos_point("before-gate-write")
        with tx(self.conn):
            self._fence(run)
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO model_versions (created_from_run, target, spec, rmspe_candidate, rmspe_incumbent, "
                "rmspe_naive, gate_years, decision, reason, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run.id, run.target, json.dumps(candidate.to_dict()), gate.rmspe_candidate, gate.rmspe_incumbent,
                 gate.rmspe_naive, json.dumps(gate.years), decision, reason, time.time()))
            if cur.rowcount and decision in ("promoted", "bootstrap"):  # retained and rejected leave the incumbent alone
                self.conn.execute("UPDATE model_versions SET is_incumbent=0 WHERE target=? AND is_incumbent=1", (run.target,))
                self.conn.execute("UPDATE model_versions SET is_incumbent=1 WHERE id=?", (cur.lastrowid,))
            log_event(self.conn, run.id, self.id, run.attempt, "gate", {"decision": decision, "reason": reason, **gate.metrics()})
        return self.conn.execute("SELECT * FROM model_versions WHERE created_from_run=?", (run.id,)).fetchone()

    def _publish(self, run: Claimed, mv: sqlite3.Row) -> None:
        if self.conn.execute("SELECT 1 FROM nowcasts WHERE run_id=?", (run.id,)).fetchone():
            with tx(self.conn):  # crashed after the publish commit: only the bookkeeping is left
                self.conn.execute("UPDATE runs SET status='done', finished_at=coalesce(finished_at, ?), lease_until=NULL, "
                                  "error=NULL WHERE id=? AND worker_id=?", (time.time(), run.id, self.id))
            return
        incumbent = self._incumbent(run.target)
        spec = Spec.from_dict(json.loads(incumbent["spec"]))
        t = TARGETS[run.target]
        year = run.ref_year + 1
        vis = visible(self.raw, run.target, run.ready_at)
        train, rows = build_features(run.target, vis, year)
        point, lo, hi = predict(fit(spec, train, run.target), rows, run.target)
        last = vis[vis["year"] == run.ref_year].set_index(t.group_col)[t.value_col]
        values = [{"grp": g, "last_actual": float(last[g]), "value": float(p), "lo": None if pd.isna(a) else float(a),
                   "hi": None if pd.isna(b) else float(b)}
                  for g, p, a, b in zip(rows[t.group_col], point, lo, hi, strict=True)]
        nowcast = {"forecast_year": year, "data_through": run.ref_year, "produced_by": spec.name}
        payload = brief_mod.build_payload(run.target, nowcast, values, {**dict(mv), "gate_years": json.loads(mv["gate_years"])})
        text = brief_mod.render(payload)
        passed, numbers, bad = brief_mod.check(text, payload)
        if not passed:  # never publish a brief with a figure that is not in the data
            raise ValueError(f"brief failed the numeric check: {bad}")
        self._chaos_point("before-publish")
        with tx(self.conn):
            self._fence(run)
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO nowcasts (run_id, model_version_id, produced_by, target, data_through, forecast_year, "
                "published_at) VALUES (?,?,?,?,?,?,?)",
                (run.id, incumbent["id"], spec.name, run.target, run.ref_year, year, time.time()))
            if cur.rowcount:
                nid = cur.lastrowid
                self.conn.executemany(
                    "INSERT INTO nowcast_values (nowcast_id, grp, last_actual, value, lo, hi) VALUES (?,?,?,?,?,?)",
                    [(nid, v["grp"], v["last_actual"], v["value"], v["lo"], v["hi"]) for v in values])
                self.conn.execute("INSERT INTO briefs (nowcast_id, text, passed, numbers) VALUES (?,?,?,?)",
                                  (nid, text, 1, json.dumps(numbers)))
                self._chaos_point("inside-publish-tx")  # a crash here must roll the whole publish back
                log_event(self.conn, run.id, self.id, run.attempt, "published",
                          {"forecast_year": year, "model": spec.name, "decision": mv["decision"]})
            # same transaction as the nowcast: the run is done exactly when the nowcast exists
            self.conn.execute("UPDATE runs SET status='done', finished_at=?, lease_until=NULL, error=NULL WHERE id=?",
                              (time.time(), run.id))

    def process(self, run: Claimed) -> None:
        stage = "fitting"
        try:
            self._chaos_point("claimed")
            candidate = self._candidate(run)
            self._set_stage(run, "validating")
            stage = "validating"
            self._chaos_point("validating")
            mv = self._validate(run, candidate)
            self._set_stage(run, "publishing")
            stage = "publishing"
            self._chaos_point("publishing")
            self._publish(run, mv)
            self._chaos_point("published")
        except LeaseLost:
            log.warning("run %s: lease lost, another worker owns it", run.id)
        except Exception as exc:  # any stage error -> retry with backoff, then dead_letter
            log.exception("run %s failed in %s", run.id, stage)
            self.fail(run, stage, exc)

    def run_once(self) -> bool:
        run = self.claim()
        if run is None:
            return False
        self.process(run)
        return True

    def run_forever(self, idle_sleep: float = 1.0, exit_when_idle: bool = False) -> None:
        while True:
            if not self.run_once():
                if exit_when_idle and not self.pending_work():
                    return
                time.sleep(idle_sleep)

    def pending_work(self) -> bool:
        """True while any run could still be claimed later (pending, or active with a lease that may expire)."""
        return self.conn.execute(
            "SELECT 1 FROM runs WHERE status IN ('pending','fitting','validating','publishing') LIMIT 1").fetchone() is not None

