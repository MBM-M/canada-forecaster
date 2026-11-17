"""Service guarantees: idempotent releases, exactly-once publishing, the promotion gate, faithful briefs."""
import json

from service import brief as brief_mod
from service.cli import replay
from service.data import TARGETS, ready_at
from service.store import record_release
from service.worker import Settings, Worker
from tests.helpers import drain


def test_replay_is_idempotent(conn, raw):
    new, dup = replay(conn, raw=raw)
    assert new == 12 and dup == 0
    assert replay(conn, raw=raw) == (0, 12)
    assert conn.execute("SELECT count(*) FROM releases").fetchone()[0] == 12
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 12


def test_every_release_publishes_exactly_once(finished_db):
    c = finished_db
    assert c.execute("SELECT count(*) FROM releases").fetchone()[0] == 12
    assert c.execute("SELECT count(*) FROM runs WHERE status='done'").fetchone()[0] == 12
    assert c.execute("SELECT count(*) FROM nowcasts").fetchone()[0] == 12
    assert c.execute("SELECT count(*) FROM briefs WHERE passed=1").fetchone()[0] == 12
    dup = c.execute("SELECT run_id FROM nowcasts GROUP BY run_id HAVING count(*) > 1").fetchall()
    assert not dup
    # every nowcast covers every group of its target
    for target, t in TARGETS.items():
        counts = c.execute("SELECT count(*) FROM nowcast_values v JOIN nowcasts n ON n.id=v.nowcast_id "
                           "WHERE n.target=? GROUP BY n.id", (target,)).fetchall()
        assert counts and len({r[0] for r in counts}) == 1, t.name


def test_one_incumbent_per_target(finished_db):
    rows = finished_db.execute("SELECT target, count(*) FROM model_versions WHERE is_incumbent=1 GROUP BY target").fetchall()
    assert {r[0] for r in rows} == set(TARGETS) and all(r[1] == 1 for r in rows)


def test_gate_audit_no_promoted_model_is_worse(finished_db):
    """Success metric: no promoted model has a worse rolling error than the incumbent it replaced, or than naive."""
    promoted = finished_db.execute("SELECT * FROM model_versions WHERE decision='promoted'").fetchall()
    assert promoted, "the replay should promote at least once, otherwise this audit proves nothing"
    for m in promoted:
        assert m["rmspe_candidate"] <= m["rmspe_incumbent"] + 1e-9
        assert m["rmspe_candidate"] <= m["rmspe_naive"] + 1e-9
    for m in finished_db.execute("SELECT * FROM model_versions WHERE decision='rejected'"):
        assert m["rmspe_candidate"] > min(m["rmspe_incumbent"], m["rmspe_naive"]) - 1e-9


def test_gate_rejects_a_candidate_with_no_signal(conn, raw):
    """A candidate trained on shuffled targets must be rejected, and the incumbent must keep publishing."""
    for year in (2021, 2022, 2023):  # build a real incumbent first
        record_release(conn, "wages", year, ready_at("wages", year), "test")
    drain(conn, raw)
    incumbent = conn.execute("SELECT id FROM model_versions WHERE target='wages' AND is_incumbent=1").fetchone()[0]
    record_release(conn, "wages", 2024, ready_at("wages", 2024), "test")
    bad = json.dumps({"family": "rf", "n_estimators": 40, "max_depth": 6, "min_samples_leaf": 2, "seed": 1, "shuffle_seed": 7})
    drain(conn, raw, candidate_spec_json=bad)
    run_id = conn.execute("SELECT r.id FROM runs r JOIN releases l ON l.id=r.release_id WHERE l.ref_year=2024").fetchone()[0]
    m = conn.execute("SELECT * FROM model_versions WHERE created_from_run=?", (run_id,)).fetchone()
    assert m["decision"] == "rejected" and m["rmspe_candidate"] > m["rmspe_incumbent"]
    assert conn.execute("SELECT id FROM model_versions WHERE target='wages' AND is_incumbent=1").fetchone()[0] == incumbent
    nc = conn.execute("SELECT * FROM nowcasts WHERE run_id=?", (run_id,)).fetchone()
    assert nc["model_version_id"] == incumbent and "shuffled" not in nc["produced_by"]


def test_briefs_are_faithful(finished_db):
    c = finished_db
    for r in c.execute("SELECT n.*, b.text FROM nowcasts n JOIN briefs b ON b.nowcast_id=n.id"):
        values = [dict(v) for v in c.execute("SELECT * FROM nowcast_values WHERE nowcast_id=?", (r["id"],))]
        mv = dict(c.execute("SELECT * FROM model_versions WHERE created_from_run=?", (r["run_id"],)).fetchone())  # this run's gate
        mv["gate_years"] = json.loads(mv["gate_years"])
        # recompute the payload from the stored rows, independently of what the worker held in memory
        payload = brief_mod.build_payload(r["target"], dict(r), values, mv)
        ok, found, bad = brief_mod.check(r["text"], payload)
        assert ok and found and not bad
        # and every headline number matches the data row it came from
        changes = sorted(100 * (v["value"] - v["last_actual"]) / v["last_actual"] for v in values)
        assert brief_mod.pct(changes[0]) in r["text"] and brief_mod.pct(changes[-1]) in r["text"]


def test_check_catches_a_wrong_number(finished_db):
    c = finished_db
    r = c.execute("SELECT n.*, b.text FROM nowcasts n JOIN briefs b ON b.nowcast_id=n.id LIMIT 1").fetchone()
    values = [dict(v) for v in c.execute("SELECT * FROM nowcast_values WHERE nowcast_id=?", (r["id"],))]
    mv = dict(c.execute("SELECT * FROM model_versions WHERE created_from_run=?", (r["run_id"],)).fetchone())  # this run's gate
    mv["gate_years"] = json.loads(mv["gate_years"])
    payload = brief_mod.build_payload(r["target"], dict(r), values, mv)
    tampered = r["text"].replace(brief_mod.pct(payload["median_change"]), "+99.9", 1)
    ok, _, bad = brief_mod.check(tampered, payload)
    assert not ok and "+99.9" in bad


def test_retry_after_failure_reuses_the_gate_decision(conn, raw, monkeypatch):
    """A failure between the gate and the publish must not change the decision or publish twice."""
    replay(conn, min_year=2024, targets=["shelter"], raw=raw)
    w = Worker(conn, Settings(backoff_base_seconds=0), raw=raw)
    original = Worker._publish
    calls = {"n": 0}

    def flaky(self, run, mv):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated failure after the gate")
        return original(self, run, mv)

    monkeypatch.setattr(Worker, "_publish", flaky)
    assert w.run_once()  # fails, goes back to pending
    first = conn.execute("SELECT * FROM model_versions").fetchall()
    assert len(first) == 1 and conn.execute("SELECT count(*) FROM nowcasts").fetchone()[0] == 0
    assert w.run_once()  # succeeds
    assert [dict(r) for r in conn.execute("SELECT * FROM model_versions")] == [dict(r) for r in first]
    assert conn.execute("SELECT count(*) FROM nowcasts").fetchone()[0] == 1
    assert conn.execute("SELECT status FROM runs").fetchone()[0] == "done"


def test_a_worker_that_lost_its_lease_cannot_publish(conn, raw):
    replay(conn, min_year=2024, targets=["shelter"], raw=raw)
    slow = Worker(conn, Settings(), worker_id="slow", raw=raw)
    run = slow.claim()
    # the lease expires and another worker takes the run over
    conn.execute("UPDATE runs SET lease_until=0 WHERE id=?", (run.id,))
    fast = Worker(conn, Settings(), worker_id="fast", raw=raw)
    fast.run_once()
    assert conn.execute("SELECT count(*) FROM nowcasts").fetchone()[0] == 1
    slow.process(run)  # fenced out: must write nothing
    assert conn.execute("SELECT count(*) FROM nowcasts").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM model_versions").fetchone()[0] == 1
