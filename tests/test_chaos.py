"""Chaos: real worker processes that crash at random (including inside the publish transaction), two at a time,
restarted until the queue is empty. Whatever happens, each release must end with exactly one published nowcast."""
import os
import subprocess
import sys
import time

import pytest

from service.cli import replay
from service.data import TARGETS, load_raw
from service.store import connect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_random_worker_crashes_never_duplicate_or_lose_a_publish(tmp_path):
    db = tmp_path / "chaos.db"
    conn = connect(db)
    replay(conn, raw=load_raw())
    releases = conn.execute("SELECT count(*) FROM releases").fetchone()[0]
    env = {**os.environ, "SERVICE_DB": str(db), "SERVICE_RF_TREES": "40", "CHAOS_CRASH_PROB": "0.12",
           "LEASE_SECONDS": "1", "MAX_ATTEMPTS": "200", "BACKOFF_BASE_SECONDS": "0"}
    crashes = rounds = 0
    deadline = time.time() + 600
    while time.time() < deadline:
        left = conn.execute("SELECT count(*) FROM runs WHERE status NOT IN ('done','dead_letter')").fetchone()[0]
        if left == 0:
            break
        rounds += 1
        procs = [subprocess.Popen([sys.executable, "-m", "service", "work", "--exit-when-idle"], cwd=ROOT,
                                  env={**env, "CHAOS_SEED": f"{rounds}-{i}"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 for i in range(2)]
        crashes += sum(p.wait(timeout=300) == 137 for p in procs)
    assert rounds < 200, "the queue never drained"
    assert crashes > 0, "the chaos switch did nothing, so this test proves nothing"

    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    assert q("SELECT count(*) FROM runs WHERE status='dead_letter'") == 0
    assert q("SELECT count(*) FROM runs WHERE status='done'") == releases
    assert q("SELECT count(*) FROM nowcasts") == releases
    assert q("SELECT count(*) FROM briefs WHERE passed=1") == releases
    assert q("SELECT count(*) FROM model_versions") == releases
    assert q("SELECT count(*) FROM (SELECT run_id FROM nowcasts GROUP BY run_id HAVING count(*) > 1)") == 0
    # no half-published nowcast: every nowcast has all of its values
    sizes = {r[0] for r in conn.execute("SELECT count(*) FROM nowcast_values GROUP BY nowcast_id")}
    assert len(sizes) <= len(TARGETS) and q("SELECT count(*) FROM nowcasts n WHERE NOT EXISTS "
                                           "(SELECT 1 FROM nowcast_values v WHERE v.nowcast_id = n.id)") == 0
    assert q("SELECT count(*) FROM (SELECT target FROM model_versions WHERE is_incumbent=1 GROUP BY target HAVING count(*) > 1)") == 0
    print(f"\nchaos: {releases} releases, {rounds} rounds, {crashes} crashes, "
          f"{q('SELECT sum(attempt) FROM runs')} attempts in total")
