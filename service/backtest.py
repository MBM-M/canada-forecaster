"""Rolling-origin backtests and the promotion gate.

Each test year is predicted by refitting on ONLY the years published by the day before it (the same simulated
publication lags as the live run), so a backtest never sees the future. Errors are percent errors, so wages
($/hr), shelter (index) and employment (thousands) can be compared with one metric: RMSPE, the root mean square
of the percent error pooled over groups and test years.

Gate rule: the candidate's RMSPE over the last N years must be at or below BOTH the incumbent's and the naive
baseline's (last year's value carried forward) on the same years. Candidate tuning uses an earlier window, never
the gate's years.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import TARGETS, ready_at, visible, years_of
from .model import DEFAULT_SPEC, Spec, build_features, candidate_specs, fit, predict

MIN_TRAIN_YEARS = 3  # years of training rows needed at a backtest origin
TOLERANCE = 1e-9


def first_evaluable_year(raw, target: str) -> int:
    return min(years_of(raw, target)) + 2 + MIN_TRAIN_YEARS  # two lag years, then the training years


def evaluable_years(raw, target: str, as_of: pd.Timestamp, n: int, skip_latest: int = 0) -> list[int]:
    """The last `n` years that are published by `as_of` and have enough history before them."""
    ys = [y for y in years_of(raw, target) if ready_at(target, y) <= pd.Timestamp(as_of) and y >= first_evaluable_year(raw, target)]
    if skip_latest:
        ys = ys[: len(ys) - skip_latest]
    return ys[-n:]


def backtest(raw, target: str, spec: Spec | None, years: list[int]) -> pd.DataFrame:
    """spec=None backtests the naive baseline."""
    t = TARGETS[target]
    frame = raw[t.frame]
    out = []
    for yr in years:
        vis = visible(raw, target, ready_at(target, yr - 1))
        train, rows = build_features(target, vis, yr)
        if spec is None:
            preds = rows["lag1"].to_numpy()
        else:
            preds, _, _ = predict(fit(spec, train, target), rows, target)
        actual = frame[frame["year"] == yr].set_index(t.group_col)[t.value_col]
        for grp, p in zip(rows[t.group_col], preds, strict=True):
            a = float(actual.get(grp, np.nan))
            out.append({"year": yr, "group": grp, "pred": float(p), "actual": a, "pct_error": 100.0 * (float(p) - a) / a})
    return pd.DataFrame(out)


def rmspe(frame: pd.DataFrame) -> float:
    f = frame.dropna(subset=["pct_error"])
    return float(np.sqrt(np.mean(np.square(f["pct_error"])))) if len(f) else float("nan")


@dataclass
class GateResult:
    decision: str  # promoted | rejected | retained
    reason: str
    years: list[int]
    rmspe_candidate: float
    rmspe_incumbent: float
    rmspe_naive: float
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def metrics(self) -> dict[str, float]:
        return {"rmspe_candidate": self.rmspe_candidate, "rmspe_incumbent": self.rmspe_incumbent, "rmspe_naive": self.rmspe_naive}


def evaluate(raw, target: str, as_of: pd.Timestamp, candidate: Spec, incumbent: Spec, n_years: int = 3) -> GateResult:
    years = evaluable_years(raw, target, as_of, n_years)
    if not years:
        raise ValueError("no evaluable years for the gate")
    cand = backtest(raw, target, candidate, years)
    inc = cand if incumbent == candidate else backtest(raw, target, incumbent, years)
    naive = backtest(raw, target, None, years)
    rc, ri, rn = rmspe(cand), rmspe(inc), rmspe(naive)
    if candidate == incumbent:  # the same settings refitted on newer data: nothing to promote
        decision, reason = "retained", f"candidate is identical to the incumbent ({candidate.name}); incumbent kept and refitted"
    elif rc <= ri + TOLERANCE and rc <= rn + TOLERANCE:
        decision, reason = "promoted", f"candidate RMSPE {rc:.3f}% <= incumbent {ri:.3f}% and naive {rn:.3f}%"
    elif rc > ri + TOLERANCE:
        decision, reason = "rejected", f"candidate RMSPE {rc:.3f}% > incumbent {ri:.3f}%"
    else:
        decision, reason = "rejected", f"candidate RMSPE {rc:.3f}% > naive baseline {rn:.3f}%"
    return GateResult(decision, reason, years, rc, ri, rn, {"candidate": cand, "incumbent": inc, "naive": naive})


def select_candidate(raw, target: str, as_of: pd.Timestamp, comparison_n: int = 3, tuning_n: int = 2) -> Spec:
    """Best spec on a tuning window that ends before the gate's window; the default if there is not enough history."""
    all_years = evaluable_years(raw, target, as_of, comparison_n + tuning_n)
    tuning = all_years[:-comparison_n] if len(all_years) > comparison_n else []
    if not tuning:
        return DEFAULT_SPEC
    scored = [(rmspe(backtest(raw, target, s, tuning)), i, s) for i, s in enumerate(candidate_specs())]
    return min(scored, key=lambda x: (x[0], x[1]))[2]
