"""Pinned data, the three forecast targets, and the point-in-time ("as of") view.

The snapshot holds the latest, revised values (StatCan open data, tables 14-10-0023-01, 18-10-0005-01 and
14-10-0064-01, processed to annual figures by the original forecaster). Publication is simulated with a fixed lag
per table, so a backtest never sees a year before it would have been published. No revision history is kept.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = Path(os.environ.get("SNAPSHOT_DIR", ROOT / "snapshot"))
RELEASE_CLOCK = time(8, 30)  # StatCan releases at 08:30 Eastern

PROVINCES = [
    "Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador",
    "Nova Scotia", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan",
]

# Days after the end of the reference year that an annual figure is assumed to appear. These are ASSUMPTIONS, not
# StatCan's published schedule; a target's inputs are ready when the last of its tables is out.
LAG_DAYS = {"14-10-0023-01": 40, "18-10-0005-01": 25, "14-10-0064-01": 55}


@dataclass(frozen=True)
class TargetDef:
    name: str
    frame: str  # key into load_raw()
    value_col: str
    group_col: str
    exog: tuple[str, ...]  # other columns, used lagged one year
    tables: tuple[str, ...]
    label: str
    unit: str


TARGETS: dict[str, TargetDef] = {
    t.name: t
    for t in [
        TargetDef("wages", "wages_cpi", "avg_hourly_wage", "geo", ("shelter_index", "cpi_index"),
                  ("14-10-0064-01", "18-10-0005-01"), "Average hourly wage", "$/hr"),
        TargetDef("shelter", "wages_cpi", "shelter_index", "geo", ("wage_index", "cpi_index"),
                  ("14-10-0064-01", "18-10-0005-01"), "Shelter CPI", "index, 2014=100"),
        TargetDef("employment", "employment", "avg_employed_thousands", "industry", (),
                  ("14-10-0023-01",), "Employment", "thousands"),
    ]
}


def clean_name(name: str) -> str:
    """Drop the NAICS code in brackets: 'Utilities [22]' -> 'Utilities'."""
    return name.split(" [")[0].strip()


def load_raw(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    d = Path(directory or SNAPSHOT_DIR)
    wages = pd.read_csv(d / "wages_vs_cpi.csv")
    return {
        "wages_cpi": wages[wages["geo"].isin(PROVINCES)].reset_index(drop=True),
        "employment": pd.read_csv(d / "employment_annual.csv"),
    }


def ready_at(target: str, year: int) -> pd.Timestamp:
    """When the inputs for `target` at reference `year` are all out (simulated)."""
    lag = max(LAG_DAYS[t] for t in TARGETS[target].tables)
    day = pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=lag)
    while day.weekday() >= 5:  # weekend releases roll to Monday
        day += pd.Timedelta(days=1)
    return pd.Timestamp(datetime.combine(day.date(), RELEASE_CLOCK))


def years_of(raw: dict[str, pd.DataFrame], target: str) -> list[int]:
    return sorted(int(y) for y in raw[TARGETS[target].frame]["year"].unique())


def visible(raw: dict[str, pd.DataFrame], target: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows of the target's frame that were published by `as_of`."""
    frame = raw[TARGETS[target].frame]
    ok = [y for y in years_of(raw, target) if ready_at(target, y) <= pd.Timestamp(as_of)]
    return frame[frame["year"].isin(ok)].copy()


def release_calendar(raw: dict[str, pd.DataFrame], min_year: int, targets: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for t in targets or list(TARGETS):
        for y in years_of(raw, t):
            if y >= min_year:
                rows.append({"target": t, "ref_year": y, "ready_at": ready_at(t, y)})
    return pd.DataFrame(rows).sort_values(["ready_at", "target"]).reset_index(drop=True)
