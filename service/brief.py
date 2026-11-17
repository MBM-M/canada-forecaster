"""The brief: a short paragraph written only from stored numbers, then checked against them.

The text comes from a template, so it cannot invent a figure. `check` is still run on every brief, and a brief that
fails is never published: it scans the text for numbers and requires each one to be a value in the payload at the
precision shown (or the forecast year, data year or window length).
"""
from __future__ import annotations

import re
from typing import Any

from .data import TARGETS, clean_name

NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
GROUP_WORD = {"geo": "province", "industry": "industry"}


def pct(x: float) -> str:
    return f"{x:+.1f}"


def rm(x: float) -> str:
    return f"{x:.2f}"


def build_payload(target: str, nowcast: dict, values: list[dict], mv: dict) -> dict[str, Any]:
    t = TARGETS[target]
    rows = sorted(
        ({"group": clean_name(v["grp"]), "change": 100.0 * (v["value"] - v["last_actual"]) / v["last_actual"]} for v in values),
        key=lambda r: r["change"],
    )
    changes = [r["change"] for r in rows]
    n = len(changes)
    median = changes[n // 2] if n % 2 else (changes[n // 2 - 1] + changes[n // 2]) / 2
    return {
        "target": target,
        "label": t.label,
        "group_word": GROUP_WORD[t.group_col],
        "forecast_year": nowcast["forecast_year"],
        "data_through": nowcast["data_through"],
        "median_change": median,
        "low": rows[0],
        "high": rows[-1],
        "produced_by": nowcast["produced_by"],
        "decision": mv["decision"],
        "rmspe_candidate": mv["rmspe_candidate"],
        "rmspe_incumbent": mv["rmspe_incumbent"],
        "rmspe_naive": mv["rmspe_naive"],
        "n_years": len(mv["gate_years"]),
        "beats_naive": mv["rmspe_incumbent"] <= mv["rmspe_naive"] + 1e-9,
        "strip": sorted({r["group"] for r in rows} | {nowcast["produced_by"]}, key=len, reverse=True),
    }


def render(p: dict[str, Any]) -> str:
    head = (
        f"{p['label']} nowcast for {p['forecast_year']}, with {p['data_through']} data in: the median {p['group_word']} "
        f"is expected to change {pct(p['median_change'])}% from {p['data_through']}, ranging from {pct(p['low']['change'])}% "
        f"in {p['low']['group']} to {pct(p['high']['change'])}% in {p['high']['group']}."
    )
    if p["decision"] == "retained":
        gate = (
            f"The incumbent model was kept and refitted on the new data. Over the last {p['n_years']} years its error was "
            f"{rm(p['rmspe_incumbent'])}% against {rm(p['rmspe_naive'])}% for carrying last year forward (model: {p['produced_by']})."
        )
    else:
        verdict = {
            "promoted": "A new model beat the incumbent and was promoted.",
            "bootstrap": "This is the first model and was adopted without a comparison.",
            "rejected": "A new model was tested and rejected, so the incumbent produced this nowcast.",
        }[p["decision"]]
        gate = (
            f"{verdict} Over the last {p['n_years']} years the candidate's error was {rm(p['rmspe_candidate'])}% against "
            f"{rm(p['rmspe_incumbent'])}% for the incumbent and {rm(p['rmspe_naive'])}% for carrying last year forward "
            f"(model: {p['produced_by']})."
        )
    caution = "" if p["beats_naive"] else " The model did not beat carrying last year forward on this window, so treat it with caution."
    return f"{head} {gate}{caution}"


def allowed_numbers(p: dict[str, Any]) -> set[str]:
    return {
        str(p["forecast_year"]), str(p["data_through"]), str(p["n_years"]),
        pct(p["median_change"]), pct(p["low"]["change"]), pct(p["high"]["change"]),
        rm(p["rmspe_candidate"]), rm(p["rmspe_incumbent"]), rm(p["rmspe_naive"]),
    }


def check(text: str, p: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """(passed, numbers found, numbers that are not in the payload)."""
    scan = text
    for name in p["strip"]:  # names such as "rf[d6,leaf2]" contain digits that are not claims
        scan = scan.replace(name, " ")
    found = NUMBER.findall(scan)
    allowed = allowed_numbers(p)
    bad = [n for n in found if n not in allowed and n.lstrip("+") not in allowed]
    return not bad, found, bad
