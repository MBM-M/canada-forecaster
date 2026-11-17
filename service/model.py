"""The forecaster's models (Linear Regression and Random Forest) with leak-free features.

The original `*_yoy` features are this year's growth, which contains the value being predicted. Here growth is
taken from the two previous years only (`growth_lag1 = lag1 / lag2 - 1`), so every feature is known before the
year being predicted.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .data import TARGETS, TargetDef

N_TREES = int(os.environ.get("SERVICE_RF_TREES", "200"))
COVID_FROM = 2020


@dataclass(frozen=True)
class Spec:
    family: str = "rf"  # "rf" | "linear"
    n_estimators: int = N_TREES
    max_depth: int | None = 6
    min_samples_leaf: int = 2
    seed: int = 42
    shuffle_seed: int | None = None  # gate test only: shuffles the training target, destroying the signal

    @property
    def name(self) -> str:
        base = "linear" if self.family == "linear" else f"rf[d{self.max_depth},leaf{self.min_samples_leaf}]"
        return base + (f"/shuffled{self.shuffle_seed}" if self.shuffle_seed is not None else "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Spec:
        return cls(**d)


DEFAULT_SPEC = Spec()


def candidate_specs() -> list[Spec]:
    return [Spec("linear"), Spec("rf", max_depth=3), Spec("rf", max_depth=4, min_samples_leaf=3), DEFAULT_SPEC]


def feature_names(t: TargetDef) -> list[str]:
    return ["year", "group_code", "lag1", "lag2", "growth_lag1", *[f"{c}_lag1" for c in t.exog], "covid"]


def build_features(target: str, visible_rows: pd.DataFrame, forecast_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, rows to predict) for `forecast_year`, using only `visible_rows`.

    Training rows are every earlier year with two lags; the rows to predict are one per group for `forecast_year`.
    """
    t = TARGETS[target]
    df = visible_rows.sort_values([t.group_col, "year"]).copy()
    if int(df["year"].max()) >= forecast_year:
        raise ValueError("visible data already contains the forecast year")
    groups = sorted(df[t.group_col].unique())
    stub = pd.DataFrame({t.group_col: groups, "year": forecast_year})
    df = pd.concat([df, stub], ignore_index=True).sort_values([t.group_col, "year"])
    g = df.groupby(t.group_col)
    df["lag1"] = g[t.value_col].shift(1)
    df["lag2"] = g[t.value_col].shift(2)
    df["growth_lag1"] = df["lag1"] / df["lag2"] - 1
    for c in t.exog:
        df[f"{c}_lag1"] = g[c].shift(1)
    df["covid"] = (df["year"] >= COVID_FROM).astype(int)
    df["group_code"] = df[t.group_col].map({grp: i for i, grp in enumerate(groups)})
    need = ["lag1", "lag2", *[f"{c}_lag1" for c in t.exog]]
    df = df.dropna(subset=need).reset_index(drop=True)
    return df[df["year"] < forecast_year], df[df["year"] == forecast_year]


def fit(spec: Spec, train: pd.DataFrame, target: str):
    t = TARGETS[target]
    X, y = train[feature_names(t)].to_numpy(), train[t.value_col].to_numpy()
    if spec.shuffle_seed is not None:
        y = np.random.default_rng(spec.shuffle_seed).permutation(y)
    if spec.family == "linear":
        return LinearRegression().fit(X, y)
    return RandomForestRegressor(
        n_estimators=spec.n_estimators, max_depth=spec.max_depth, min_samples_leaf=spec.min_samples_leaf,
        random_state=spec.seed, n_jobs=1,
    ).fit(X, y)


def predict(model, rows: pd.DataFrame, target: str, interval: float = 0.10):
    """(point, lo, hi). lo/hi are tree percentiles for a forest and NaN for a linear model."""
    X = rows[feature_names(TARGETS[target])].to_numpy()
    if isinstance(model, RandomForestRegressor):
        trees = np.array([e.predict(X) for e in model.estimators_])
        return trees.mean(axis=0), np.percentile(trees, 100 * interval / 2, axis=0), np.percentile(trees, 100 * (1 - interval / 2), axis=0)
    point = model.predict(X)
    return point, np.full(len(point), np.nan), np.full(len(point), np.nan)
