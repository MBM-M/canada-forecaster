"""
train_models.py  –  Train models and generate 2025–2027 forecasts.

For each target (wages, shelter, employment):
  1. Train / test split at TEST_YEAR
  2. Fit LinearRegression (baseline) + RandomForestRegressor
  3. Evaluate on hold-out set → MAE, RMSE, R²
  4. Retrain on full data
  5. Autoregressive 3-year forecast with RF percentile prediction intervals
  6. Save: models/*.joblib, models/metrics_*.json,
           data/forecasts_*.csv, data/importance_*.csv
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FORECAST_YEARS = [2025, 2026, 2027]
TEST_YEAR = 2022  # train on < TEST_YEAR, evaluate on >= TEST_YEAR

# ── Feature column definitions ────────────────────────────────────────────────

WAGE_FEATURES = [
    "year", "province_code",
    "avg_hourly_wage_lag1", "avg_hourly_wage_lag2", "avg_hourly_wage_yoy",
    "shelter_index_lag1", "cpi_index_lag1",
    "covid",
]

SHELTER_FEATURES = [
    "year", "province_code",
    "shelter_index_lag1", "shelter_index_lag2", "shelter_index_yoy",
    "wage_index_lag1", "cpi_index_lag1",
    "covid",
]

EMPLOYMENT_FEATURES = [
    "year", "industry_code",
    "avg_employed_thousands_lag1", "avg_employed_thousands_lag2",
    "avg_employed_thousands_yoy",
    "covid",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rf_intervals(rf, X, alpha=0.10):
    """Prediction interval from RF individual tree outputs (percentile method)."""
    tree_preds = np.array([t.predict(X) for t in rf.estimators_])  # (n_trees, n_samples)
    lo = np.percentile(tree_preds, 100 * alpha / 2, axis=0)
    hi = np.percentile(tree_preds, 100 * (1 - alpha / 2), axis=0)
    return tree_preds.mean(axis=0), lo, hi


def _eval(label, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"    {label:18s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ── Autoregressive forecast helpers ──────────────────────────────────────────

def _init_state(target_name: str, last: pd.Series) -> dict:
    """Seed rolling state from the last known data row."""
    s = {}
    if target_name == "wages":
        s["t1"] = last["avg_hourly_wage"]          # 2024 value → lag1 for 2025
        s["t2"] = last["avg_hourly_wage_lag1"]      # 2023 value → lag2 for 2025
        s["yoy"] = last["avg_hourly_wage_yoy"]
        s["shelter_lag1"] = last["shelter_index"]   # held constant (last-known)
        s["cpi_lag1"] = last["cpi_index"]
    elif target_name == "shelter":
        s["t1"] = last["shelter_index"]
        s["t2"] = last["shelter_index_lag1"]
        s["yoy"] = last["shelter_index_yoy"]
        s["wage_lag1"] = last["wage_index"]         # held constant
        s["cpi_lag1"] = last["cpi_index"]
    elif target_name == "employment":
        s["t1"] = last["avg_employed_thousands"]
        s["t2"] = last["avg_employed_thousands_lag1"]
        s["yoy"] = last["avg_employed_thousands_yoy"]
    return s


def _build_x(target_name: str, year: int, state: dict, feature_cols: list) -> list:
    """Assemble the feature vector for one forecast step."""
    if target_name == "wages":
        vals = {
            "year": year,
            "province_code": state["group_code"],
            "avg_hourly_wage_lag1": state["t1"],
            "avg_hourly_wage_lag2": state["t2"],
            "avg_hourly_wage_yoy": state["yoy"],
            "shelter_index_lag1": state["shelter_lag1"],
            "cpi_index_lag1": state["cpi_lag1"],
            "covid": 0,
        }
    elif target_name == "shelter":
        vals = {
            "year": year,
            "province_code": state["group_code"],
            "shelter_index_lag1": state["t1"],
            "shelter_index_lag2": state["t2"],
            "shelter_index_yoy": state["yoy"],
            "wage_index_lag1": state["wage_lag1"],
            "cpi_index_lag1": state["cpi_lag1"],
            "covid": 0,
        }
    elif target_name == "employment":
        vals = {
            "year": year,
            "industry_code": state["group_code"],
            "avg_employed_thousands_lag1": state["t1"],
            "avg_employed_thousands_lag2": state["t2"],
            "avg_employed_thousands_yoy": state["yoy"],
            "covid": 0,
        }
    return [vals[f] for f in feature_cols]


def _update_state(state: dict, new_val: float):
    """Roll state forward one year (autoregressive update)."""
    old_t1 = state["t1"]
    state["t2"] = old_t1
    state["t1"] = new_val
    state["yoy"] = (new_val - old_t1) / old_t1 if old_t1 else 0.0
    # Exogenous lags (shelter, wage, cpi) held at last-known value


# ── Main training function ────────────────────────────────────────────────────

def train_and_forecast(
    target_name: str,
    feature_file: str,
    feature_cols: list,
    target_col: str,
    group_col: str,
):
    print(f"\n{'='*60}")
    print(f"  Target : {target_name.upper()}")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(DATA_DIR, feature_file))
    df = df.sort_values([group_col, "year"])

    train = df[df["year"] < TEST_YEAR]
    test = df[df["year"] >= TEST_YEAR]

    X_tr, y_tr = train[feature_cols].values, train[target_col].values
    X_te, y_te = test[feature_cols].values, test[target_col].values

    # ── Fit ──────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    print(f"  Train {len(y_tr)} rows / test {len(y_te)} rows")
    print("  Hold-out evaluation:")
    lr_metrics = _eval("LinearRegression", y_te, lr.predict(X_te))
    rf_metrics = _eval("RandomForest", y_te, rf.predict(X_te))

    # Retrain on full data for final forecasts
    X_all, y_all = df[feature_cols].values, df[target_col].values
    lr.fit(X_all, y_all)
    rf.fit(X_all, y_all)

    # ── Persist models + metrics ──────────────────────
    joblib.dump(lr, os.path.join(MODELS_DIR, f"lr_{target_name}.joblib"))
    joblib.dump(rf, os.path.join(MODELS_DIR, f"rf_{target_name}.joblib"))

    metrics = {"linear": lr_metrics, "random_forest": rf_metrics}
    with open(os.path.join(MODELS_DIR, f"metrics_{target_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Feature importance ────────────────────────────
    pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        os.path.join(DATA_DIR, f"importance_{target_name}.csv"), index=False
    )

    # ── Autoregressive forecasts ──────────────────────
    all_groups = sorted(df[group_col].unique())
    group_to_code = {g: i for i, g in enumerate(all_groups)}

    forecast_rows = []
    for group in all_groups:
        gdf = df[df[group_col] == group].sort_values("year")
        if gdf.empty:
            continue

        state = _init_state(target_name, gdf.iloc[-1])
        state["group_code"] = group_to_code[group]

        for yr in FORECAST_YEARS:
            x_vec = _build_x(target_name, yr, state, feature_cols)
            X_pred = np.array([x_vec])

            rf_mean, rf_lo, rf_hi = _rf_intervals(rf, X_pred)
            lr_pred = lr.predict(X_pred)[0]

            forecast_rows.append({
                group_col: group,
                "year": yr,
                "lr_forecast": round(float(lr_pred), 4),
                "rf_forecast": round(float(rf_mean[0]), 4),
                "rf_lower": round(float(rf_lo[0]), 4),
                "rf_upper": round(float(rf_hi[0]), 4),
            })

            _update_state(state, rf_mean[0])

    out = pd.DataFrame(forecast_rows)
    out.to_csv(os.path.join(DATA_DIR, f"forecasts_{target_name}.csv"), index=False)
    print(f"  Saved {len(out)} forecast rows -> data/forecasts_{target_name}.csv")

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from build_features import load_raw, make_wage_features, make_shelter_features, make_employment_features

    wages_cpi, employment = load_raw()

    make_wage_features(wages_cpi).to_csv(
        os.path.join(DATA_DIR, "features_wages.csv"), index=False)
    make_shelter_features(wages_cpi).to_csv(
        os.path.join(DATA_DIR, "features_shelter.csv"), index=False)
    make_employment_features(employment).to_csv(
        os.path.join(DATA_DIR, "features_employment.csv"), index=False)

    print("Features built.")

    train_and_forecast(
        "wages", "features_wages.csv",
        WAGE_FEATURES, "avg_hourly_wage", "geo",
    )
    train_and_forecast(
        "shelter", "features_shelter.csv",
        SHELTER_FEATURES, "shelter_index", "geo",
    )
    train_and_forecast(
        "employment", "features_employment.csv",
        EMPLOYMENT_FEATURES, "avg_employed_thousands", "industry",
    )

    print("\nAll done. Run `streamlit run app/streamlit_app.py` next.")
