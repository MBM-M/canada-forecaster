"""
build_features.py  –  Feature engineering for the Canada Economic Forecaster.

Loads processed CSVs from data/ and outputs feature DataFrames for:
  - Wage forecasting        (by province)   → data/features_wages.csv
  - Shelter CPI forecasting (by province)   → data/features_shelter.csv
  - Employment forecasting  (by industry)   → data/features_employment.csv
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

PROVINCES = [
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan",
]

PROVINCE_CODES = {p: i for i, p in enumerate(sorted(PROVINCES))}


def load_raw():
    wages_cpi = pd.read_csv(os.path.join(DATA_DIR, "wages_vs_cpi.csv"))
    employment = pd.read_csv(os.path.join(DATA_DIR, "employment_annual.csv"))
    return wages_cpi, employment


def _add_lags(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    df = df.sort_values([group_col, "year"])
    df[f"{value_col}_lag1"] = df.groupby(group_col)[value_col].shift(1)
    df[f"{value_col}_lag2"] = df.groupby(group_col)[value_col].shift(2)
    df[f"{value_col}_yoy"] = df.groupby(group_col)[value_col].pct_change()
    return df


def make_wage_features(wages_cpi: pd.DataFrame) -> pd.DataFrame:
    df = wages_cpi[wages_cpi["geo"].isin(PROVINCES)].copy()
    df = _add_lags(df, "geo", "avg_hourly_wage")
    df = _add_lags(df, "geo", "shelter_index")
    df = _add_lags(df, "geo", "cpi_index")
    df["covid"] = (df["year"] >= 2020).astype(int)
    df["province_code"] = df["geo"].map(PROVINCE_CODES)
    df = df.dropna(subset=[
        "avg_hourly_wage_lag1", "avg_hourly_wage_lag2",
        "shelter_index_lag1", "cpi_index_lag1",
    ])
    return df.reset_index(drop=True)


def make_shelter_features(wages_cpi: pd.DataFrame) -> pd.DataFrame:
    df = wages_cpi[wages_cpi["geo"].isin(PROVINCES)].copy()
    df = _add_lags(df, "geo", "shelter_index")
    df = _add_lags(df, "geo", "wage_index")
    df = _add_lags(df, "geo", "cpi_index")
    df["covid"] = (df["year"] >= 2020).astype(int)
    df["province_code"] = df["geo"].map(PROVINCE_CODES)
    df = df.dropna(subset=[
        "shelter_index_lag1", "shelter_index_lag2",
        "wage_index_lag1", "cpi_index_lag1",
    ])
    return df.reset_index(drop=True)


def make_employment_features(employment: pd.DataFrame) -> pd.DataFrame:
    df = employment.copy()
    industry_codes = {ind: i for i, ind in enumerate(sorted(df["industry"].unique()))}
    df = _add_lags(df, "industry", "avg_employed_thousands")
    df["covid"] = (df["year"] >= 2020).astype(int)
    df["industry_code"] = df["industry"].map(industry_codes)
    df = df.dropna(subset=[
        "avg_employed_thousands_lag1", "avg_employed_thousands_lag2",
    ])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    wages_cpi, employment = load_raw()

    wage_df = make_wage_features(wages_cpi)
    shelter_df = make_shelter_features(wages_cpi)
    emp_df = make_employment_features(employment)

    wage_df.to_csv(os.path.join(DATA_DIR, "features_wages.csv"), index=False)
    shelter_df.to_csv(os.path.join(DATA_DIR, "features_shelter.csv"), index=False)
    emp_df.to_csv(os.path.join(DATA_DIR, "features_employment.csv"), index=False)

    print(f"  Wages features:      {wage_df.shape[0]:>4} rows × {wage_df.shape[1]} cols")
    print(f"  Shelter features:    {shelter_df.shape[0]:>4} rows × {shelter_df.shape[1]} cols")
    print(f"  Employment features: {emp_df.shape[0]:>4} rows × {emp_df.shape[1]} cols")
    print("Done — saved to data/")
