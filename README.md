# Canada Economic Forecaster

An interactive Streamlit dashboard that forecasts Canadian wages, shelter costs, and employment using Statistics Canada open data and machine learning (Linear Regression vs. Random Forest).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)

## Refresh service (branch `nowcast-service`)

A small service around the forecaster. Each time a new year of data is published it refits the models, **promotes a new
model only if it beats the current one on a rolling backtest**, and publishes one nowcast for the year in progress plus a
short brief whose numbers are checked against the data. It is the lightweight sibling of
[nowcast-as-a-service](https://github.com/MBM-M/nowcast-as-a-service), which does the same for a quarterly GDP model with
Postgres, workers, MLflow and Grafana. This one is one process and one SQLite file.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m service replay                  # record the simulated releases (safe to run twice)
python -m service work --exit-when-idle   # process them; try it with several workers at once
python -m service status                  # releases, gate decisions
python -m service brief                   # the latest brief for each target
streamlit run app/streamlit_app.py        # then open the "Refresh service" page
pytest                                    # 21 tests; `pytest -m slow` adds the crash test
```

**What it guarantees, and how that is tested**

| Guarantee | How |
|---|---|
| A release is recorded once, however often it is detected | `releases(target, ref_year)` is unique; replaying twice inserts nothing |
| Exactly one nowcast and one brief per release, even if workers crash | The nowcast, its values and its brief are written in one transaction guarded by `nowcasts(run_id)`; crashed runs are reclaimed after a lease expires; writes are fenced on the worker id. `test_chaos.py` runs two real worker processes that crash at random (including inside the publish transaction): 12 releases, 20 crashes, 12 nowcasts, no duplicates |
| No promoted model is worse than the incumbent | The gate promotes only if the candidate's error is at or below the incumbent's **and** a naive baseline's; a test audits every promotion, and another forces a no-signal candidate and checks it is rejected |
| A backtest never sees the future | Publication lags are simulated per table; a test destroys every value from the test year on and checks the prediction does not change |
| Every number in a brief is in the data | The brief is a template filled from stored values, and each number is checked against them before publishing (a brief that fails is never published); a test tampers with a figure and checks it is caught |

**Measured results** (one replay of the pinned data on a laptop, 12 releases in 28 s). The gate compares rolling percent
error (RMSPE) over the last 3 years:

| Target | Data through 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| Wages | bootstrap | rejected (3.23% vs incumbent 3.12%) | rejected (2.98% vs 2.64%) | retained |
| Shelter CPI | bootstrap | promoted (3.71% vs 4.62%) | retained | retained |
| Employment | bootstrap | retained | retained | retained |

`retained` means the candidate had the same settings as the incumbent, so there was nothing to promote.

**Two findings you should know about**
1. **The original features leak the target.** In `build_features.py`, `*_yoy` is this year's growth
   (`value / lag1 - 1`), which contains the value being predicted. On the wages target, the Linear Regression hold-out
   RMSE is 0.072 with that feature and 0.700 without it (the forest barely changes: 1.553 vs 1.539), so the dashboard's
   Linear Regression scores are optimistic. The service builds its own leak-free features (growth comes from the two
   previous years), and a test guards against the leak coming back. The original pipeline and app are unchanged on `main`.
2. **For employment, the models lose to carrying last year forward in every window** (8 to 14% error against 5 to 7%).
   The brief says so ("did not beat carrying last year forward, treat with caution") instead of presenting the
   forecast as reliable. One likely reason is that forests predicting levels cannot extrapolate a trend; predicting
   growth would be the first thing to try.

**Limits**
- **Replay only.** Releases come from a calendar simulated with fixed lags per table (assumptions, not StatCan's real
  schedule). A live poller for StatCan's Web Data Service is not built, because StatCan's server returned a
  bot-protection page (HTTP 403) to every request from my network while I was building this, and I did not work around it.
- **Very little data.** 11 annual points (2014 to 2024) for 10 provinces and 28 industries, so gate windows are 3
  years and a decision can flip on one noisy year. There is no revision history: values are the latest, and only the
  publication timing is simulated.
- It nowcasts the year in progress one step ahead; the three-year forecasts in the main app are separate.
- Briefs are templated text with a numeric check, not written by a language model.
- A model "version" is its settings; the model is refitted deterministically from the data, not stored as a file.
- One SQLite file means one machine. The page was tested headlessly, not looked at in a browser.

## Features

- **Three forecast targets** — average hourly wages, shelter CPI, and employment by province/industry
- **Two models compared** — Linear Regression (baseline) vs. Random Forest (200 trees)
- **Autoregressive rollout** — 3-year forecasts (2025–2027) with 90% prediction intervals from RF tree distribution
- **Interactive charts** — historical + forecast overlay, feature importance, model comparison metrics, multi-province/industry overview
- **First-run pipeline** — run feature engineering and model training directly from the app

## Project Structure

```
canada-forecaster/
├── app/
│   └── streamlit_app.py       # Streamlit dashboard
├── src/
│   ├── build_features.py      # Feature engineering (lags, YoY %, COVID dummy)
│   └── train_models.py        # Model training, evaluation, and forecast generation
├── data/                      # CSV inputs and generated outputs (gitignored)
├── models/                    # Trained .joblib models and metrics JSON (gitignored)
├── notebooks/
│   └── 01_modelling.ipynb     # Exploratory modelling notebook
└── requirements.txt
```

## Data Sources

Statistics Canada open data:
- **Table 14-10-0023-01** — Employment by industry (annual)
- **Table 18-10-0005-01** — CPI by province (wages, shelter, all-items)

Raw CSVs (`wages_vs_cpi.csv`, `employment_annual.csv`) must be placed in `data/` before running.

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/MBM-M/canada-forecaster.git
cd canada-forecaster

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/streamlit_app.py
```

On first launch, the app will prompt you to run the pipeline (feature engineering + model training, ~30 seconds). Click **"Run pipeline & generate forecasts"** and the dashboard will populate automatically.

## Models

| Target | Features | Train / Test split |
|---|---|---|
| Wages ($/hr) | Wage lags, shelter CPI lag, all-items CPI lag, province, year, COVID | Pre-2022 / 2022–2024 |
| Shelter CPI (index) | Shelter lags, wage index lag, CPI lag, province, year, COVID | Pre-2022 / 2022–2024 |
| Employment (thousands) | Employment lags, industry, year, COVID | Pre-2022 / 2022–2024 |

Random Forest: `n_estimators=200`, `max_depth=6`, `min_samples_leaf=2`. Prediction intervals use the 5th/95th percentile of individual tree outputs.

## Re-running the Pipeline

Use the **"Re-run pipeline"** button in the sidebar to retrain models from scratch. This deletes all generated CSVs and triggers the full feature engineering + training cycle.

Or run from the command line:

```bash
python src/train_models.py
```
