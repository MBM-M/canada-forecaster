# Canada Economic Forecaster

An interactive Streamlit dashboard that forecasts Canadian wages, shelter costs, and employment using Statistics Canada open data and machine learning (Linear Regression vs. Random Forest).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)

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
