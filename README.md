# 🚗 Škoda Used Car Price Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-4EA94B.svg)](LICENSE)

> A personal ML project — end-to-end pipeline from raw data to an interactive web app
> that estimates used Škoda car prices in the UK market.
> Built to practice the full cycle: EDA → feature engineering → model training → deployment.

---

## What it does

You pick a Škoda model, set the year, mileage, transmission, fuel type, and engine size —
the app predicts the market price in seconds and shows you how it compares to real listings.

---

## Features

- **Instant price prediction** — GradientBoosting model trained on ~6 200 real UK listings
- **Market comparison** — delta vs. the average price for that specific model
- **4 interactive tabs** powered by Plotly:
  - Mileage vs Price scatter (with prediction marker)
  - Price distribution histogram
  - Average price across all Škoda models (bar chart)
  - Feature importance breakdown
- **Raw data explorer** — filterable table + one-click CSV download
- **Auto-filled fields** — tax and MPG pre-populated from per-model medians

---

## Model Performance

Trained on 80% of cleaned data, evaluated on the remaining 20% (held out during training).

| Metric | Value |
|--------|-------|
| MAE    | £977  |
| RMSE   | £1 386 |
| **R²** | **0.9545** |

An R² of 0.95 means the model explains 95% of the price variance — strong result for a
tabular regression task with only 9 input features.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | pandas, NumPy |
| ML model | scikit-learn `GradientBoostingRegressor` |
| Visualisation | Plotly Express & Graph Objects |
| Web app | Streamlit |
| Serialisation | pickle |
| Language | Python 3.10+ |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — 100 000 UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) |
| File | `skoda.csv` |
| Raw records | 6 267 |
| After cleaning | 6 263 |
| Features | `model`, `year`, `price`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize` |

Cleaning steps: dropped rows with nulls, removed prices outside £500–£60 000,
mileage > 200 000, and year < 2000.

---

## How It Works

```
Raw CSV  →  Clean  →  Feature engineering  →  Train/test split  →  GBR model
                                                                       ↓
                                                              model.pkl + encoders.pkl
                                                                       ↓
                                                             Streamlit UI (inference)
```

1. **Feature engineering** — `age = 2024 − year`; `LabelEncoder` for model, transmission, fuel type
2. **Model** — `GradientBoostingRegressor(n_estimators=300, max_depth=5, lr=0.1)`
3. **Inference** — artefacts loaded once at startup via `@st.cache_resource`

Top features by importance: **MPG (44%)**, Car Age (19%), Model (18%), Mileage (8%)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/denyskotcode/skoda_used_car_price_predictor.git
cd skoda_used_car_price_predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model  (creates model.pkl, label_encoders.pkl, feature_ranges.pkl)
python train_model.py

# 4. Run the app
streamlit run streamlit_app.py
# → open http://localhost:8501
```

> The `.pkl` files are already committed, so you can skip step 3 and run the app directly.

---

## Project Structure

```
skoda_used_car_price_predictor/
├── data/
│   └── skoda.csv               # 6 267 UK used Škoda listings
├── notebooks/
│   └── skoda_eda.ipynb         # Exploratory data analysis
├── .streamlit/
│   └── config.toml             # Škoda green theme
├── streamlit_app.py            # Main web application
├── train_model.py              # Training + artefact export
├── model.pkl                   # Trained GradientBoostingRegressor
├── label_encoders.pkl          # Fitted LabelEncoders
├── feature_ranges.pkl          # UI metadata (valid values, medians)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Why GradientBoosting?

For structured tabular data like this (mixed numeric + categorical, ~6k rows),
ensemble tree methods consistently outperform neural networks:
- No need for feature scaling
- Handles mixed feature types natively
- Interpretable via feature importance
- Strong baseline without hyperparameter tuning

---

## Possible Improvements

- [ ] Add `GridSearchCV` / `Optuna` for hyperparameter tuning
- [ ] Try `XGBoost` or `LightGBM` (faster, often ±same accuracy)
- [ ] Include more features — region, number of owners, service history
- [ ] Add SHAP values for per-prediction explainability
- [ ] Deploy to Streamlit Cloud with GitHub Actions CI

---

## License

[MIT](LICENSE) © 2026
