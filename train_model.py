"""
train_model.py
--------------
Train a GradientBoostingRegressor on the Skoda used car dataset and
persist the model, label encoders, and feature metadata to disk.

Usage:
    python train_model.py
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Skoda Used Car Price Predictor — Model Training")
print("=" * 60)

df = pd.read_csv("data/skoda.csv")
print(f"\n[INFO] Loaded {len(df):,} rows from data/skoda.csv")
print(df.head(3).to_string())


# ---------------------------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------------------------
initial_size = len(df)

df = df.dropna()
df = df[df["price"].between(500, 60_000)]
df = df[df["mileage"] <= 200_000]
df = df[df["year"] >= 2000]

print(f"\n[INFO] After cleaning: {len(df):,} rows (removed {initial_size - len(df):,})")


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
df["age"] = 2024 - df["year"]

# Label encode categorical columns
cat_cols = ["model", "transmission", "fuelType"]
label_encoders: dict[str, LabelEncoder] = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

FEATURES = ["age", "mileage", "engineSize", "tax", "mpg",
            "model_enc", "transmission_enc", "fuelType_enc"]
TARGET = "price"

X = df[FEATURES].values
y = df[TARGET].values


# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n[INFO] Training set: {len(X_train):,} samples")
print(f"[INFO] Test set    : {len(X_test):,} samples")


# ---------------------------------------------------------------------------
# 5. Train model
# ---------------------------------------------------------------------------
print("\n[INFO] Training GradientBoostingRegressor …")
model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)
print("[INFO] Training complete.")


# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 40)
print("  Model Performance (test set)")
print("=" * 40)
print(f"  MAE  : £{mae:,.0f}")
print(f"  RMSE : £{rmse:,.0f}")
print(f"  R²   :  {r2:.4f}")
print("=" * 40)

# Feature importance summary
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n[INFO] Feature importances:")
for feat, imp in fi.items():
    print(f"  {feat:<20} {imp:.4f}")


# ---------------------------------------------------------------------------
# 7. Build feature ranges for UI
# ---------------------------------------------------------------------------
tax_median  = df.groupby("model")["tax"].median().to_dict()
mpg_median  = df.groupby("model")["mpg"].median().to_dict()

feature_ranges = {
    "models":             sorted(df["model"].unique().tolist()),
    "transmissions":      sorted(df["transmission"].unique().tolist()),
    "fuel_types":         sorted(df["fuelType"].unique().tolist()),
    "year_range":         (int(df["year"].min()), int(df["year"].max())),
    "mileage_range":      (int(df["mileage"].min()), int(df["mileage"].max())),
    "engine_sizes":       sorted(df["engineSize"].unique().tolist()),
    "tax_median_by_model": tax_median,
    "mpg_median_by_model": mpg_median,
}


# ---------------------------------------------------------------------------
# 8. Save artefacts
# ---------------------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("feature_ranges.pkl", "wb") as f:
    pickle.dump(feature_ranges, f)

print("\n[INFO] Saved artefacts:")
print("  model.pkl")
print("  label_encoders.pkl")
print("  feature_ranges.pkl")
print("\n[DONE] Training pipeline complete.")
