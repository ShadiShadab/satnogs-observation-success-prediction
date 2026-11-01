"""Robustness check for the SatNOGS observation-success model.

Added during the portfolio review (November 2026). It does not change the
original results. It answers three questions about them:

1. Reproducibility: does the saved LightGBM model reproduce its reported
   test score on the saved test split?
2. Feature dependence: how much does performance depend on `archived`, a
   field that SatNOGS sets only after an observation's recordings are moved
   to the Internet Archive (so it is not known at scheduling time)?
3. Forward-looking performance: how does a model trained on 2021-2024 do on
   2025 observations? The original evaluation used a random split.

Usage (from the repository root):
    1. Extract 4_data_analysis/ml_results/advanced/fixed_data.rar so that
       4_data_analysis/ml_results/advanced/fixed_data/ exists.
    2. pip install -r requirements.txt
    3. python 4_data_analysis/04_robustness_check.py
"""

import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ADVANCED = ROOT / "4_data_analysis" / "ml_results" / "advanced"
DATA_DIR = ADVANCED / "fixed_data"
OUT_DIR = ROOT / "4_data_analysis" / "ml_results" / "robustness"
SEED = 42


def load_split(name):
    x = pd.read_csv(DATA_DIR / f"X_{name}_engineered.csv")
    y = np.load(DATA_DIR / f"y_{name}.npy", allow_pickle=True).astype(int)
    return x, y


def fit_and_score(x_train, y_train, x_test, y_test):
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.1, num_leaves=63, random_state=SEED, verbose=-1
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, proba > 0.5)), 4),
    }


def main():
    if not DATA_DIR.exists():
        sys.exit(f"Missing {DATA_DIR}. Extract fixed_data.rar first (see the docstring).")

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")
    results = {}

    # 1. Reproduce the saved LightGBM model on the saved test split
    with open(ADVANCED / "lightgbm_model.pkl", "rb") as f:
        saved = pickle.load(f)
    proba = saved.predict_proba(x_test)[:, 1]
    results["saved_lightgbm_on_test"] = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, proba > 0.5)), 4),
    }

    # 2. Feature ablation on the original random split
    station_cols = ["station_lat", "station_lng", "station_alt", "abs_latitude", "horizon"]
    ablations = {
        "all_features": [],
        "without_archived": ["archived"],
        "without_station_location_features": station_cols,
    }
    results["random_split_ablation"] = {
        name: fit_and_score(x_train.drop(columns=drop), y_train, x_test.drop(columns=drop), y_test)
        for name, drop in ablations.items()
    }

    # 3. Time-based split: train on 2021-2024, test on 2025 (`year` removed)
    x_all = pd.concat([x_train, x_val, x_test], ignore_index=True)
    y_all = np.concatenate([y_train, y_val, y_test])
    past, future = x_all["year"] < 2025, x_all["year"] == 2025
    results["time_split_train_2021_2024_test_2025"] = {
        "train_rows": int(past.sum()),
        "test_rows": int(future.sum()),
    }
    for name, drop in {"all_features": ["year"], "without_archived": ["year", "archived"]}.items():
        results["time_split_train_2021_2024_test_2025"][name] = fit_and_score(
            x_all[past].drop(columns=drop), y_all[past], x_all[future].drop(columns=drop), y_all[future]
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "robustness_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
