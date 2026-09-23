"""
Trains a couple of candidate models and picks the better one based on
cross-validated RMSE. Nothing exotic on purpose — with 150 rows of data,
a linear model or a small random forest is plenty, and it's easier to
explain "sleep matters most" to future-me than it would be to explain a
neural net's weights.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import load_features

CANDIDATES = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
}


def cross_validate_models(X, y, n_splits=5):
    """Returns {model_name: mean_rmse} using k-fold CV so the comparison
    isn't just luck of one particular train/test split."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for name, model in CANDIDATES.items():
        scores = cross_val_score(model, X, y, cv=kfold, scoring="neg_root_mean_squared_error")
        results[name] = -scores.mean()

    return results


def evaluate_on_holdout(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": round(float(mean_absolute_error(y_test, preds)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
        "r2": round(float(r2_score(y_test, preds)), 3),
    }


def train_and_select_best(data_path: str, model_out_path: str):
    X, y = load_features(data_path)

    cv_results = cross_validate_models(X, y)
    best_name = min(cv_results, key=cv_results.get)
    print("cross-validated RMSE (lower is better):")
    for name, rmse in sorted(cv_results.items(), key=lambda kv: kv[1]):
        marker = "  <- best" if name == best_name else ""
        print(f"  {name}: {rmse:.3f}{marker}")

    # fit the winner on a train/test split so we get an honest holdout score too
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    best_model = CANDIDATES[best_name]
    best_model.fit(X_train, y_train)
    holdout_metrics = evaluate_on_holdout(best_model, X_test, y_test)
    print(f"\nholdout metrics for {best_name}: {holdout_metrics}")

    # refit on all the data before saving, no reason to throw away 20% of
    # a dataset this small once we've already picked the model
    best_model.fit(X, y)
    joblib.dump({"model": best_model, "name": best_name, "features": list(X.columns)}, model_out_path)
    print(f"\nsaved {best_name} to {model_out_path}")

    return best_name, cv_results, holdout_metrics


if __name__ == "__main__":
    train_and_select_best("data/raw/daily_log.csv", "models/productivity_model.joblib")
