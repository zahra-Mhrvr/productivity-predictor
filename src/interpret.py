"""
Two things I actually wanted out of this project once it was working:
1. what actually drives my productivity (feature importance / coefficients)
2. a way to type in tomorrow's plan and get a prediction back
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features import FEATURE_COLUMNS


def plot_feature_importance(model_path: str, out_path: str):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    if isinstance(model, LinearRegression):
        importances = model.coef_
        title = "Linear regression coefficients"
    else:
        importances = model.feature_importances_
        title = "Feature importance"

    order = sorted(range(len(features)), key=lambda i: abs(importances[i]))
    sorted_features = [features[i] for i in order]
    sorted_values = [importances[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e45756" if v < 0 else "#4c78a8" for v in sorted_values]
    ax.barh(sorted_features, sorted_values, color=colors)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def predict_one_day(model_path: str, day: dict) -> float:
    """day is a plain dict with the raw fields, e.g.:
    {"sleep_hours": 7.5, "exercise_minutes": 30, "screen_time_hours": 2,
     "caffeine_mg": 180, "mood_1_10": 7, "date": "2026-09-22"}
    """
    bundle = joblib.load(model_path)
    model = bundle["model"]

    date = pd.Timestamp(day["date"])
    row = {
        "sleep_hours": day["sleep_hours"],
        "exercise_minutes": day["exercise_minutes"],
        "screen_time_hours": day["screen_time_hours"],
        "caffeine_mg": day["caffeine_mg"],
        "mood_1_10": day["mood_1_10"],
        "is_weekend": int(date.dayofweek >= 5),
        "day_of_week": date.dayofweek,
        "well_rested": int(day["sleep_hours"] >= 7),
    }
    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    prediction = model.predict(X)[0]
    return round(float(prediction), 2)


if __name__ == "__main__":
    plot_feature_importance("models/productivity_model.joblib", "output/feature_importance.png")

    example = {
        "date": "2026-09-22",  # a Tuesday
        "sleep_hours": 7.5,
        "exercise_minutes": 30,
        "screen_time_hours": 2.0,
        "caffeine_mg": 180,
        "mood_1_10": 7.0,
    }
    predicted = predict_one_day("models/productivity_model.joblib", example)
    print(f"predicted productive hours: {predicted}")
