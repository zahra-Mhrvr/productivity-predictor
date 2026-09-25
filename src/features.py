"""
Turns the raw daily log into a feature matrix ready for modeling.

Nothing fancy — the raw columns are already mostly usable, but I add a
couple of derived features that turned out to matter once I started
looking at the data:

- day_of_week, since productivity clearly dips on weekends
- a binary "well_rested" flag (7+ hours), since the relationship between
  sleep and productivity looked more like a threshold than a straight line
  when I first plotted it
"""

import pandas as pd

FEATURE_COLUMNS = [
    "sleep_hours",
    "exercise_minutes",
    "screen_time_hours",
    "caffeine_mg",
    "mood_1_10",
    "is_weekend",
    "day_of_week",
    "well_rested",
]
TARGET_COLUMN = "productive_hours"


def load_daily_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["well_rested"] = (df["sleep_hours"] >= 7).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Returns (X, y) ready to hand to a scikit-learn model."""
    df = add_features(df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def load_features(path: str):
    df = load_daily_log(path)
    return build_feature_matrix(df)
