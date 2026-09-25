import pandas as pd

from src.features import add_features, build_feature_matrix, FEATURE_COLUMNS


def make_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-09-21", "2026-09-23"]),  # Mon, Wed
        "sleep_hours": [6.5, 7.5],
        "exercise_minutes": [10, 30],
        "screen_time_hours": [4.0, 2.0],
        "caffeine_mg": [200, 150],
        "mood_1_10": [5.0, 8.0],
        "is_weekend": [False, False],
        "productive_hours": [5.0, 8.0],
    })


def test_day_of_week_is_extracted_correctly():
    df = add_features(make_df())
    # 2026-09-21 is a Monday (0), 2026-09-23 is a Wednesday (2)
    assert df["day_of_week"].tolist() == [0, 2]


def test_well_rested_flag_uses_seven_hour_threshold():
    df = add_features(make_df())
    assert df["well_rested"].tolist() == [0, 1]


def test_build_feature_matrix_returns_expected_columns():
    X, y = build_feature_matrix(make_df())
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.tolist() == [5.0, 8.0]


def test_is_weekend_becomes_integer():
    X, _ = build_feature_matrix(make_df())
    assert X["is_weekend"].dtype in ("int64", "int32")
