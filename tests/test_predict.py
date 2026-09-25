import pandas as pd
import pytest

from src.train import train_and_select_best
from src.interpret import predict_one_day


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Train on the real sample data once per test module — training is
    fast enough (150 rows, simple models) that this doesn't slow tests
    down noticeably."""
    out_dir = tmp_path_factory.mktemp("model")
    model_path = str(out_dir / "model.joblib")
    train_and_select_best("data/raw/daily_log.csv", model_path)
    return model_path


def test_predict_returns_a_float(trained_model):
    day = {
        "date": "2026-09-22",
        "sleep_hours": 7.5,
        "exercise_minutes": 30,
        "screen_time_hours": 2.0,
        "caffeine_mg": 180,
        "mood_1_10": 7.0,
    }
    prediction = predict_one_day(trained_model, day)
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 12


def test_more_sleep_and_exercise_predicts_higher_productivity(trained_model):
    tired_day = {
        "date": "2026-09-22", "sleep_hours": 5.0, "exercise_minutes": 0,
        "screen_time_hours": 6.0, "caffeine_mg": 100, "mood_1_10": 4.0,
    }
    rested_day = {
        "date": "2026-09-22", "sleep_hours": 8.0, "exercise_minutes": 40,
        "screen_time_hours": 1.0, "caffeine_mg": 180, "mood_1_10": 8.0,
    }

    tired_prediction = predict_one_day(trained_model, tired_day)
    rested_prediction = predict_one_day(trained_model, rested_day)

    assert rested_prediction > tired_prediction
