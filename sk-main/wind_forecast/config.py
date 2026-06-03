"""
Global configuration for the wind power forecasting pipeline.
"""
from __future__ import annotations

from pathlib import Path

# Project root (wind_forecast/)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "hourly.csv"
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_SAVED_ROOT = PROJECT_ROOT / "models" / "saved"
PIPELINE_ERROR_LOG = RESULTS_ROOT / "pipeline_errors.log"

TARGET_COL = "WS"
RAW_EXOG_COLS = [
    "WD",
    "AT",
    "RH",
    "ET",
    "SR",
    "VP",
    "DPT",
    "PCP",
    "ST",
]

SEASONS = ("spring", "summer", "autumn", "winter")

SEASON_MONTHS = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
    "winter": (12, 1, 2),
}

SEASON_EXOG_COLS = {
    "spring": [
        "ET",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "WD_sin",
        "WD_cos",
    ],
    "summer": [
        "ET",
        "AT",
        "RH",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "WD_sin",
        "WD_cos",
    ],
    "autumn": [
        "ET",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "WD_sin",
        "WD_cos",
    ],
    "winter": [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "WD_sin",
        "WD_cos",
    ],
}

SEASON_DIFFERENTIATION = {
    "spring": None,
    "summer": 1,
    "autumn": 1,
    "winter": 1,
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
LAGS_GRID = [
    [1, 2, 3, 6, 12, 24],
    [1, 2, 3, 6, 12, 24, 48],
    [1, 2, 3, 6, 12, 24, 48, 168],
]

FORECAST_STEPS = 24
HORIZONS = [1, 2, 3, 24]
RANDOM_STATE = 42

N_TRIALS_TUNING = 20
TUNING_METRIC = "mean_absolute_error"

MODEL_KEYS = {
    "A": "forecaster_recursive_lgbm",
    "B": "forecaster_direct_lgbm",
    "C": "forecaster_baseline",
    "D_lower": "forecaster_quantile_lower",
    "D_upper": "forecaster_quantile_upper",
    "E": "forecaster_recursive_xgb",
    "F": "forecaster_recursive_ridge",
    "G": "forecaster_recursive_rf",
    "H": "stacking_meta_learner",
}

TUNABLE_MODELS = ("A", "B", "E", "F", "G")
ALL_MODEL_LABELS = ("A", "B", "C", "D", "E", "F", "G", "H")
