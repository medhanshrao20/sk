"""
Backtesting wrappers around skforecast.model_selection.backtesting_forecaster.
"""
from __future__ import annotations

import pickle

import pandas as pd
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster

from config import FORECAST_STEPS, HORIZONS
from evaluation.metrics import compute_horizon_metrics


def clone_forecaster(forecaster):
    """
    Clone a fitted skforecast forecaster.

    ``copy.deepcopy`` can fail on pandas indexes with a pinned hourly freq
    (winter seasonal subsets); pickle round-trip is safe.
    """
    return pickle.loads(pickle.dumps(forecaster))


def differentiation_for_forecaster(forecaster) -> int | None:
    """Match TimeSeriesFold differentiation to the forecaster (skforecast 0.22+)."""
    return getattr(forecaster, "differentiation", None)


def make_cv_tune(
    initial_train_size: int,
    differentiation: int | None = None,
) -> TimeSeriesFold:
    return TimeSeriesFold(
        steps=FORECAST_STEPS,
        initial_train_size=initial_train_size,
        refit=False,
        fixed_train_size=False,
        gap=0,
        differentiation=differentiation,
    )


def make_cv_validation(
    train_len: int,
    differentiation: int | None = None,
) -> TimeSeriesFold:
    return TimeSeriesFold(
        steps=FORECAST_STEPS,
        initial_train_size=train_len,
        refit=False,
        fixed_train_size=False,
        gap=0,
        differentiation=differentiation,
    )


def make_cv_test(
    train_val_len: int,
    differentiation: int | None = None,
) -> TimeSeriesFold:
    return TimeSeriesFold(
        steps=FORECAST_STEPS,
        initial_train_size=train_val_len,
        refit=True,
        fixed_train_size=False,
        gap=0,
        differentiation=differentiation,
    )


def run_backtesting(
    forecaster,
    y: pd.Series,
    exog: pd.DataFrame | None,
    cv: TimeSeriesFold,
    metrics: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run backtesting_forecaster and return (metric_df, predictions)."""
    if metrics is None:
        metrics = [
            "mean_absolute_error",
            "mean_squared_error",
            "mean_absolute_percentage_error",
        ]
    metric_df, predictions = backtesting_forecaster(
        forecaster=clone_forecaster(forecaster),
        y=y,
        exog=exog,
        cv=cv,
        metric=metrics,
        verbose=True,
        show_progress=True,
    )
    return metric_df, predictions


def evaluate_on_split(
    forecaster,
    y: pd.Series,
    exog: pd.DataFrame | None,
    cv: TimeSeriesFold,
    model_label: str,
    season: str,
    split_name: str,
) -> pd.DataFrame:
    _, predictions = run_backtesting(forecaster, y, exog, cv)
    val_start = y.index[min(cv.initial_train_size, len(y) - 1)]
    y_eval = y.loc[val_start:]
    horizon_metrics = compute_horizon_metrics(
        y_true=y_eval,
        predictions=predictions,
        horizons=HORIZONS,
        steps=FORECAST_STEPS,
    )
    return aggregate_with_labels(horizon_metrics, model_label, season, split_name)


def aggregate_with_labels(
    horizon_metrics: pd.DataFrame,
    model_label: str,
    season: str,
    split_name: str,
) -> pd.DataFrame:
    out = horizon_metrics.copy()
    out.insert(0, "Season", season)
    out.insert(1, "Model", model_label)
    out.insert(2, "Split", split_name)
    return out


def select_best_model(validation_metrics: pd.DataFrame) -> str:
    """Select model with lowest mean RMSE across horizons on validation."""
    summary = (
        validation_metrics.groupby("Model")["RMSE"]
        .mean()
        .sort_values()
    )
    best = summary.index[0]
    print(f"Best model by validation RMSE: {best}")
    return best


def extract_oof_residuals(
    y: pd.Series,
    predictions: pd.DataFrame,
) -> pd.Series:
    """Out-of-sample residuals from backtesting predictions."""
    pred_col = "pred" if "pred" in predictions.columns else predictions.columns[-1]
    aligned = predictions[[pred_col]].copy()
    aligned["y_true"] = y.reindex(aligned.index)
    aligned = aligned.dropna()
    return aligned["y_true"] - aligned[pred_col]
