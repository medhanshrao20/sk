"""
Bayesian hyperparameter search (Optuna) following skforecast 0.22+ API.

Parameter ranges mirror the user-specified scikit-optimize Real/Integer grids.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect

from config import (
    FORECAST_STEPS,
    LAGS_GRID,
    MODEL_KEYS,
    N_TRIALS_TUNING,
    RANDOM_STATE,
    TUNABLE_MODELS,
    TUNING_METRIC,
)
from models.forecasters import build_all_forecasters

# scikit-optimize space equivalents (documented ranges; search uses Optuna)
try:
    from skopt.space import Integer, Real  # noqa: F401 — required dependency
except ImportError:  # pragma: no cover
    Integer = Real = None  # type: ignore


def _tree_search_space(trial, is_lgbm: bool) -> dict:
    space = {
        "lags": trial.suggest_categorical("lags", LAGS_GRID),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if is_lgbm:
        space["num_leaves"] = trial.suggest_int("num_leaves", 20, 100)
        space["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 50)
    return space


def _ridge_search_space(trial) -> dict:
    return {
        "lags": trial.suggest_categorical("lags", LAGS_GRID),
        "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
    }


def _rf_search_space(trial) -> dict:
    return {
        "lags": trial.suggest_categorical("lags", LAGS_GRID),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
    }


def make_search_space(model_key: str) -> Callable:
    if model_key in ("A", "B"):
        is_lgbm = True

        def search_space(trial):
            return _tree_search_space(trial, is_lgbm=is_lgbm)

        return search_space
    if model_key == "E":

        def search_space(trial):
            return _tree_search_space(trial, is_lgbm=False)

        return search_space
    if model_key == "F":
        return _ridge_search_space
    if model_key == "G":
        return _rf_search_space
    raise ValueError(f"Model {model_key} is not tunable.")


def tune_forecasters(
    season: str,
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    results_dir: Path,
) -> dict[str, object]:
    """
    Run bayesian_search_forecaster on train only for models A, B, E, F, G.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    forecasters = build_all_forecasters(season)
    tuned: dict[str, object] = {}

    for key in TUNABLE_MODELS:
        forecaster = forecasters[key]
        cv_tune = TimeSeriesFold(
            steps=FORECAST_STEPS,
            initial_train_size=int(len(y_train) * 0.8),
            refit=False,
            fixed_train_size=False,
            gap=0,
            differentiation=getattr(forecaster, "differentiation", None),
        )
        search_space = make_search_space(key)
        print(f"[{season}] Bayesian tuning model {key} ...")
        results, study = bayesian_search_forecaster(
            forecaster=forecaster,
            y=y_train,
            exog=exog_train,
            cv=cv_tune,
            search_space=search_space,
            metric=TUNING_METRIC,
            n_trials=N_TRIALS_TUNING,
            random_state=RANDOM_STATE,
            return_best=True,
            verbose=True,
            show_progress=True,
        )
        model_name = MODEL_KEYS[key]
        results.to_csv(
            results_dir / f"{model_name}_bayesian_search_results.csv",
            index=False,
        )
        best_params = _extract_best_params(study, forecaster, key)
        with open(
            results_dir / f"{model_name}_best_params.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(best_params, fh, indent=2, default=str)
        tuned[key] = forecaster
        print(f"[{season}] Best params ({key}): {best_params}")

    for key in ("C", "D_lower", "D_upper"):
        tuned[key] = forecasters[key]

    return tuned


def _extract_best_params(study, forecaster, model_key: str) -> dict:
    trial = study.best_trial
    params = dict(trial.params)
    if "lags" in params:
        params["lags"] = list(params["lags"])
    if model_key == "F" and hasattr(forecaster, "estimator"):
        params["estimator"] = type(forecaster.estimator).__name__
    return params
