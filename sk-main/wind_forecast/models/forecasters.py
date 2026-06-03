"""
Build all eight forecaster objects (A–H) per skforecast documentation patterns.
"""
from __future__ import annotations

from copy import deepcopy

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirect
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive
from xgboost import XGBRegressor

from config import FORECAST_STEPS, LAGS, RANDOM_STATE, SEASON_DIFFERENTIATION

LGBM_BASE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    verbose=-1,
)

XGB_BASE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    verbosity=0,
    tree_method="hist",
)


def _transformers() -> tuple[StandardScaler, StandardScaler]:
    return StandardScaler(), StandardScaler()


def build_window_features() -> list[RollingFeatures]:
    """Rolling mean/std/min/max at 6h, 12h, and 24h (skforecast 1:1 stat/window pairing)."""
    return [
        RollingFeatures(
            stats=["mean", "std", "min", "max"],
            window_sizes=[ws, ws, ws, ws],
        )
        for ws in (6, 12, 24)
    ]


def build_forecaster_a(season: str) -> ForecasterRecursive:
    ty, tex = _transformers()
    return ForecasterRecursive(
        estimator=LGBMRegressor(**LGBM_BASE_PARAMS),
        lags=LAGS,
        window_features=build_window_features(),
        transformer_y=ty,
        transformer_exog=tex,
        differentiation=SEASON_DIFFERENTIATION[season],
    )


def build_forecaster_b() -> ForecasterDirect:
    ty, tex = _transformers()
    return ForecasterDirect(
        estimator=LGBMRegressor(**LGBM_BASE_PARAMS),
        steps=FORECAST_STEPS,
        lags=LAGS,
        window_features=build_window_features(),
        transformer_y=ty,
        transformer_exog=tex,
    )


def build_forecaster_c() -> ForecasterEquivalentDate:
    return ForecasterEquivalentDate(
        offset=pd.DateOffset(weeks=1),
        n_offsets=4,
    )


def build_forecaster_d_lower(season: str) -> ForecasterRecursive:
    ty, tex = _transformers()
    params = {**LGBM_BASE_PARAMS, "objective": "quantile", "alpha": 0.1}
    return ForecasterRecursive(
        estimator=LGBMRegressor(**params),
        lags=LAGS,
        window_features=build_window_features(),
        transformer_y=ty,
        transformer_exog=tex,
        differentiation=SEASON_DIFFERENTIATION[season],
    )


def build_forecaster_d_upper(season: str) -> ForecasterRecursive:
    ty, tex = _transformers()
    params = {**LGBM_BASE_PARAMS, "objective": "quantile", "alpha": 0.9}
    return ForecasterRecursive(
        estimator=LGBMRegressor(**params),
        lags=LAGS,
        window_features=build_window_features(),
        transformer_y=ty,
        transformer_exog=tex,
        differentiation=SEASON_DIFFERENTIATION[season],
    )


def build_forecaster_e(season: str) -> ForecasterRecursive:
    ty, tex = _transformers()
    return ForecasterRecursive(
        estimator=XGBRegressor(**XGB_BASE_PARAMS),
        lags=LAGS,
        window_features=build_window_features(),
        transformer_y=ty,
        transformer_exog=tex,
        differentiation=SEASON_DIFFERENTIATION[season],
    )


def build_forecaster_f(season: str) -> ForecasterRecursive:
    ty, tex = _transformers()
    return ForecasterRecursive(
        estimator=Ridge(alpha=1.0),
        lags=LAGS,
        window_features=build_window_features(),
        transformer_y=ty,
        transformer_exog=tex,
        differentiation=SEASON_DIFFERENTIATION[season],
    )


def build_forecaster_g(season: str) -> ForecasterRecursive:
    ty, tex = _transformers()
    return ForecasterRecursive(
        estimator=RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        lags=LAGS,
        window_features=build_window_features(),
        transformer_y=ty,
        transformer_exog=tex,
        differentiation=SEASON_DIFFERENTIATION[season],
    )


def build_all_forecasters(season: str) -> dict[str, object]:
    """Return dict with keys A–G (base) for a season."""
    return {
        "A": build_forecaster_a(season),
        "B": build_forecaster_b(),
        "C": build_forecaster_c(),
        "D_lower": build_forecaster_d_lower(season),
        "D_upper": build_forecaster_d_upper(season),
        "E": build_forecaster_e(season),
        "F": build_forecaster_f(season),
        "G": build_forecaster_g(season),
    }


def get_stacking_base_keys() -> tuple[str, ...]:
    return ("A", "E", "F")
