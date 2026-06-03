"""
Manual out-of-fold stacking ensemble (model H).

Base models: A (LightGBM), E (XGBoost), F (Ridge).
Meta-learner: Ridge trained on validation OOF predictions only.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import FORECAST_STEPS
from models.forecasters import get_stacking_base_keys


@dataclass
class StackingArtifacts:
    meta_learner: Ridge
    base_keys: tuple[str, ...]
    weights: np.ndarray
    oof_val_predictions: pd.DataFrame


def _predict_multistep(
    forecaster,
    steps: int,
    y_history: pd.Series,
    exog_future: pd.DataFrame | None,
) -> np.ndarray:
    last_window = y_history.iloc[-forecaster.window_size :]
    if exog_future is not None and len(exog_future) >= steps:
        next_idx = y_history.index[-1] + pd.Timedelta(hours=1)
        exog_aligned = exog_future.loc[exog_future.index >= next_idx].iloc[:steps]
        preds = forecaster.predict(
            steps=steps,
            exog=exog_aligned,
            last_window=last_window,
        )
    else:
        preds = forecaster.predict(steps=steps, last_window=last_window)
    return np.asarray(preds).ravel()[:steps]


def fit_stacking_meta_learner(
    base_forecasters: dict[str, object],
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    y_val: pd.Series,
    exog_val: pd.DataFrame,
) -> StackingArtifacts:
    """
    Fit base models on train; build OOF features on validation; fit Ridge meta.
    """
    base_keys = get_stacking_base_keys()
    val_len = len(y_val)
    oof_matrix = np.zeros((val_len, len(base_keys)))

    for j, key in enumerate(base_keys):
        fc = base_forecasters[key]
        fc.fit(y=y_train, exog=exog_train)
        preds = _predict_multistep(
            fc,
            steps=val_len,
            y_history=y_train,
            exog_future=exog_val,
        )
        oof_matrix[:, j] = preds

    meta = Ridge(alpha=1.0)
    meta.fit(oof_matrix, y_val.values)

    oof_df = pd.DataFrame(
        oof_matrix,
        index=y_val.index,
        columns=[f"base_{k}" for k in base_keys],
    )
    return StackingArtifacts(
        meta_learner=meta,
        base_keys=base_keys,
        weights=meta.coef_,
        oof_val_predictions=oof_df,
    )


def predict_stacking(
    artifacts: StackingArtifacts,
    base_forecasters: dict[str, object],
    y_train: pd.Series,
    y_val: pd.Series,
    exog_train: pd.DataFrame,
    exog_val: pd.DataFrame,
    exog_future: pd.DataFrame,
    steps: int = FORECAST_STEPS,
) -> pd.Series:
    """Predict using base models refit on train+val and meta-learner."""
    y_fit = pd.concat([y_train, y_val])
    exog_fit = pd.concat([exog_train, exog_val])
    base_features = np.zeros((steps, len(artifacts.base_keys)))

    for j, key in enumerate(artifacts.base_keys):
        fc = base_forecasters[key]
        fc.fit(y=y_fit, exog=exog_fit)
        base_features[:, j] = _predict_multistep(
            fc,
            steps=steps,
            y_history=y_fit,
            exog_future=exog_future,
        )

    preds = artifacts.meta_learner.predict(base_features)
    index = exog_future.index[:steps]
    return pd.Series(preds, index=index, name="pred")


def save_stacking_artifacts(path, artifacts: StackingArtifacts) -> None:
    joblib.dump(artifacts, path)


def load_stacking_artifacts(path) -> StackingArtifacts:
    return joblib.load(path)
