"""
SHAP explainability for the best tree-based forecaster (or linear fallback).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from lightgbm import LGBMRegressor
from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterEquivalentDate
from xgboost import XGBRegressor

# Cap rows for SHAP on large training sets (memory / runtime on other machines).
MAX_SHAP_SAMPLES = 500


def _resolve_fitted_estimator(forecaster):
    """Return a fitted sklearn estimator suitable for SHAP, or None to skip."""
    if isinstance(forecaster, ForecasterEquivalentDate):
        return None
    if isinstance(forecaster, ForecasterDirect):
        fitted = getattr(forecaster, "estimators_", None)
        if not fitted:
            return None
        step = min(fitted.keys())
        return fitted[step]
    est = getattr(forecaster, "estimator", None)
    if est is None:
        return None
    if hasattr(est, "booster_"):
        return est if est.booster_ is not None else None
    if isinstance(est, RandomForestRegressor):
        return est if getattr(est, "estimators_", None) else None
    if hasattr(est, "coef_"):
        return est
    return est


def _subsample_matrix(X: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X.iloc[sorted(idx)]


def _as_2d_shap_values(shap_values) -> np.ndarray:
    """TreeExplainer may return a list (multi-output) or a 3D array."""
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _scalar_base_value(base) -> float:
    """Waterfall plots require a scalar base value."""
    return float(np.asarray(base).ravel()[0])


def run_shap_analysis(
    forecaster,
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train_matrix, _ = forecaster.create_train_X_y(y=y_train, exog=exog_train)
    X_train_matrix = _subsample_matrix(X_train_matrix, MAX_SHAP_SAMPLES)

    estimator = _resolve_fitted_estimator(forecaster)
    if estimator is None:
        print(
            f"[SHAP] Skipped: no fitted estimator for {type(forecaster).__name__}"
        )
        return pd.DataFrame()

    n_features = getattr(estimator, "n_features_in_", None)
    if n_features is not None and n_features != X_train_matrix.shape[1]:
        print(
            f"[SHAP] Skipped: feature count mismatch "
            f"(model={n_features}, matrix={X_train_matrix.shape[1]})"
        )
        return pd.DataFrame()

    if isinstance(estimator, (LGBMRegressor, XGBRegressor, RandomForestRegressor)):
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(X_train_matrix)
        shap_values = _as_2d_shap_values(explanation.values)
    else:
        explainer = shap.Explainer(estimator, X_train_matrix)
        explanation = explainer(X_train_matrix)
        shap_values = _as_2d_shap_values(explanation.values)

    shap_df = pd.DataFrame(shap_values, columns=X_train_matrix.columns)
    shap_df.to_csv(output_dir / "shap_values.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_train_matrix, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values, X_train_matrix, plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    row_values = np.asarray(shap_values[0]).ravel()
    if hasattr(explanation, "base_values") and explanation.base_values is not None:
        row_base = _scalar_base_value(explanation.base_values[0])
    elif hasattr(explainer, "expected_value"):
        row_base = _scalar_base_value(explainer.expected_value)
    else:
        row_base = float(np.mean(shap_values))

    waterfall_exp = shap.Explanation(
        values=row_values,
        base_values=row_base,
        data=X_train_matrix.iloc[0].values,
        feature_names=X_train_matrix.columns.tolist(),
    )
    if hasattr(shap, "plots") and hasattr(shap.plots, "waterfall"):
        shap.plots.waterfall(waterfall_exp, show=False)
    else:
        shap.waterfall_plot(waterfall_exp, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    top_feature = X_train_matrix.columns[top_idx]
    plt.figure()
    shap.dependence_plot(
        top_feature,
        shap_values,
        X_train_matrix,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_dependence_top.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SHAP] Saved plots and shap_values.csv to {output_dir}")
    return shap_df
