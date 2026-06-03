"""
Generate and save all season-level diagnostic plots (no plt.show()).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skforecast.model_selection import TimeSeriesFold

from config import FORECAST_STEPS, HORIZONS


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_data_split_overview(
    y: pd.Series,
    train_end,
    val_end,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(y.index, y.values, color="0.7", lw=0.5)
    ax.axvline(train_end, color="C0", ls="--", label="train|val")
    ax.axvline(val_end, color="C1", ls="--", label="val|test")
    ax.set_title("Train / Validation / Test timeline")
    ax.legend()
    _save(fig, out_path)


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    intervals: pd.DataFrame | None,
    out_path: Path,
    title: str,
    last_n: int | None = None,
) -> None:
    yt = y_true.copy()
    yp = y_pred.reindex(yt.index).dropna()
    if last_n is not None:
        yt = yt.iloc[-last_n:]
        yp = yp.iloc[-last_n:]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(yt.index, yt.values, label="Actual WS", lw=1)
    ax.plot(yp.index, yp.values, label="Predicted WS", lw=1)
    if intervals is not None and not intervals.empty:
        lo, hi = _interval_cols(intervals)
        band = intervals.reindex(yp.index).dropna()
        ax.fill_between(
            band.index,
            band[lo],
            band[hi],
            alpha=0.25,
            label="80% interval",
        )
    ax.set_title(title)
    ax.legend()
    _save(fig, out_path)


def plot_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    out_path: Path,
    title: str,
) -> None:
    aligned = pd.concat(
        [y_true.rename("y"), y_pred.rename("pred")], axis=1
    ).dropna()
    residuals = aligned["y"] - aligned["pred"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title(f"{title} — residual distribution")
    axes[1].plot(residuals.index, residuals.values, lw=0.6)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title(f"{title} — residuals over time")
    fig.tight_layout()
    _save(fig, out_path)


def plot_horizon_rmse_bar(
    validation_metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    pivot = validation_metrics.pivot_table(
        index="Model", columns="Horizon", values="RMSE"
    )
    horizon_cols = [h for h in HORIZONS if h in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    if not horizon_cols:
        plt.close(fig)
        return
    pivot[horizon_cols].plot(kind="bar", ax=ax)
    ax.set_title("RMSE by model and horizon (validation)")
    ax.set_ylabel("RMSE")
    ax.legend(title="Horizon")
    fig.tight_layout()
    _save(fig, out_path)


def plot_model_comparison_heatmap(
    validation_metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    agg = (
        validation_metrics.groupby(["Model", "Horizon"])[
            ["MAE", "RMSE", "MAPE", "MedAE"]
        ]
        .mean()
        .reset_index()
    )
    heat = agg.pivot(index="Model", columns="Horizon", values="RMSE")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("Validation RMSE heatmap (all models)")
    _save(fig, out_path)


def plot_prediction_intervals_comparison(
    y_true: pd.Series,
    point: pd.Series,
    bootstrap_df: pd.DataFrame,
    conformal_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    idx = point.index
    ax.plot(idx, y_true.reindex(idx).values, label="Actual", marker="o", ms=3)
    ax.plot(idx, point.values, label="Point forecast", lw=1.5)
    for label, frame, color in [
        ("Bootstrap", bootstrap_df, "C0"),
        ("Conformal", conformal_df, "C1"),
        ("Quantile", quantile_df, "C2"),
    ]:
        lo, hi = _interval_cols(frame)
        band = frame.reindex(idx)
        ax.fill_between(
            band.index,
            band[lo],
            band[hi],
            alpha=0.15,
            color=color,
            label=label,
        )
    ax.set_title("Probabilistic intervals comparison (24h test window)")
    ax.legend()
    _save(fig, out_path)


def plot_feature_importance_lgbm(
    forecaster,
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    out_path: Path,
    top_n: int = 20,
) -> None:
    from lightgbm import LGBMRegressor

    est = forecaster.estimator
    if not isinstance(est, LGBMRegressor):
        return
    X_train_matrix, _ = forecaster.create_train_X_y(y=y_train, exog=exog_train)
    imp = pd.Series(est.feature_importances_, index=X_train_matrix.columns)
    imp = imp.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_title("Top LightGBM split importances")
    _save(fig, out_path)


def plot_backtesting_coverage(
    y: pd.Series,
    cv: TimeSeriesFold,
    out_path: Path,
) -> None:
    folds_df = cv.split(y, as_pandas=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    for _, row in folds_df.iterrows():
        t0 = y.index[int(row["test_start"])]
        t1 = y.index[int(row["test_end"]) - 1]
        ax.barh(
            int(row["fold"]),
            width=(t1 - t0).total_seconds(),
            left=t0,
            height=0.4,
            alpha=0.6,
        )
    ax.set_title("Backtesting fold prediction windows")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fold")
    _save(fig, out_path)


def plot_master_metrics_heatmap(
    master_df: pd.DataFrame,
    out_path: Path,
) -> None:
    test = master_df[master_df["Split"] == "test"]
    heat = test.pivot_table(
        index="Season", columns="Horizon", values="RMSE", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="mako", ax=ax)
    ax.set_title("RMSE across Season × Horizon (test, best models)")
    _save(fig, out_path)


def generate_all_season_plots(
    season: str,
    plots_dir: Path,
    y_full: pd.Series,
    splits,
    validation_metrics: pd.DataFrame,
    val_predictions: pd.Series,
    test_predictions: pd.Series,
    val_intervals: pd.DataFrame | None,
    test_intervals: pd.DataFrame | None,
    bootstrap_df: pd.DataFrame,
    conformal_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    best_forecaster,
    cv_val: TimeSeriesFold,
    y_val: pd.Series,
    y_test: pd.Series,
    y_train: pd.Series,
    exog_train: pd.DataFrame,
) -> None:
    train_end = splits.train.index[-1]
    val_end = splits.validation.index[-1]

    plot_data_split_overview(
        y_full, train_end, val_end, plots_dir / "data_split_overview.png"
    )
    plot_actual_vs_predicted(
        y_val,
        val_predictions,
        val_intervals,
        plots_dir / "actual_vs_predicted_val.png",
        title=f"{season} validation — best model",
    )
    plot_actual_vs_predicted(
        y_test,
        test_predictions,
        test_intervals,
        plots_dir / "actual_vs_predicted_test.png",
        title=f"{season} test — best model (last 30 days)",
        last_n=24 * 30,
    )
    plot_residuals(
        y_val,
        val_predictions,
        plots_dir / "residuals_val.png",
        title=f"{season} validation",
    )
    plot_residuals(
        y_test,
        test_predictions,
        plots_dir / "residuals_test.png",
        title=f"{season} test",
    )
    plot_horizon_rmse_bar(validation_metrics, plots_dir / "horizon_rmse_bar.png")
    plot_model_comparison_heatmap(
        validation_metrics, plots_dir / "model_comparison_val.png"
    )
    point = test_predictions.iloc[:FORECAST_STEPS]
    plot_prediction_intervals_comparison(
        y_test.iloc[:FORECAST_STEPS],
        point,
        bootstrap_df.iloc[:FORECAST_STEPS],
        conformal_df.iloc[:FORECAST_STEPS],
        quantile_df.iloc[:FORECAST_STEPS],
        plots_dir / "prediction_intervals_comparison.png",
    )
    plot_feature_importance_lgbm(
        best_forecaster,
        y_train,
        exog_train,
        plots_dir / "feature_importance_lgbm.png",
    )
    plot_backtesting_coverage(
        y_full, cv_val, plots_dir / "backtesting_coverage.png"
    )


def _interval_cols(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        raise ValueError("Interval frame is empty; cannot resolve bound columns.")
    cols = list(frame.columns)
    lower = [c for c in cols if "lower" in c.lower()]
    upper = [c for c in cols if "upper" in c.lower()]
    if not lower or not upper:
        raise ValueError(
            f"No lower/upper interval columns found in: {cols}"
        )
    return lower[0], upper[0]
