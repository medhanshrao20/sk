"""
Probabilistic forecasting: bootstrap, conformal, and quantile intervals.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import FORECAST_STEPS
from evaluation.metrics import interval_coverage_width


def run_bootstrap_intervals(
    forecaster,
    steps: int,
    exog_future: pd.DataFrame,
    interval: list[int] | tuple[int, int] = (10, 90),
    n_boot: int = 500,
) -> pd.DataFrame:
    preds = forecaster.predict_interval(
        steps=steps,
        exog=exog_future.iloc[:steps],
        interval=list(interval),
        n_boot=n_boot,
        use_in_sample_residuals=True,
    )
    return preds


def run_conformal_intervals(
    forecaster,
    y_true: pd.Series,
    y_pred: pd.Series,
    steps: int,
    exog_future: pd.DataFrame,
    interval: list[int] | tuple[int, int] = (10, 90),
    n_boot: int = 500,
) -> pd.DataFrame:
    aligned = pd.concat(
        [y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1
    ).dropna()
    forecaster.set_out_sample_residuals(
        y_true=aligned["y_true"],
        y_pred=aligned["y_pred"],
    )
    preds = forecaster.predict_interval(
        steps=steps,
        exog=exog_future.iloc[:steps],
        interval=list(interval),
        n_boot=n_boot,
        use_in_sample_residuals=False,
    )
    return preds


def run_quantile_intervals(
    forecaster_lower,
    forecaster_upper,
    y_fit: pd.Series,
    exog_fit: pd.DataFrame,
    steps: int,
    exog_future: pd.DataFrame,
) -> pd.DataFrame:
    forecaster_lower.fit(y=y_fit, exog=exog_fit)
    forecaster_upper.fit(y=y_fit, exog=exog_fit)
    lower = forecaster_lower.predict(steps=steps, exog=exog_future.iloc[:steps])
    upper = forecaster_upper.predict(steps=steps, exog=exog_future.iloc[:steps])
    point = (lower + upper) / 2
    return pd.DataFrame(
        {
            "pred": point,
            "lower_bound": lower,
            "upper_bound": upper,
        },
        index=exog_future.index[:steps],
    )


def save_probabilistic_outputs(
    season_dir: Path,
    bootstrap_df: pd.DataFrame,
    conformal_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    y_true: pd.Series,
    interval_metrics: pd.DataFrame,
) -> None:
    prob_dir = season_dir / "probabilistic"
    prob_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_df.to_csv(prob_dir / "bootstrap_intervals.csv")
    conformal_df.to_csv(prob_dir / "conformal_intervals.csv")
    quantile_df.to_csv(prob_dir / "quantile_intervals.csv")
    interval_metrics.to_csv(prob_dir / "interval_metrics.csv", index=False)
    print(f"[probabilistic] Saved interval outputs to {prob_dir}")


def build_interval_metrics(
    y_true: pd.Series,
    bootstrap_df: pd.DataFrame,
    conformal_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for method, frame in [
        ("bootstrap", bootstrap_df),
        ("conformal", conformal_df),
        ("quantile", quantile_df),
    ]:
        lo_col, hi_col = _resolve_bound_columns(frame)
        stats = interval_coverage_width(
            y_true=y_true,
            lower=frame[lo_col],
            upper=frame[hi_col],
        )
        rows.append({"Method": method, **stats})
    return pd.DataFrame(rows)


def _resolve_bound_columns(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        raise ValueError("Interval frame is empty; cannot resolve bound columns.")
    cols = {c.lower(): c for c in frame.columns}
    if "lower_bound" in cols:
        return cols["lower_bound"], cols["upper_bound"]
    lower = [c for c in frame.columns if "lower" in c.lower()]
    upper = [c for c in frame.columns if "upper" in c.lower()]
    if not lower or not upper:
        raise ValueError(
            f"No lower/upper interval columns found in: {list(frame.columns)}"
        )
    return lower[0], upper[0]
