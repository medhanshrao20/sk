"""
Per-horizon regression metrics for multi-step backtesting outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_horizon_metrics(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    horizons: list[int],
    steps: int = 24,
) -> pd.DataFrame:
    """
    Compute MAE, RMSE, MAPE, MedAE at selected horizons from backtest predictions.

    Predictions DataFrame must contain columns ``fold`` and ``pred`` indexed by time.
    """
    pred_col = "pred" if "pred" in predictions.columns else predictions.columns[-1]
    merged = predictions[[pred_col, "fold"]].copy()
    merged["y_true"] = y_true.reindex(merged.index)
    merged = merged.dropna(subset=["y_true", pred_col])

    records: list[dict] = []
    for fold_id, group in merged.groupby("fold"):
        group = group.sort_index()
        for i, (idx, row) in enumerate(group.iterrows()):
            horizon = (i % steps) + 1
            if horizon in horizons:
                records.append(
                    {
                        "fold": fold_id,
                        "horizon": horizon,
                        "datetime": idx,
                        "y_true": row["y_true"],
                        "y_pred": row[pred_col],
                    }
                )

    detail = pd.DataFrame(records)
    if detail.empty:
        raise ValueError("No aligned predictions for horizon metrics.")

    rows = []
    for h in horizons:
        sub = detail[detail["horizon"] == h]
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        rows.append(
            {
                "Horizon": h,
                "MAE": mean_absolute_error(yt, yp),
                "RMSE": _rmse(yt, yp),
                "MAPE": mean_absolute_percentage_error(yt, yp),
                "MedAE": median_absolute_error(yt, yp),
                "n_samples": len(sub),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(
    metrics_df: pd.DataFrame,
    model: str,
    season: str,
    split: str,
) -> pd.DataFrame:
    out = metrics_df.copy()
    out.insert(0, "Season", season)
    out.insert(1, "Model", model)
    out.insert(2, "Split", split)
    out = out.rename(columns={"Horizon": "Horizon"})
    return out


def interval_coverage_width(
    y_true: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> dict[str, float]:
    aligned = pd.concat(
        [y_true.rename("y"), lower.rename("lo"), upper.rename("hi")],
        axis=1,
    ).dropna()
    covered = (aligned["y"] >= aligned["lo"]) & (aligned["y"] <= aligned["hi"])
    return {
        "IntervalCoverage": float(covered.mean()),
        "IntervalWidth": float((aligned["hi"] - aligned["lo"]).mean()),
    }
