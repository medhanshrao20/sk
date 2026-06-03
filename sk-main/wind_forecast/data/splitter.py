"""
Seasonal partitioning and chronological train/validation/test splits.
"""
from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from config import SEASON_MONTHS, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


class SeasonSplits(NamedTuple):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _longest_hourly_contiguous_block(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the longest run of hourly observations (gap > 1h starts a new run)."""
    if len(df) == 0:
        return df
    out = df.sort_index()
    gap = out.index.to_series().diff() > pd.Timedelta(hours=1)
    block_id = gap.cumsum()
    best_id = block_id.value_counts().idxmax()
    block = out.loc[block_id == best_id].copy()
    hourly_idx = pd.date_range(block.index.min(), block.index.max(), freq="h")
    block = block.reindex(hourly_idx).ffill().bfill()
    if block.isna().any().any():
        raise ValueError("Gaps remain inside the longest contiguous hourly block.")
    return block


def ensure_datetime_freq(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Ensure a regular hourly DatetimeIndex for skforecast."""
    out = df.copy().sort_index()
    if out.index.freq is not None:
        return out
    if len(out) == 0:
        return out
    inferred = pd.infer_freq(out.index[: min(len(out), 500)])
    if inferred is not None:
        out.index = pd.DatetimeIndex(out.index, freq=inferred)
        return out
    return _longest_hourly_contiguous_block(out)


def split_by_season(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split full dataset into four meteorological seasons."""
    seasons: dict[str, pd.DataFrame] = {}
    for season, months in SEASON_MONTHS.items():
        mask = df.index.month.isin(months)
        subset = df.loc[mask].copy()
        if season == "winter":
            subset = _longest_hourly_contiguous_block(subset)
        else:
            subset = ensure_datetime_freq(subset)
        seasons[season] = subset
    return seasons


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> SeasonSplits:
    """
    Chronological 70/15/15 split via iloc (never random).
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = ensure_datetime_freq(data.iloc[:train_end].copy())
    validation = ensure_datetime_freq(data.iloc[train_end:val_end].copy())
    test = ensure_datetime_freq(data.iloc[val_end:].copy())
    return SeasonSplits(train=train, validation=validation, test=test)


def print_split_sizes(season: str, splits: SeasonSplits) -> None:
    """Print train/val/test sizes for a season."""
    print(
        f"[{season}] train={len(splits.train)}, "
        f"validation={len(splits.validation)}, test={len(splits.test)}"
    )
