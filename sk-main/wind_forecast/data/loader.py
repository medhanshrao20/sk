"""
Load hourly wind data from CSV, validate, impute, and assert data quality.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATA_PATH, PROJECT_ROOT, TARGET_COL

DATETIME_CANDIDATES = ("datetime", "Datetime", "date_time", "timestamp", "DateTime")


def _resolve_data_path(data_path: str | Path | None) -> Path:
    if data_path is None:
        return DATA_PATH
    path = Path(data_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_and_preprocess(data_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load CSV, parse datetime index, set hourly frequency, impute NaNs.

    Raises
    ------
    FileNotFoundError
        If ``data/hourly.csv`` is missing.
    """
    path = _resolve_data_path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place hourly.csv inside wind_forecast/data/"
        )

    df = pd.read_csv(path)
    df = _build_datetime_index(df)
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.asfreq("h")
    if df.index.freq is None:
        df.index.freq = "h"

    df = _standardize_column_names(df)
    df = df.ffill().bfill()
    if df.isna().any().any():
        raise ValueError("NaNs remain after ffill/bfill imputation.")

    return df


COLUMN_ALIASES = {
    "Wind Speed (mph)": "WS",
    "Wind Dir (0-360)": "WD",
    "Air Temp (F)": "AT",
    "Rel Hum (%)": "RH",
    "ETo (in)": "ET",
    "Sol Rad (Ly/day)": "SR",
    "Vap Pres (mBars)": "VP",
    "Dew Point (F)": "DPT",
    "Precip (in)": "PCP",
    "Soil Temp (F)": "ST",
}


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=COLUMN_ALIASES)
    return out


def _build_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DATETIME_CANDIDATES:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
            return out.set_index(col).sort_index()
    if {"Date", "Hour (PST)"}.issubset(out.columns):
        dt = pd.to_datetime(
            out["Date"].astype(str)
            + " "
            + out["Hour (PST)"].astype(str).str.zfill(4),
            format="%m/%d/%Y %H%M",
            errors="coerce",
        )
        return out.set_index(dt).sort_index()
    raise ValueError(
        "No datetime column found. Provide 'datetime' or 'Date' + 'Hour (PST)'."
    )


def extract_target_exog(df: pd.DataFrame, exog_cols: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Return target series WS and exogenous DataFrame for a season."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")
    missing = [c for c in exog_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing exogenous columns: {missing}")
    y = df[TARGET_COL].copy()
    y.name = TARGET_COL
    exog = df[exog_cols].copy()
    return y, exog
