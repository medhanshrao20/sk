"""
Cyclical feature encoding for temporal and wind-direction variables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour/month/WD cyclical encodings and drop raw WD column.
    """
    out = df.copy()
    hours = out.index.hour
    months = out.index.month

    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    out["month_sin"] = np.sin(2 * np.pi * months / 12)
    out["month_cos"] = np.cos(2 * np.pi * months / 12)

    if "WD" in out.columns:
        wd = out["WD"].astype(float)
        out["WD_sin"] = np.sin(2 * np.pi * wd / 360)
        out["WD_cos"] = np.cos(2 * np.pi * wd / 360)
        out = out.drop(columns=["WD"])

    return out
