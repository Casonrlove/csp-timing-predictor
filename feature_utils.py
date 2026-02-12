"""
Shared feature utilities for CSP timing model training and validation.

Having apply_rolling_threshold() here ensures that both the training scripts
and walkforward_validation.py produce labels from the same logic so AUC
numbers are comparable.
"""

import numpy as np
import pandas as pd


def apply_rolling_threshold(
    df: pd.DataFrame,
    window: int = 90,
    quantile: float = 0.60,
) -> pd.DataFrame:
    """Assign binary target using a per-ticker rolling window threshold.

    For each row the threshold is the ``quantile``-th percentile of the
    trailing ``window`` days of Max_Drawdown_35D for that ticker.  This keeps
    the positive rate near (1 - quantile) in every calendar year regardless of
    whether the market is in a bull or bear regime — the model always sees
    roughly the same class balance, rather than wildly swinging from 17 % in a
    bear year to 46 % in a bull year.

    Falls back to the full-history threshold for early rows that don't yet have
    ``window // 2`` observations.

    Args:
        df: DataFrame with columns ``ticker`` and ``Max_Drawdown_35D``.
        window: Look-back window in trading days (default 90).
        quantile: Target quantile for threshold (default 0.60 → P60).

    Returns:
        Copy of ``df`` with a new integer column ``target`` (0/1).
    """
    df = df.copy()
    df["target"] = 0
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        drawdowns = df.loc[mask, "Max_Drawdown_35D"]
        rolling_thresh = drawdowns.rolling(
            window=window, min_periods=window // 2
        ).quantile(quantile)
        # Rows without enough history fall back to the full-ticker percentile.
        fallback = float(drawdowns.quantile(quantile)) if len(drawdowns) >= 10 else -5.0
        rolling_thresh = rolling_thresh.fillna(fallback)
        df.loc[mask, "target"] = (drawdowns > rolling_thresh).astype(int)
    return df


def calculate_group_threshold(df: pd.DataFrame) -> float:
    """60th-percentile drawdown threshold over full history (simple baseline).

    Kept for reference / fallback.  Prefer apply_rolling_threshold() for
    training data so that training labels match walk-forward validation labels.
    """
    drawdowns = df["Max_Drawdown_35D"].dropna()
    return float(drawdowns.quantile(0.60)) if len(drawdowns) >= 10 else -5.0
