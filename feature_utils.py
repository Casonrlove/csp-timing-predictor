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


# ---------------------------------------------------------------------------
# Contract-level utilities (Phase 1 of contract model reframe)
# ---------------------------------------------------------------------------

def create_breach_target(df: pd.DataFrame) -> pd.DataFrame:
    """Set target = strike_breached.  No rolling threshold needed.

    For contract-level data the target is the direct physical outcome
    (did the strike get breached during the forward window?), so we
    just copy the column rather than computing a rolling quantile.

    Args:
        df: DataFrame produced by ``CSPDataCollector.create_contract_targets()``.
            Must contain column ``strike_breached``.

    Returns:
        Copy of ``df`` with a new integer column ``target``.
    """
    df = df.copy()
    df["target"] = df["strike_breached"].astype(int)
    return df


def contract_aware_time_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    purge_days: int = 35,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit on unique dates, keeping all strikes for the same
    (date, ticker) in the same fold, with an optional purge gap.

    Standard TimeSeriesSplit on row indices would leak information when
    different-delta rows for the same date end up in train vs val.
    A purge gap is also needed because each label depends on a forward window
    (e.g., 35 days), so train rows too close to validation start can leak
    future information.

    Args:
        df: Multi-strike expanded DataFrame (index = dates, may have
            duplicate dates for different deltas).
        n_splits: Number of folds (default 5).
        purge_days: Embargo gap in calendar days before each validation fold.
            Train dates newer than (val_start - purge_days) are removed.

    Returns:
        List of (train_indices, val_indices) tuples — integer position
        indices into ``df``.
    """
    from sklearn.model_selection import TimeSeriesSplit

    # Get unique dates in chronological order
    unique_dates = pd.DatetimeIndex(np.sort(pd.to_datetime(df.index.unique())))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    splits = []
    for date_train_idx, date_val_idx in tscv.split(unique_dates):
        train_dates = unique_dates[date_train_idx]
        val_dates = unique_dates[date_val_idx]

        if purge_days > 0 and len(val_dates) > 0:
            val_start = val_dates.min()
            cutoff = val_start - pd.Timedelta(days=purge_days)
            train_dates = train_dates[train_dates <= cutoff]

        if len(train_dates) == 0 or len(val_dates) == 0:
            continue

        # Map back to row indices
        train_mask = df.index.isin(train_dates)
        val_mask = df.index.isin(val_dates)

        train_rows = np.where(train_mask)[0]
        val_rows = np.where(val_mask)[0]
        if len(train_rows) == 0 or len(val_rows) == 0:
            continue
        splits.append((train_rows, val_rows))

    if not splits:
        raise ValueError("contract_aware_time_split produced no valid folds")

    return splits


# 7 contract interaction feature names (must match data_collector.create_contract_targets)
CONTRACT_FEATURE_COLS = [
    'target_delta',
    'delta_x_vol',
    'delta_x_vix',
    'delta_x_iv_rank',
    'delta_squared',
    'delta_x_rsi',
    'strike_otm_x_atr',
]
