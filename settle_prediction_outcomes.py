"""
Bulk-settle prediction outcomes in predictions_log.json.

Settlement logic (default):
- Mature only predictions at least `forward_days` trading bars old.
- Outcome = 1 if max drawdown over the forward window is > -drop_threshold.
- Outcome = 0 otherwise.

Uses Schwab API when available (preferred), falls back to Yahoo Finance.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Try Schwab first, fall back to Yahoo
try:
    from schwab_client import get_price_history
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False

import yfinance as yf

CONTRACT_FORWARD_DAYS_DEFAULT = 35


def parse_args():
    p = argparse.ArgumentParser(description="Bulk-settle CSP prediction outcomes")
    p.add_argument("--log", default="predictions_log.json", help="Path to predictions log JSON")
    p.add_argument("--forward-days", type=int, default=35, help="Forward window in trading bars")
    p.add_argument("--drop-threshold", type=float, default=0.05, help="Drop threshold fraction (0.05 = 5%%)")
    p.add_argument("--period", default="6mo", help="History period to fetch per ticker (Yahoo fallback)")
    p.add_argument("--dry-run", action="store_true", help="Compute settlements without writing")
    p.add_argument("--backup", action="store_true", help="Write .bak copy before saving")
    p.add_argument("--force-yahoo", action="store_true", help="Force Yahoo Finance even if Schwab is available")
    return p.parse_args()


def load_log(path: Path):
    with path.open("r") as f:
        return json.load(f)


def fetch_close_series_schwab(ticker: str) -> pd.Series | None:
    """Fetch daily close prices via Schwab API (1 year)."""
    try:
        data = get_price_history(ticker, period_type='year', period=1,
                                 frequency_type='daily', frequency=1)
        if not data or 'candles' not in data or not data['candles']:
            return None
        candles = data['candles']
        dates = pd.to_datetime([c['datetime'] for c in candles], unit='ms').normalize()
        closes = [c['close'] for c in candles]
        return pd.Series(closes, index=dates, name='Close')
    except Exception as e:
        print(f"  Schwab fetch failed for {ticker}: {e}")
        return None


def fetch_close_series_yahoo(ticker: str, period: str) -> pd.Series | None:
    """Fetch daily close prices via Yahoo Finance."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        s = hist["Close"].copy()
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s.index = idx.normalize()
        return s
    except Exception as e:
        print(f"  Yahoo fetch failed for {ticker}: {e}")
        return None


def fetch_close_series(
    ticker: str,
    period: str,
    use_schwab: bool = True,
    allow_yahoo_fallback: bool = True,
) -> pd.Series | None:
    """Fetch close prices, preferring Schwab API with optional Yahoo fallback."""
    if use_schwab and SCHWAB_AVAILABLE:
        series = fetch_close_series_schwab(ticker)
        if series is not None and len(series) > 0:
            return series
        if not allow_yahoo_fallback:
            print(f"  Schwab returned no data for {ticker}; Yahoo fallback disabled")
            return None
        print(f"  Schwab returned no data for {ticker}, falling back to Yahoo")
    if not allow_yahoo_fallback:
        return None
    return fetch_close_series_yahoo(ticker, period)


def settle_prediction(pred: dict, close_series: pd.Series, forward_days: int, drop_threshold: float):
    if close_series is None or close_series.empty:
        return None

    ts = pred.get("timestamp")
    if not ts:
        return None

    try:
        pred_dt = pd.Timestamp(datetime.fromisoformat(ts)).normalize()
    except Exception:
        return None

    # Determine settlement horizon in trading bars.
    # Contract-mode predictions should use the same fixed horizon as training labels
    # (default 35 bars), not expiration-derived windows.
    effective_forward_days = forward_days
    explicit_horizon = False

    logged_horizon = pred.get("label_forward_days")
    if logged_horizon is not None:
        try:
            h = int(logged_horizon)
            if 5 <= h <= 180:
                effective_forward_days = h
                explicit_horizon = True
        except Exception:
            pass

    model_version = str(pred.get("model_version", "")).lower()
    is_contract_prediction = (
        model_version.startswith("contract_")
        or pred.get("model_p_breach") is not None
    )
    if is_contract_prediction and not explicit_horizon:
        # Keep contract settlement fixed to the label horizon.
        effective_forward_days = CONTRACT_FORWARD_DAYS_DEFAULT

    # Legacy behavior: infer horizon from suggested expiration when available.
    if not is_contract_prediction and not explicit_horizon:
        logged_exp = pred.get("suggested_expiration")
        if logged_exp:
            try:
                exp_dt = pd.Timestamp(logged_exp).normalize()
                delta_days = (exp_dt - pred_dt).days
                if 7 <= delta_days <= 120:  # sanity bounds
                    # Convert calendar days to approximate trading bars (~252/365 ratio)
                    effective_forward_days = max(5, int(delta_days * 252 / 365))
            except Exception:
                pass

    # Use first trading bar on/after prediction date.
    pos_candidates = np.where(close_series.index >= pred_dt)[0]
    if len(pos_candidates) == 0:
        return None
    start_pos = int(pos_candidates[0])
    end_pos = start_pos + effective_forward_days
    if end_pos >= len(close_series):
        return None  # not matured yet

    window = close_series.iloc[start_pos : end_pos + 1]
    entry = float(window.iloc[0])
    min_price = float(window.min())
    end_price = float(window.iloc[-1])

    # Contract-aware settlement: if the prediction logged a specific strike,
    # use it directly instead of the flat drop_threshold.
    suggested_strike = pred.get("suggested_strike")
    if suggested_strike is not None and float(suggested_strike) > 0:
        strike = float(suggested_strike)
        # outcome = 1 means profitable (not breached), 0 means assigned
        outcome = 1 if min_price >= strike else 0
    else:
        # Legacy: flat drop threshold
        max_drawdown = (min_price - entry) / entry
        outcome = 1 if max_drawdown > -drop_threshold else 0

    actual_return = (end_price - entry) / entry
    return outcome, actual_return


def main():
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    data = load_log(log_path)
    predictions = data.get("predictions", [])
    if not predictions:
        raise SystemExit("No predictions found in log.")

    # Default behavior is Schwab-only unless explicitly overridden.
    schwab_only = os.getenv("SCHWAB_ONLY", "1") == "1"
    if schwab_only and args.force_yahoo:
        raise SystemExit("SCHWAB_ONLY=1 is set; --force-yahoo is disabled.")

    use_schwab = not args.force_yahoo
    allow_yahoo_fallback = not schwab_only
    if use_schwab and SCHWAB_AVAILABLE and allow_yahoo_fallback:
        source = "Schwab API (Yahoo fallback)"
    elif use_schwab and SCHWAB_AVAILABLE:
        source = "Schwab API (fallback disabled)"
    else:
        source = "Yahoo Finance"
    print(f"Data source: {source}")

    by_ticker = {}
    for p in predictions:
        t = (p.get("ticker") or "").upper()
        if t:
            by_ticker[t] = True

    series_cache = {}
    for ticker in sorted(by_ticker.keys()):
        print(f"  Fetching {ticker}...", end=" ", flush=True)
        series_cache[ticker] = fetch_close_series(
            ticker,
            args.period,
            use_schwab=use_schwab,
            allow_yahoo_fallback=allow_yahoo_fallback,
        )
        n = len(series_cache[ticker]) if series_cache[ticker] is not None else 0
        print(f"{n} bars")

    updated = 0
    skipped_existing = 0
    skipped_unmatured = 0
    skipped_missing = 0

    now_iso = datetime.now().isoformat()
    for pred in predictions:
        if pred.get("outcome") is not None:
            skipped_existing += 1
            continue

        ticker = (pred.get("ticker") or "").upper()
        settled = settle_prediction(
            pred,
            series_cache.get(ticker),
            forward_days=args.forward_days,
            drop_threshold=args.drop_threshold,
        )

        if settled is None:
            # could be unmatured or no data
            if ticker not in series_cache or series_cache[ticker] is None:
                skipped_missing += 1
            else:
                skipped_unmatured += 1
            continue

        outcome, actual_return = settled
        pred["outcome"] = int(outcome)
        pred["outcome_date"] = now_iso
        pred["actual_return"] = float(actual_return)
        updated += 1

    print(f"\nPredictions total:    {len(predictions)}")
    print(f"Updated outcomes:     {updated}")
    print(f"Already settled:      {skipped_existing}")
    print(f"Unmatured/no-window:  {skipped_unmatured}")
    print(f"Missing ticker data:  {skipped_missing}")

    if args.dry_run:
        print("Dry run: no file changes written.")
        return

    if args.backup:
        backup_path = log_path.with_suffix(log_path.suffix + ".bak")
        shutil.copy2(log_path, backup_path)
        print(f"Backup written: {backup_path}")

    with log_path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Updated log written: {log_path}")


if __name__ == "__main__":
    main()
