"""
Temporal probability recalibration from settled prediction logs.

Fits an isotonic and Platt calibrator on historical predictions with known
outcomes, evaluates on a chronological holdout, and saves the best calibrator.
"""

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def _load_samples(log_path: str):
    with open(log_path, "r") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    rows = []
    for p in preds:
        outcome = p.get("outcome")
        p_safe = p.get("p_safe")
        ts = p.get("timestamp")
        if outcome is None or p_safe is None or ts is None:
            continue
        try:
            rows.append((datetime.fromisoformat(ts), float(p_safe), int(outcome)))
        except Exception:
            continue

    rows.sort(key=lambda x: x[0])
    if not rows:
        return np.array([]), np.array([])

    probs = np.array([r[1] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows], dtype=int)
    return probs, y


def _evaluate(name: str, y_true: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray):
    raw_brier = brier_score_loss(y_true, p_raw)
    cal_brier = brier_score_loss(y_true, p_cal)
    raw_logloss = log_loss(y_true, p_raw, labels=[0, 1])
    cal_logloss = log_loss(y_true, p_cal, labels=[0, 1])
    print(
        f"{name:8s}  Brier {raw_brier:.4f} -> {cal_brier:.4f}  "
        f"LogLoss {raw_logloss:.4f} -> {cal_logloss:.4f}"
    )
    return cal_brier, cal_logloss


def main():
    parser = argparse.ArgumentParser(description="Fit probability calibrator from settled prediction logs")
    parser.add_argument("--log", default="predictions_log.json", help="Path to predictions log")
    parser.add_argument("--output", default="timing_probability_calibrator.pkl", help="Output calibrator file")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum settled predictions required")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Chronological train fraction")
    args = parser.parse_args()

    probs, y = _load_samples(args.log)
    n = len(y)
    if n < args.min_samples:
        raise SystemExit(
            f"Not enough settled predictions: {n}. Need at least {args.min_samples}."
        )

    split = int(n * args.train_frac)
    split = min(max(split, 50), n - 20)

    p_train, y_train = probs[:split], y[:split]
    p_val, y_val = probs[split:], y[split:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise SystemExit("Need both classes in train and validation splits.")

    print(f"Samples: total={n}, train={len(y_train)}, val={len(y_val)}")

    # Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_train, y_train)
    p_iso = np.clip(iso.predict(p_val), 0.001, 0.999)
    iso_brier, _ = _evaluate("isotonic", y_val, p_val, p_iso)

    # Platt
    platt = LogisticRegression(max_iter=1000, random_state=42)
    platt.fit(p_train.reshape(-1, 1), y_train)
    p_platt = np.clip(platt.predict_proba(p_val.reshape(-1, 1))[:, 1], 0.001, 0.999)
    platt_brier, _ = _evaluate("platt", y_val, p_val, p_platt)

    if iso_brier <= platt_brier:
        method = "isotonic"
        calibrator = iso
        best_brier = iso_brier
    else:
        method = "platt"
        calibrator = platt
        best_brier = platt_brier

    out = {
        "calibrator": calibrator,
        "method": method,
        "fitted_at": datetime.now().isoformat(),
        "n_total": int(n),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "brier_raw_val": float(brier_score_loss(y_val, p_val)),
        "brier_cal_val": float(best_brier),
    }

    joblib.dump(out, args.output)
    print(f"Saved calibrator: {args.output} ({method})")


if __name__ == "__main__":
    main()
