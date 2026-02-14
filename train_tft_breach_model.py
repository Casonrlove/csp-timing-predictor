"""
Temporal Fusion Transformer — CSP Breach Probability Model
===========================================================
Trains a transformer-based model to predict P(breach) for a given delta bucket
using 60 days of technical indicator history as temporal context.

Why transformers saturate the GPU:
  - Self-attention is O(seq_len²) — at seq=60, each sample touches 3,600 pairs
  - Backward pass is ~2× the forward pass
  - Mixed-precision on tensor cores doubles throughput
  - Optuna searches 30 hyperparameter configurations

Expected runtime: 2–4 hours at 80–95% GPU utilization on RTX 5070 Ti

Usage:
    python train_tft_breach_model.py                    # full run
    python train_tft_breach_model.py --trials 5         # quick test
    python train_tft_breach_model.py --no-optuna        # single config, faster
    python train_tft_breach_model.py --epochs 20        # fewer epochs per trial
"""

import argparse
import os
import sys
import warnings
import time
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Tickers to train on (same groups as contract model + extras for more data)
# ---------------------------------------------------------------------------
TRAIN_TICKERS = [
    # high_vol group
    'TSLA', 'NVDA', 'AMD', 'MSTR', 'COIN', 'PLTR',
    # tech_growth group
    'META', 'GOOGL', 'AMZN', 'NFLX', 'CRM',
    # additional coverage
    'AAPL', 'MSFT', 'SHOP', 'SNOW', 'UBER', 'LYFT',
    # broad market for context
    'QQQ', 'SPY', 'IWM',
]

DELTA_BUCKETS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SEQ_LEN      = 60     # days of lookback (attention window)
FORWARD_DAYS = 35     # prediction horizon (same as contract model)

# Features to use as temporal input (subset of technical indicators)
TEMPORAL_FEATURES = [
    'Return_1D', 'Return_5D', 'Return_20D',
    'Volatility_20D', 'Volatility_10D',
    'RSI', 'MACD_Diff', 'BB_Position', 'BB_Width',
    'ATR_Pct', 'ADX',
    'Volume_Ratio',
    'Price_to_SMA20', 'Price_to_SMA50', 'Price_to_SMA200',
    'VIX', 'VIX_Rank', 'VIX_SMA_10', 'VIX_vs_SMA',
    'IV_RV_Ratio', 'VIX_Change_1D', 'VIX_Change_5D',
    'Return_ZScore', 'Return_Skew_20D', 'Return_Kurt_20D',
    'Drawdown_From_52W_High',
    'Consecutive_Down_Days',
    'Hurst_Exponent_60D',
    'Vol_Term_Structure',
    'Recent_Drop_1D', 'Recent_Drop_3D', 'Recent_Drop_5D',
    'Pullback_From_20D_High',
    'Return_Autocorr_20D',
    'VIX9D_Ratio',
    'Stock_vs_SPY_5D', 'Stock_vs_SPY_20D',
    'Sector_RS_5D', 'Sector_RS_20D',
    'Near_Earnings',
    'Days_To_Earnings',
]

# Static (non-temporal) features added at the prediction head
STATIC_FEATURES = ['target_delta', 'delta_x_vol', 'delta_x_vix', 'delta_x_iv_rank',
                   'delta_squared', 'delta_x_rsi', 'strike_otm_x_atr']

OUTPUT_PATH = 'tft_breach_model.pt'
SCALER_PATH = 'tft_breach_scaler.pkl'


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BreachDataset(Dataset):
    """
    Each sample: (temporal_seq, static_features, label)
      temporal_seq : (SEQ_LEN, n_temporal_features)
      static_feats : (n_static_features,)
      label        : float 0/1
    """

    def __init__(self, temporal_data, static_data, labels):
        self.temporal = torch.tensor(temporal_data, dtype=torch.float32)
        self.static   = torch.tensor(static_data,   dtype=torch.float32)
        self.labels   = torch.tensor(labels,         dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.temporal[idx], self.static[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TFTBreachModel(nn.Module):
    """
    Simplified Temporal Fusion Transformer for breach probability prediction.

    Architecture:
      1. Input projection: temporal features → d_model
      2. Positional encoding
      3. TransformerEncoder (multi-head self-attention × n_layers)
      4. Temporal pooling (mean + last token)
      5. Static feature projection
      6. Fusion MLP → sigmoid
    """

    def __init__(self, n_temporal: int, n_static: int,
                 d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_temporal, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Static feature branch
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        # Fusion head
        fusion_in = d_model * 2 + d_model // 2   # mean + last + static
        self.head = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, temporal, static):
        # temporal : (B, SEQ_LEN, n_temporal)
        # static   : (B, n_static)
        x = self.input_proj(temporal)       # (B, T, d_model)
        x = self.pos_enc(x)                 # (B, T, d_model)
        x = self.transformer(x)             # (B, T, d_model)

        mean_pool = x.mean(dim=1)           # (B, d_model)
        last_tok  = x[:, -1, :]             # (B, d_model)  — most recent day
        static_e  = self.static_proj(static) # (B, d_model//2)

        fused = torch.cat([mean_pool, last_tok, static_e], dim=-1)
        return self.head(fused).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_all_data(period: str = '5y') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect data for all training tickers, create sequences, return arrays.
    Returns (temporal_seqs, static_feats, labels) as numpy arrays.
    """
    from data_collector import CSPDataCollector

    all_temporal, all_static, all_labels = [], [], []
    feature_cols = None

    for ticker in TRAIN_TICKERS:
        try:
            print(f"  [{ticker}] fetching {period} data...", end=' ', flush=True)
            t0 = time.perf_counter()
            collector = CSPDataCollector(ticker, period=period)
            collector.fetch_data()
            collector.calculate_technical_indicators()
            df_raw = collector.create_contract_targets()
            elapsed = time.perf_counter() - t0
            print(f"{len(df_raw)} rows ({elapsed:.1f}s)")

            if df_raw is None or len(df_raw) < SEQ_LEN + FORWARD_DAYS + 10:
                print(f"    Skipping — not enough data")
                continue

            # Identify available temporal features
            avail_temporal = [f for f in TEMPORAL_FEATURES if f in df_raw.columns]
            avail_static   = [f for f in STATIC_FEATURES   if f in df_raw.columns]
            if feature_cols is None:
                feature_cols = (avail_temporal, avail_static)
            else:
                avail_temporal = feature_cols[0]
                avail_static   = feature_cols[1]

            # Pivot so we have one row per (date, delta)
            # df_raw already has one row per (date, delta)
            df_raw = df_raw.sort_values(['Date', 'target_delta'] if 'Date' in df_raw.columns else ['target_delta'])
            dates = df_raw.index.unique() if 'Date' not in df_raw.columns else df_raw['Date'].unique()

            # Build sequences: for each (date_idx, delta), take SEQ_LEN days of history
            # We use a single-delta slice of df_raw for temporal features (same for all deltas)
            delta_0 = DELTA_BUCKETS[0]
            df_single = df_raw[df_raw['target_delta'] == delta_0].copy()
            df_single = df_single.reset_index(drop=True)

            temporal_arr = df_single[avail_temporal].fillna(0).values  # (n_dates, n_feats)
            n_dates = len(df_single)

            seqs_t, seqs_s, lbls = [], [], []
            for i in range(SEQ_LEN, n_dates):
                seq = temporal_arr[i - SEQ_LEN:i]   # (SEQ_LEN, n_feats)
                # For each delta bucket at this date
                row_slice = df_raw[df_raw['target_delta'].isin(DELTA_BUCKETS)].iloc[
                    i * len(DELTA_BUCKETS):(i + 1) * len(DELTA_BUCKETS)
                ] if 'target_delta' in df_raw.columns else pd.DataFrame()

                for delta in DELTA_BUCKETS:
                    if 'target_delta' in df_raw.columns:
                        rows = df_raw[df_raw['target_delta'] == delta]
                        if i >= len(rows):
                            continue
                        row = rows.iloc[i]
                        label = float(row.get('strike_breached', 0))
                        static_vals = [float(row.get(f, 0)) for f in avail_static]
                    else:
                        label = 0.0
                        static_vals = [delta, 0, 0, 0, delta**2, 0, 0][:len(avail_static)]

                    seqs_t.append(seq)
                    seqs_s.append(static_vals)
                    lbls.append(label)

            if seqs_t:
                all_temporal.extend(seqs_t)
                all_static.extend(seqs_s)
                all_labels.extend(lbls)
                print(f"    Added {len(seqs_t):,} sequences")

        except Exception as e:
            print(f"  [{ticker}] ERROR: {e}")
            continue

    if not all_temporal:
        raise RuntimeError("No training data collected!")

    return (np.array(all_temporal, dtype=np.float32),
            np.array(all_static,   dtype=np.float32),
            np.array(all_labels,   dtype=np.float32),
            feature_cols)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler_amp, device):
    model.train()
    total_loss = 0.0
    n = 0
    for temporal, static, labels in loader:
        temporal = temporal.to(device, non_blocking=True)
        static   = static.to(device,   non_blocking=True)
        labels   = labels.to(device,   non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(temporal, static)
            loss   = F.binary_cross_entropy_with_logits(logits, labels)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for temporal, static, labels in loader:
        temporal = temporal.to(device, non_blocking=True)
        static   = static.to(device,   non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(temporal, static)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    try:
        auc   = roc_auc_score(labels, preds)
        brier = brier_score_loss(labels, preds)
    except Exception:
        auc, brier = 0.5, 0.25
    return auc, brier


def run_trial(params, train_loader, val_loader, n_temporal, n_static,
              n_epochs, device, trial_num=0):
    """Train one model configuration and return validation AUC."""
    model = TFTBreachModel(
        n_temporal = n_temporal,
        n_static   = n_static,
        d_model    = params['d_model'],
        n_heads    = params['n_heads'],
        n_layers   = params['n_layers'],
        d_ff       = params['d_ff'],
        dropout    = params['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model params: {n_params/1e6:.2f}M  |  "
          f"d_model={params['d_model']} heads={params['n_heads']} "
          f"layers={params['n_layers']} d_ff={params['d_ff']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params['lr'], weight_decay=params['wd']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    amp_scaler = torch.cuda.amp.GradScaler()

    best_auc = 0.0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, amp_scaler, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == n_epochs:
            val_auc, val_brier = eval_epoch(model, val_loader, device)
            elapsed = time.perf_counter() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch:3d}/{n_epochs}  loss={train_loss:.4f}  "
                  f"val_AUC={val_auc:.4f}  brier={val_brier:.4f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.1f}s)")
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_auc, best_state, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train TFT Breach Probability Model')
    parser.add_argument('--period',    type=str, default='5y',
                        help='Historical data period (e.g. 3y, 5y, 10y)')
    parser.add_argument('--epochs',    type=int, default=60,
                        help='Epochs per Optuna trial')
    parser.add_argument('--trials',    type=int, default=30,
                        help='Number of Optuna hyperparameter trials')
    parser.add_argument('--no-optuna', action='store_true',
                        help='Skip Optuna, use default hyperparameters')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--no-gpu',    action='store_true')
    args = parser.parse_args()

    # Device
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({total_vram:.0f}GB VRAM)")
        # Enable TF32 for better performance on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\nConfig: {args.epochs} epochs/trial × {args.trials} trials = "
          f"up to {args.epochs * args.trials} total epochs")
    print(f"Batch size: {args.batch_size} | Period: {args.period}\n")

    # -----------------------------------------------------------------------
    # 1. Data collection
    # -----------------------------------------------------------------------
    print(f"Collecting data for {len(TRAIN_TICKERS)} tickers...")
    temporal_arr, static_arr, labels, feature_cols = collect_all_data(args.period)
    avail_temporal, avail_static = feature_cols
    n_temporal = len(avail_temporal)
    n_static   = len(avail_static)

    print(f"\nDataset: {len(labels):,} sequences  "
          f"({n_temporal} temporal features, {n_static} static features)")
    print(f"Breach rate: {labels.mean()*100:.1f}%")

    # -----------------------------------------------------------------------
    # 2. Normalize temporal features
    # -----------------------------------------------------------------------
    B, T, F_ = temporal_arr.shape
    scaler = StandardScaler()
    temporal_flat = temporal_arr.reshape(-1, F_)
    temporal_norm = scaler.fit_transform(temporal_flat).reshape(B, T, F_).astype(np.float32)
    np.nan_to_num(temporal_norm, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)

    # Normalize static features
    static_scaler = StandardScaler()
    static_norm = static_scaler.fit_transform(static_arr).astype(np.float32)
    np.nan_to_num(static_norm, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)

    # -----------------------------------------------------------------------
    # 3. Train / val split (temporal — last 15% as val)
    # -----------------------------------------------------------------------
    n_val = int(len(labels) * 0.15)
    idx_split = len(labels) - n_val

    train_ds = BreachDataset(temporal_norm[:idx_split], static_norm[:idx_split], labels[:idx_split])
    val_ds   = BreachDataset(temporal_norm[idx_split:], static_norm[idx_split:], labels[idx_split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")
    print(f"Steps/epoch: {len(train_loader):,}\n")

    # -----------------------------------------------------------------------
    # 4. Hyperparameter optimization with Optuna
    # -----------------------------------------------------------------------
    best_overall_auc = 0.0
    best_overall_state = None
    best_params = None

    if args.no_optuna:
        search_space = [{
            'd_model': 256, 'n_heads': 8, 'n_layers': 4, 'd_ff': 1024,
            'dropout': 0.1, 'lr': 3e-4, 'wd': 1e-4,
        }]
    else:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                nonlocal best_overall_auc, best_overall_state, best_params
                params = {
                    'd_model': trial.suggest_categorical('d_model', [128, 256, 512]),
                    'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
                    'n_layers': trial.suggest_int('n_layers', 2, 8),
                    'd_ff':    trial.suggest_categorical('d_ff', [512, 1024, 2048]),
                    'dropout': trial.suggest_float('dropout', 0.05, 0.3),
                    'lr':      trial.suggest_float('lr', 1e-5, 5e-4, log=True),
                    'wd':      trial.suggest_float('wd', 1e-5, 1e-2, log=True),
                }
                # Enforce n_heads divides d_model
                if params['d_model'] % params['n_heads'] != 0:
                    params['n_heads'] = 8 if params['d_model'] >= 256 else 4

                print(f"\n--- Optuna Trial {trial.number + 1}/{args.trials} ---")
                auc, state, _ = run_trial(
                    params, train_loader, val_loader, n_temporal, n_static,
                    args.epochs, device, trial.number
                )
                if auc > best_overall_auc:
                    best_overall_auc   = auc
                    best_overall_state = state
                    best_params        = params
                    print(f"    *** New best AUC: {auc:.4f} ***")
                return auc

            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

            print(f"\nOptuna complete. Best AUC: {best_overall_auc:.4f}")
            print(f"Best params: {best_params}")
            search_space = []  # already done

        except ImportError:
            print("optuna not installed — using default hyperparameters")
            search_space = [{
                'd_model': 256, 'n_heads': 8, 'n_layers': 4, 'd_ff': 1024,
                'dropout': 0.1, 'lr': 3e-4, 'wd': 1e-4,
            }]

    # Fallback if Optuna wasn't used
    for i, params in enumerate(search_space):
        print(f"\n--- Config {i + 1}/{len(search_space)} ---")
        auc, state, _ = run_trial(
            params, train_loader, val_loader, n_temporal, n_static,
            args.epochs, device, i
        )
        if auc > best_overall_auc:
            best_overall_auc   = auc
            best_overall_state = state
            best_params        = params

    # -----------------------------------------------------------------------
    # 5. Save best model
    # -----------------------------------------------------------------------
    if best_overall_state is None:
        print("ERROR: No model was trained successfully.")
        sys.exit(1)

    # Build final model with best params and load best weights
    final_model = TFTBreachModel(
        n_temporal = n_temporal,
        n_static   = n_static,
        **{k: v for k, v in best_params.items() if k not in ('lr', 'wd')}
    )
    final_model.load_state_dict(best_overall_state)

    torch.save({
        'model_state':     best_overall_state,
        'model_config': {
            'n_temporal': n_temporal,
            'n_static':   n_static,
            **{k: v for k, v in best_params.items() if k not in ('lr', 'wd')},
        },
        'temporal_features': avail_temporal,
        'static_features':   avail_static,
        'val_auc':           best_overall_auc,
        'seq_len':           SEQ_LEN,
        'delta_buckets':     DELTA_BUCKETS,
        'trained_tickers':   TRAIN_TICKERS,
        'trained_at':        datetime.now().isoformat(),
    }, OUTPUT_PATH)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'temporal': scaler, 'static': static_scaler}, f)

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Best val AUC: {best_overall_auc:.4f}")
    print(f"  Model saved → {OUTPUT_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")
    print(f"{'=' * 60}\n")

    # Final GPU stats
    if device.type == 'cuda':
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM used: {peak_gb:.2f}GB")


if __name__ == '__main__':
    main()
