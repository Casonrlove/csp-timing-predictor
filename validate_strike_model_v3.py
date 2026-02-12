"""
Validation script for V3 residual strike model.
Compares V3 vs V2 vs raw delta on out-of-sample data.
"""

import sys
sys.path.insert(0, '/home/cason/model')

import numpy as np
import pandas as pd
from data_collector import CSPDataCollector
from strike_probability import StrikeProbabilityCalculatorV2, StrikeProbabilityCalculatorV3
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_absolute_error

def calculate_strike_from_delta(delta, vol, T=35/365):
    """Calculate strike OTM % from delta using Black-Scholes"""
    try:
        strike_pct = vol * np.sqrt(T) * norm.ppf(1 - delta) * 100
        return max(1, min(30, strike_pct))
    except:
        return delta * 30

def run_validation(tickers, period='2y', forward_days=35):
    """
    Run comprehensive validation comparing V3, V2, and Delta.
    """
    
    # Try to load models
    try:
        v3_model = StrikeProbabilityCalculatorV3('strike_model_v3.pkl')
        has_v3 = v3_model.model_loaded
    except:
        has_v3 = False
        
    try:
        v2_model = StrikeProbabilityCalculatorV2('strike_model_v2.pkl')
        has_v2 = v2_model.model_loaded
    except:
        has_v2 = False

    print("="*70)
    print("STRIKE MODEL VALIDATION - V3 vs V2 vs Delta")
    print("="*70)
    print(f"V3 available: {has_v3}")
    print(f"V2 available: {has_v2}")
    print()

    if not (has_v3 or has_v2):
        print("ERROR: No models available to test")
        return

    # Collect results
    results = []
    
    for ticker in tickers:
        print(f"Validating {ticker}...")
        
        try:
            collector = CSPDataCollector(ticker, period=period)
            collector.fetch_data()
            collector.calculate_technical_indicators()
            df = collector.data.copy()

            # Calculate forward outcomes
            for idx in range(50, len(df) - forward_days, 10):  # Sample every 10 days
                row = df.iloc[idx:idx+1].copy()
                
                # Calculate forward breach drop (max drop within holding window)
                current = df['Close'].iloc[idx]
                future_window = df['Close'].iloc[idx:idx + forward_days + 1]
                min_future = future_window.min()
                forward_breach_drop = ((current - min_future) / current) * 100

                # Get IV
                rv = row['Volatility_20D'].iloc[0] / 100 if 'Volatility_20D' in row else 0.20
                iv_ratio = row['IV_RV_Ratio'].iloc[0] if 'IV_RV_Ratio' in row else 1.2
                vol = rv * max(1.0, min(iv_ratio, 2.5))

                # Test at different deltas
                for target_delta in [0.15, 0.25, 0.35]:
                    # Calculate strike for this delta
                    strike_pct = calculate_strike_from_delta(target_delta, vol)
                    
                    # Did the strike get breached anytime in the holding window?
                    breached_strike = 1 if forward_breach_drop >= strike_pct else 0

                    # Get predictions
                    delta_pred = target_delta
                    
                    v2_pred = v2_model.predict_for_delta(row, target_delta, ticker) if has_v2 else None
                    v3_pred = v3_model.predict_for_delta(row, target_delta, ticker) if has_v3 else None

                    results.append({
                        'ticker': ticker,
                        'delta': target_delta,
                        'actual': breached_strike,
                        'delta_pred': delta_pred,
                        'v2_pred': v2_pred,
                        'v3_pred': v3_pred,
                        'strike_pct': strike_pct,
                        'forward_drop': forward_breach_drop
                    })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print("ERROR: No results collected")
        return
    
    print(f"\nTotal samples: {len(df_results)}")
    print(f"Actual strike-breach rate: {df_results['actual'].mean()*100:.1f}%")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    
    actuals = df_results['actual'].values
    
    # Delta baseline
    delta_preds = df_results['delta_pred'].values
    delta_brier = brier_score_loss(actuals, delta_preds)
    delta_mae = mean_absolute_error(actuals, delta_preds)
    try:
        delta_auc = roc_auc_score(actuals, delta_preds)
    except:
        delta_auc = None
    
    print(f"\nDelta Baseline:")
    print(f"  Brier Score: {delta_brier:.4f}")
    print(f"  MAE: {delta_mae*100:.1f}%")
    if delta_auc:
        print(f"  ROC-AUC: {delta_auc:.4f}")
    
    # V2
    if has_v2:
        v2_mask = df_results['v2_pred'].notna()
        v2_preds = df_results.loc[v2_mask, 'v2_pred'].values
        v2_actuals = df_results.loc[v2_mask, 'actual'].values
        
        v2_brier = brier_score_loss(v2_actuals, v2_preds)
        v2_mae = mean_absolute_error(v2_actuals, v2_preds)
        try:
            v2_auc = roc_auc_score(v2_actuals, v2_preds)
        except:
            v2_auc = None
        
        print(f"\nV2 Model:")
        print(f"  Brier Score: {v2_brier:.4f} ({(delta_brier-v2_brier)/delta_brier*100:+.1f}% vs delta)")
        print(f"  MAE: {v2_mae*100:.1f}% ({(delta_mae-v2_mae)/delta_mae*100:+.1f}% vs delta)")
        if v2_auc:
            print(f"  ROC-AUC: {v2_auc:.4f} ({(v2_auc-delta_auc)/delta_auc*100:+.1f}% vs delta)")
    
    # V3
    if has_v3:
        v3_mask = df_results['v3_pred'].notna()
        v3_preds = df_results.loc[v3_mask, 'v3_pred'].values
        v3_actuals = df_results.loc[v3_mask, 'actual'].values
        
        v3_brier = brier_score_loss(v3_actuals, v3_preds)
        v3_mae = mean_absolute_error(v3_actuals, v3_preds)
        try:
            v3_auc = roc_auc_score(v3_actuals, v3_preds)
        except:
            v3_auc = None
        
        print(f"\nV3 Residual Model:")
        print(f"  Brier Score: {v3_brier:.4f} ({(delta_brier-v3_brier)/delta_brier*100:+.1f}% vs delta)")
        print(f"  MAE: {v3_mae*100:.1f}% ({(delta_mae-v3_mae)/delta_mae*100:+.1f}% vs delta)")
        if v3_auc:
            print(f"  ROC-AUC: {v3_auc:.4f} ({(v3_auc-delta_auc)/delta_auc*100:+.1f}% vs delta)")
        
        if has_v2:
            print(f"\nV3 vs V2:")
            print(f"  Brier improvement: {(v2_brier-v3_brier)/v2_brier*100:+.1f}%")
            print(f"  MAE improvement: {(v2_mae-v3_mae)/v2_mae*100:+.1f}%")
            if v2_auc and v3_auc:
                print(f"  AUC improvement: {(v3_auc-v2_auc)/v2_auc*100:+.1f}%")

    # Calibration by delta bucket
    print("\n" + "="*70)
    print("CALIBRATION BY DELTA LEVEL")
    print("="*70)
    print(f"{'Delta':>10} {'Count':>8} {'Actual':>10} {'Delta':>10} {'V2':>10} {'V3':>10}")
    print("-"*70)
    
    for delta in [0.15, 0.25, 0.35]:
        mask = df_results['delta'] == delta
        if mask.sum() < 5:
            continue
            
        subset = df_results[mask]
        actual = subset['actual'].mean()
        delta_avg = subset['delta_pred'].mean()
        v2_avg = subset['v2_pred'].mean() if has_v2 else None
        v3_avg = subset['v3_pred'].mean() if has_v3 else None
        
        print(f"{delta:>10.2f} {mask.sum():>8} {actual*100:>9.1f}% {delta_avg*100:>9.1f}%", end='')
        if v2_avg is not None:
            print(f" {v2_avg*100:>9.1f}%", end='')
        else:
            print(f" {'N/A':>10}", end='')
        if v3_avg is not None:
            print(f" {v3_avg*100:>9.1f}%")
        else:
            print(f" {'N/A':>10}")

    print("\n" + "="*70)
    print("EDGE SIGNAL ACCURACY")
    print("="*70)
    
    if has_v3:
        # V3 edge analysis
        df_results['v3_edge'] = df_results['delta'] - df_results['v3_pred']
        
        pos_edge = df_results['v3_edge'] > 0.02
        neg_edge = df_results['v3_edge'] < -0.02
        
        if pos_edge.sum() > 0:
            print(f"\nPositive Edge Signals (n={pos_edge.sum()}):")
            print(f"  Avg delta: {df_results.loc[pos_edge, 'delta'].mean()*100:.1f}%")
            print(f"  Avg V3 prob: {df_results.loc[pos_edge, 'v3_pred'].mean()*100:.1f}%")
            print(f"  Actual ITM: {df_results.loc[pos_edge, 'actual'].mean()*100:.1f}%")
            
        if neg_edge.sum() > 0:
            print(f"\nNegative Edge Signals (n={neg_edge.sum()}):")
            print(f"  Avg delta: {df_results.loc[neg_edge, 'delta'].mean()*100:.1f}%")
            print(f"  Avg V3 prob: {df_results.loc[neg_edge, 'v3_pred'].mean()*100:.1f}%")
            print(f"  Actual ITM: {df_results.loc[neg_edge, 'actual'].mean()*100:.1f}%")

if __name__ == "__main__":
    test_tickers = ['NVDA', 'AAPL', 'TSLA', 'AMD', 'SPY', 'META']
    run_validation(test_tickers, period='2y')
