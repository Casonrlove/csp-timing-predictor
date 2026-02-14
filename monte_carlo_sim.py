"""
Monte Carlo CSP Simulator — GPU-accelerated GARCH(1,1) price path simulation.

Standalone script. Does NOT integrate with the API server.

Usage:
    python monte_carlo_sim.py NVDA
    python monte_carlo_sim.py NVDA --paths 10000000
    python monte_carlo_sim.py NVDA --no-gpu
    python monte_carlo_sim.py NVDA --save-only
    python monte_carlo_sim.py NVDA --forward-days 45
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants (mirror simple_api_server.py)
# ---------------------------------------------------------------------------
DELTA_BUCKETS   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
FORWARD_DAYS    = 35
ASSIGNMENT_LOSS = 0.05          # 5% of strike assumed as loss if assigned
GARCH_FIT_DAYS  = 252           # fit window for GARCH parameters
SAMPLE_PATHS    = 200           # paths to draw in spaghetti chart


# ---------------------------------------------------------------------------
# GARCH(1,1) helpers — CPU-only (one-time fit)
# ---------------------------------------------------------------------------

def _garch_nll(params, returns):
    """Negative log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns)
    var = np.full(n, np.var(returns))
    nll = 0.0
    for t in range(1, n):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
        if var[t] <= 0:
            return 1e10
        nll += 0.5 * (np.log(var[t]) + returns[t] ** 2 / var[t])
    return nll


def fit_garch(returns):
    """
    Fit GARCH(1,1) to an array of log returns.
    Returns (omega, alpha, beta, sigma2_0).
    Falls back to constant-vol if fit fails.
    """
    rv = float(np.var(returns))
    x0 = [rv * 0.05, 0.10, 0.85]
    bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.9999)]

    # Try arch library first (faster, more robust)
    try:
        from arch import arch_model
        am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        o = float(res.params['omega']) / 10000
        a = float(res.params['alpha[1]'])
        b = float(res.params['beta[1]'])
        if o > 0 and a >= 0 and b >= 0 and (a + b) < 1:
            s2_0 = o / max(1e-8, 1 - a - b)
            return o, a, b, s2_0
    except Exception:
        pass

    # Fallback: scipy minimize
    result = minimize(_garch_nll, x0, args=(returns,), method='L-BFGS-B', bounds=bounds)
    if result.success:
        omega, alpha, beta = result.x
        s2_0 = omega / max(1e-8, 1 - alpha - beta)
        return omega, alpha, beta, s2_0

    # Last resort: constant vol
    s2_0 = rv
    return s2_0 * 0.01, 0.10, 0.85, s2_0


# ---------------------------------------------------------------------------
# GPU Monte Carlo simulation
# ---------------------------------------------------------------------------

def run_monte_carlo(current_price, omega, alpha, beta, sigma2_0,
                    n_paths, n_days, device):
    """
    Simulate n_paths price paths of n_days using GARCH(1,1) on GPU.

    All n_paths × n_days standard normals are pre-generated in one GPU call
    so the memory bus is fully saturated. The GARCH time loop is then a series
    of large vectorised operations rather than many small kernel launches.

    Returns:
        S_final : (n_paths,) tensor — terminal prices
        S_min   : (n_paths,) tensor — path minimums (for breach detection)
        S_paths : (SAMPLE_PATHS, n_days+1) tensor — sampled paths for plotting
    """
    omega_t = torch.tensor(omega, device=device, dtype=torch.float32)
    alpha_t = torch.tensor(alpha, device=device, dtype=torch.float32)
    beta_t  = torch.tensor(beta,  device=device, dtype=torch.float32)

    # Pre-generate ALL random shocks at once — fills the GPU memory bus
    # Shape: (n_days, n_paths)  — axis 0 = time, axis 1 = path
    z_all = torch.randn(n_days, n_paths, device=device, dtype=torch.float32)

    S     = torch.full((n_paths,), float(current_price), device=device, dtype=torch.float32)
    var   = torch.full((n_paths,), float(sigma2_0),      device=device, dtype=torch.float32)
    S_min = S.clone()

    # Store sample paths for chart
    path_buf = [S[:SAMPLE_PATHS].clone().cpu()]

    for t in range(n_days):
        sigma = torch.sqrt(var)                        # (n_paths,)
        eps   = z_all[t] * sigma                       # scaled shock
        S     = S * torch.exp(eps - 0.5 * var)         # log-normal step
        var   = (omega_t + alpha_t * eps ** 2 + beta_t * var).clamp_(min=1e-8)
        S_min = torch.minimum(S_min, S)
        path_buf.append(S[:SAMPLE_PATHS].clone().cpu())

    S_paths = torch.stack(path_buf, dim=1)  # (SAMPLE_PATHS, n_days+1)
    return S, S_min, S_paths


# ---------------------------------------------------------------------------
# Strike estimation (mirrors simple_api_server.py lines 851-854)
# ---------------------------------------------------------------------------

def delta_to_strike(current_price, delta, vol_est, forward_days):
    T = forward_days / 365.0
    otm = vol_est * np.sqrt(T) * norm.ppf(1 - delta)
    otm = float(np.clip(otm, 0.01, 0.30))
    return current_price * (1 - otm), otm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GPU Monte Carlo CSP Simulator')
    parser.add_argument('ticker',            type=str)
    parser.add_argument('--paths',           type=int,   default=5_000_000)
    parser.add_argument('--forward-days',    type=int,   default=FORWARD_DAYS)
    parser.add_argument('--no-gpu',          action='store_true')
    parser.add_argument('--save-only',       action='store_true',
                        help='Save chart without displaying it')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    n_paths = args.paths
    fwd = args.forward_days

    # -----------------------------------------------------------------------
    # 1. Device
    # -----------------------------------------------------------------------
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"Using CPU")
    else:
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")

    # -----------------------------------------------------------------------
    # 2. Data — Schwab via CSPDataCollector
    # -----------------------------------------------------------------------
    print(f"\nFetching data for {ticker}...")
    try:
        from data_collector import CSPDataCollector
        collector = CSPDataCollector(ticker, period='1y')
        collector.fetch_data()
        collector.calculate_technical_indicators()
        df = collector.data
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    last = df.iloc[-1]
    current_price = float(last['Close'])

    # Volatility inputs (same as simple_api_server.py lines 851-854)
    rv_val   = max(float(last.get('Volatility_20D', 20.0)) / 100.0, 0.01)
    iv_rr    = max(min(float(last.get('IV_RV_Ratio', 1.2)), 2.5), 1.0)
    vol_est  = rv_val * iv_rr
    vix_val  = float(last.get('VIX', 20.0))
    iv_rank  = float(last.get('VIX_Rank', 50.0))

    print(f"  Price: ${current_price:.2f} | RV(20D): {rv_val*100:.1f}% | "
          f"IV/RV: {iv_rr:.2f}x | Vol Est: {vol_est*100:.1f}% | VIX: {vix_val:.1f}")

    # -----------------------------------------------------------------------
    # 3. Live premiums from Schwab
    # -----------------------------------------------------------------------
    live_premiums = {}   # delta -> {'premium', 'bid', 'ask', 'strike', 'dte', 'expiration'}
    try:
        from schwab_client import get_csp_options_schwab
        print(f"  Fetching live options from Schwab...")
        options = get_csp_options_schwab(
            ticker, min_delta=0.08, max_delta=0.55,
            min_dte=20, max_dte=55, monthlies_only=True
        )
        # Map each delta bucket to the closest real option
        for delta in DELTA_BUCKETS:
            closest = min(
                options,
                key=lambda o: abs(abs(o.get('delta', 0)) - delta),
                default=None
            )
            if closest and abs(abs(closest.get('delta', 0)) - delta) < 0.08:
                live_premiums[delta] = {
                    'premium':    float(closest.get('premium', 0)),
                    'bid':        float(closest.get('bid', 0)),
                    'ask':        float(closest.get('ask', 0)),
                    'strike':     float(closest.get('strike', 0)),
                    'dte':        int(closest.get('dte', fwd)),
                    'expiration': closest.get('expiration', ''),
                }
        print(f"  Got {len(live_premiums)} live options matched to delta buckets")
    except Exception as e:
        print(f"  Schwab options unavailable ({e}) — using estimated premiums")

    # -----------------------------------------------------------------------
    # 4. GARCH(1,1) fit on recent returns
    # -----------------------------------------------------------------------
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna().values
    fit_returns = log_returns[-GARCH_FIT_DAYS:]
    print(f"\nFitting GARCH(1,1) on {len(fit_returns)} days of returns...")
    omega, alpha, beta, sigma2_0 = fit_garch(fit_returns)
    persistence = alpha + beta
    print(f"  ω={omega:.2e}  α={alpha:.4f}  β={beta:.4f}  "
          f"persistence={persistence:.4f}  σ_0={np.sqrt(sigma2_0)*100:.2f}%/day")

    # -----------------------------------------------------------------------
    # 5. GPU simulation
    # -----------------------------------------------------------------------
    vram_gb = n_paths * fwd * 4 / 1e9  # float32 = 4 bytes
    print(f"\nRunning {n_paths:,} paths × {fwd} days on {device}  "
          f"(~{vram_gb:.1f} GB VRAM for random tensor)...")
    with torch.no_grad():
        S_final, S_min, S_paths = run_monte_carlo(
            current_price, omega, alpha, beta, sigma2_0,
            n_paths, fwd, device
        )
    S_final_np = S_final.cpu().numpy()
    S_min_np   = S_min.cpu().numpy()
    S_paths_np = S_paths.numpy()  # already on CPU

    # -----------------------------------------------------------------------
    # 6. Compute breach probs + EV per delta bucket
    # -----------------------------------------------------------------------
    results = []
    for delta in DELTA_BUCKETS:
        # Strike from our vol_est (same formula as contract model)
        mc_strike, otm_pct = delta_to_strike(current_price, delta, vol_est, fwd)

        # Use live Schwab strike if available
        lp = live_premiums.get(delta)
        if lp and lp['strike'] > 0:
            actual_strike = lp['strike']
            premium = lp['premium']
            dte = lp['dte']
            exp_label = lp['expiration']
        else:
            actual_strike = mc_strike
            # Rough premium estimate: delta × 1% of strike × sqrt(DTE/35)
            premium = actual_strike * delta * 0.01 * np.sqrt(fwd / 35)
            dte = fwd
            exp_label = f'~{fwd}d'

        # MC breach probability
        p_breach_mc = float((S_min_np < actual_strike).mean())

        # EV
        slippage = (lp['ask'] - lp['bid']) * 0.25 if lp else 0.0
        ev = premium - p_breach_mc * actual_strike * ASSIGNMENT_LOSS - slippage

        results.append({
            'delta':      delta,
            'strike':     actual_strike,
            'otm_pct':    otm_pct * 100,
            'p_breach':   p_breach_mc,
            'premium':    premium,
            'ev':         ev,
            'dte':        dte,
            'expiration': exp_label,
        })

    # -----------------------------------------------------------------------
    # 7. Print table
    # -----------------------------------------------------------------------
    sep = '─' * 80
    print(f'\n{sep}')
    print(f'  {ticker} Monte Carlo CSP Analysis')
    print(f'  {n_paths:,} paths | GARCH(1,1) β={beta:.3f} | {fwd}d horizon | {device}')
    print(f'  Price: ${current_price:.2f}  Vol Est: {vol_est*100:.1f}%  '
          f'VIX: {vix_val:.1f}  IV Rank: {iv_rank:.0f}')
    print(sep)
    header = f"  {'Delta':>6}  {'Strike':>8}  {'OTM%':>5}  "
    header += f"{'P(Breach)':>10}  {'Premium':>8}  {'EV/share':>9}  {'Exp':>12}"
    print(header)
    print(sep)

    best_ev_row = max(results, key=lambda r: r['ev'])
    for r in results:
        marker = ' ◄ best EV' if r is best_ev_row else ''
        ev_color = '+' if r['ev'] >= 0 else ''
        print(
            f"  {r['delta']:>6.2f}  ${r['strike']:>7.2f}  {r['otm_pct']:>4.1f}%  "
            f"  {r['p_breach']*100:>8.1f}%  ${r['premium']:>7.2f}  "
            f"  {ev_color}${r['ev']:>7.3f}  {r['expiration']:>12}{marker}"
        )
    print(sep)

    pct10  = float(np.percentile(S_final_np, 10))
    pct25  = float(np.percentile(S_final_np, 25))
    median = float(np.median(S_final_np))
    pct75  = float(np.percentile(S_final_np, 75))
    pct90  = float(np.percentile(S_final_np, 90))
    print(f'\n  Terminal price distribution (day {fwd}):')
    print(f'  P10=${pct10:.2f}  P25=${pct25:.2f}  Median=${median:.2f}  '
          f'P75=${pct75:.2f}  P90=${pct90:.2f}')
    print(sep + '\n')

    # -----------------------------------------------------------------------
    # 8. Charts
    # -----------------------------------------------------------------------
    os.makedirs('mc_output', exist_ok=True)
    date_str   = datetime.now().strftime('%Y%m%d_%H%M')
    chart_path = f'mc_output/{ticker}_{date_str}.png'

    fig = plt.figure(figsize=(16, 11), facecolor='#0d1117')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    title_color  = '#e2e8f0'
    label_color  = '#8b949e'
    green        = '#22c55e'
    red          = '#ef4444'
    purple       = '#805ad5'
    blue         = '#3b82f6'

    def _style(ax):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors=label_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # --- Chart 1: Terminal price distribution ---
    _style(ax1)
    ax1.hist(S_final_np, bins=120, color=blue, alpha=0.7, edgecolor='none')
    for r in results:
        color = green if r['ev'] >= 0 else red
        ax1.axvline(r['strike'], color=color, lw=0.8, alpha=0.7,
                    label=f"Δ{r['delta']:.2f} ${r['strike']:.0f}")
    ax1.axvline(current_price, color='white', lw=1.2, linestyle='--', label='Current')
    ax1.set_title(f'{ticker} Terminal Price Distribution (Day {fwd})',
                  color=title_color, fontsize=10, pad=8)
    ax1.set_xlabel('Price', color=label_color, fontsize=8)
    ax1.set_ylabel('Frequency', color=label_color, fontsize=8)
    ax1.legend(fontsize=6, facecolor='#161b22', labelcolor=label_color,
               edgecolor='#30363d', ncol=2)

    # --- Chart 2: P(Breach) by delta ---
    _style(ax2)
    deltas   = [r['delta'] for r in results]
    pbreaches = [r['p_breach'] * 100 for r in results]
    # Theoretical (BSM approximation = delta itself)
    bsm_approx = [d * 100 for d in deltas]
    x = np.arange(len(deltas))
    w = 0.35
    ax2.bar(x - w/2, pbreaches,   width=w, color=purple, label='MC GARCH', alpha=0.85)
    ax2.bar(x + w/2, bsm_approx,  width=w, color=blue,   label='BSM (Δ)',  alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{d:.2f}' for d in deltas], fontsize=7)
    ax2.set_title('P(Breach): MC GARCH vs Black-Scholes',
                  color=title_color, fontsize=10, pad=8)
    ax2.set_xlabel('Delta', color=label_color, fontsize=8)
    ax2.set_ylabel('P(Breach) %', color=label_color, fontsize=8)
    ax2.legend(fontsize=8, facecolor='#161b22', labelcolor=label_color, edgecolor='#30363d')

    # --- Chart 3: Sample price paths ---
    _style(ax3)
    days_x = np.arange(fwd + 1)
    for i in range(min(SAMPLE_PATHS, 150)):
        ax3.plot(days_x, S_paths_np[i], color=blue, alpha=0.08, lw=0.5)
    # Highlight best-EV strike as a horizontal line
    ax3.axhline(best_ev_row['strike'], color=green, lw=1.2, linestyle='--',
                label=f"Best EV strike ${best_ev_row['strike']:.0f}")
    ax3.axhline(current_price, color='white', lw=1.0, linestyle=':',
                label=f"Current ${current_price:.2f}")
    ax3.set_title(f'{SAMPLE_PATHS} Sample Price Paths (GARCH)',
                  color=title_color, fontsize=10, pad=8)
    ax3.set_xlabel('Trading Days', color=label_color, fontsize=8)
    ax3.set_ylabel('Price', color=label_color, fontsize=8)
    ax3.legend(fontsize=8, facecolor='#161b22', labelcolor=label_color, edgecolor='#30363d')

    # --- Chart 4: EV curve ---
    _style(ax4)
    strikes_plot = [r['strike'] for r in results]
    evs_plot     = [r['ev'] for r in results]
    ax4.plot(strikes_plot, evs_plot, color=purple, lw=2, marker='o', markersize=5)
    ax4.axhline(0, color=label_color, lw=0.8, linestyle='--')
    ax4.fill_between(strikes_plot, evs_plot, 0,
                     where=[e >= 0 for e in evs_plot], alpha=0.2, color=green)
    ax4.fill_between(strikes_plot, evs_plot, 0,
                     where=[e < 0 for e in evs_plot],  alpha=0.2, color=red)
    best_s = best_ev_row['strike']
    best_e = best_ev_row['ev']
    ax4.annotate(f"  Best EV\n  ${best_e:.3f}/share",
                 xy=(best_s, best_e), color=green, fontsize=8)
    ax4.set_title('Expected Value per Strike', color=title_color, fontsize=10, pad=8)
    ax4.set_xlabel('Strike', color=label_color, fontsize=8)
    ax4.set_ylabel('EV per Share ($)', color=label_color, fontsize=8)

    fig.suptitle(
        f'{ticker}  |  Monte Carlo GARCH(1,1)  |  {n_paths:,} paths  |  '
        f'{fwd}d  |  {datetime.now().strftime("%Y-%m-%d")}',
        color=title_color, fontsize=12, y=0.98
    )

    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f'Chart saved → {chart_path}')

    if not args.save_only:
        try:
            matplotlib.use('TkAgg')
            plt.show()
        except Exception:
            print('(Cannot display chart in this environment — chart saved to file)')

    plt.close(fig)


if __name__ == '__main__':
    main()
