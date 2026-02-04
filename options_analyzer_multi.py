"""
Get multiple CSP options at different delta levels
Allows user to choose their preferred strike/delta
"""

import yfinance as yf
from datetime import datetime
from advanced_options_pricing import get_accurate_greeks
import pandas as pd


def get_all_csp_options(ticker, target_dte_min=30, target_dte_max=45,
                        min_delta=0.10, max_delta=0.40, debug=False):
    """
    Get ALL suitable CSP options within delta range

    Args:
        ticker: Stock symbol
        target_dte_min: Minimum DTE
        target_dte_max: Maximum DTE
        min_delta: Minimum delta (e.g., 0.10 for 10 delta)
        max_delta: Maximum delta (e.g., 0.40 for 40 delta)
        debug: Print debug information

    Returns:
        List of options sorted by delta (closest to 0.30 first)
    """
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]

        if debug:
            print(f'[DEBUG] Current price: {current_price:.2f}')

        expirations = stock.options
        if not expirations:
            return []

        all_options = []

        if debug:
            print(f'[DEBUG] Total expirations: {len(expirations)}')

        for exp_date in expirations:
            # Calculate DTE
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            dte = (exp_datetime - datetime.now()).days

            # Skip if outside DTE range
            if dte < target_dte_min or dte > target_dte_max:
                continue

            if debug:
                print(f'[DEBUG] Checking {exp_date} (DTE={dte})')

            try:
                opt_chain = stock.option_chain(exp_date)
                puts = opt_chain.puts

                # Filter for OTM puts only (strike < stock price)
                puts = puts[puts['strike'] < current_price]
                puts = puts[puts['strike'] >= current_price * 0.80]  # Not too far OTM

                if debug:
                    print(f'[DEBUG]   Found {len(puts)} OTM puts in strike range')

                for _, put in puts.iterrows():
                    strike = put['strike']

                    # Skip if no volume/OI
                    volume = int(put['volume']) if pd.notna(put['volume']) else 0
                    oi = int(put['openInterest']) if pd.notna(put['openInterest']) else 0

                    if volume == 0 and oi == 0:
                        if debug:
                            print(f'[DEBUG]     Strike {strike:.2f}: Skipped (no volume/OI)')
                        continue

                    # Get accurate Greeks
                    greeks = get_accurate_greeks(ticker, strike, exp_date, 'put', use_binomial=True)

                    if not greeks:
                        if debug:
                            print(f'[DEBUG]     Strike {strike:.2f}: Skipped (greeks calculation failed)')
                        continue

                    delta = greeks['delta']
                    delta_magnitude = abs(delta)

                    # Check if delta is in range
                    if delta_magnitude < min_delta or delta_magnitude > max_delta:
                        if debug:
                            print(f'[DEBUG]     Strike {strike:.2f}: Skipped (delta {delta_magnitude:.3f} outside range {min_delta}-{max_delta})')
                        continue

                    if debug:
                        print(f'[DEBUG]     Strike {strike:.2f}: ACCEPTED (delta={delta_magnitude:.3f}, premium={greeks["market_price"]:.2f})')

                    premium = greeks['market_price']

                    if premium <= 0:
                        continue

                    # Calculate metrics
                    ror = (premium / strike) * 100
                    annualized_ror = ror * (365 / dte)
                    theta_efficiency = abs(greeks['theta'] / strike) * 100

                    # Distance from ideal (0.30 delta, 37.5 DTE)
                    delta_score = abs(delta_magnitude - 0.30)
                    dte_score = abs(dte - 37.5) / 37.5

                    all_options.append({
                        'ticker': ticker,
                        'strike': float(strike),
                        'premium': float(premium),
                        'bid': float(greeks['bid']),
                        'ask': float(greeks['ask']),
                        'delta': round(delta, 4),
                        'gamma': round(greeks['gamma'], 5),
                        'theta': round(greeks['theta'], 4),
                        'vega': round(greeks['vega'], 4),
                        'implied_volatility': round(greeks['implied_volatility'], 4),
                        'volume': volume,
                        'open_interest': oi,
                        'dte': dte,
                        'expiration': exp_date,
                        'ror': round(ror, 2),
                        'annualized_ror': round(annualized_ror, 2),
                        'theta_efficiency': round(theta_efficiency, 4),
                        'moneyness': round(strike / current_price, 4),
                        'model_used': greeks['model_used'],
                        'score': delta_score + dte_score * 0.5  # For sorting
                    })

            except Exception as e:
                continue

        # Sort by score (best matches first)
        all_options.sort(key=lambda x: x['score'])

        return all_options

    except Exception as e:
        print(f"Error getting options: {e}")
        return []


def format_options_table(options):
    """Format multiple options for display"""
    if not options:
        return "No suitable options found"

    output = f"\n{'Strike':<8} {'Delta':<8} {'DTE':<5} {'Premium':<8} {'ROR':<7} {'Annual':<8} {'Theta/Day':<10} {'Vol':<6} {'OI':<6}\n"
    output += "-" * 90 + "\n"

    for opt in options:
        output += (f"${opt['strike']:<7.2f} "
                  f"{abs(opt['delta']):<7.3f} "
                  f"{opt['dte']:<4} "
                  f"${opt['premium']:<7.2f} "
                  f"{opt['ror']:<6.2f}% "
                  f"{opt['annualized_ror']:<7.1f}% "
                  f"{opt['theta_efficiency']:<9.4f}% "
                  f"{opt['volume']:<5} "
                  f"{opt['open_interest']:<5}\n")

    return output


if __name__ == "__main__":
    # Test
    print("Finding all CSP options for NVDA (0.10 to 0.40 delta)...")
    options = get_all_csp_options('NVDA', min_delta=0.10, max_delta=0.40)
    print(f"\nFound {len(options)} options:")
    print(format_options_table(options[:10]))  # Show top 10
