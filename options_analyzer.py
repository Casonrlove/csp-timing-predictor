"""
Options data analyzer for CSP timing
Calculates ROR and theta efficiency for optimal CSP strikes
Uses advanced Black-Scholes with Binomial tree for maximum accuracy
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from advanced_options_pricing import get_accurate_greeks


def get_best_csp_option(ticker, target_delta_min=0.25, target_delta_max=0.35,
                        target_dte_min=30, target_dte_max=45):
    """
    Find the best CSP option matching our criteria

    Args:
        ticker: Stock symbol
        target_delta_min: Minimum delta (default 0.25)
        target_delta_max: Maximum delta (default 0.35)
        target_dte_min: Minimum DTE (default 30)
        target_dte_max: Maximum DTE (default 45)

    Returns:
        dict with option data and calculated metrics, or None if not available
    """
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]

        # Get options expiration dates
        expirations = stock.options

        if not expirations:
            return None

        best_option = None
        best_score = float('inf')  # Lower is better (closer to target)

        for exp_date in expirations:
            # Calculate DTE
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            dte = (exp_datetime - datetime.now()).days

            # Skip if outside DTE range
            if dte < target_dte_min or dte > target_dte_max:
                continue

            # Get options chain for this expiration
            try:
                opt_chain = stock.option_chain(exp_date)
                puts = opt_chain.puts

                # Filter for ITM/ATM/slightly OTM puts (likely to have delta data)
                # Delta for puts is negative, so we look for -0.35 to -0.25
                puts = puts[puts['strike'] <= current_price * 1.05]  # Within 5% of current price

                for _, put in puts.iterrows():
                    strike = put['strike']
                    moneyness = strike / current_price

                    # Only process reasonable strikes (not too far OTM)
                    if moneyness < 0.85:  # More than 15% OTM
                        continue

                    # Calculate ACCURATE Greeks using advanced Black-Scholes
                    greeks = get_accurate_greeks(ticker, strike, exp_date, 'put', use_binomial=True)

                    if not greeks:
                        continue

                    delta = greeks['delta']
                    delta_magnitude = abs(delta)

                    # Check if delta is in target range
                    if delta_magnitude < target_delta_min or delta_magnitude > target_delta_max:
                        continue

                    # Use market price from Greeks calculation
                    premium = greeks['market_price']

                    if premium <= 0 or strike <= 0:
                        continue

                    # ROR: Total return on capital (premium / strike)
                    ror = (premium / strike) * 100

                    # Annualized ROR
                    annualized_ror = ror * (365 / dte)

                    # Theta efficiency: Daily premium as % of capital (use actual theta!)
                    theta_efficiency = abs(greeks['theta'] / strike) * 100  # Theta is already per day

                    # Score: how close to target delta center (0.30) and target DTE center (37.5)
                    target_delta = 0.30
                    target_dte = 37.5
                    delta_diff = abs(delta_magnitude - target_delta)
                    dte_diff = abs(dte - target_dte) / 37.5  # Normalize
                    score = delta_diff + dte_diff * 0.5  # Weight DTE less

                    if score < best_score:
                        best_score = score
                        best_option = {
                            'strike': float(strike),
                            'premium': float(premium),
                            'bid': float(greeks['bid']),
                            'ask': float(greeks['ask']),
                            'delta': round(delta, 4),  # Accurate delta!
                            'gamma': round(greeks['gamma'], 5),
                            'theta': round(greeks['theta'], 4),
                            'vega': round(greeks['vega'], 4),
                            'implied_volatility': round(greeks['implied_volatility'], 4),
                            'volume': int(put['volume']) if pd.notna(put['volume']) else 0,
                            'open_interest': int(put['openInterest']) if pd.notna(put['openInterest']) else 0,
                            'dte': dte,
                            'expiration': exp_date,
                            'ror': round(ror, 2),
                            'annualized_ror': round(annualized_ror, 2),
                            'theta_efficiency': round(theta_efficiency, 4),
                            'moneyness': round(moneyness, 4),
                            'model_used': greeks['model_used']
                        }

            except Exception as e:
                # Skip this expiration if there's an error
                continue

        return best_option

    except Exception as e:
        # Market closed or data unavailable
        return None


def format_options_display(options_data):
    """Format options data for display"""
    if not options_data:
        return "Options data not available (market closed or no suitable options found)"

    return f"""
    Strike: ${options_data['strike']:.2f}
    Premium: ${options_data['premium']:.2f}
    Delta: {options_data['delta']:.3f}
    DTE: {options_data['dte']} days
    Expiration: {options_data['expiration']}

    ROR: {options_data['ror']:.2f}% (total)
    Annualized ROR: {options_data['annualized_ror']:.2f}%
    Theta Efficiency: {options_data['theta_efficiency']:.4f}%/day

    Volume: {options_data['volume']}
    Open Interest: {options_data['open_interest']}
    """
