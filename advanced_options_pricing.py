"""
Advanced Options Pricing with Maximum Accuracy
Uses Black-Scholes-Merton with American option adjustments
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def get_risk_free_rate():
    """
    Get current risk-free rate from 1-month Treasury bill
    Falls back to reasonable estimate if API fails
    """
    try:
        # Get 1-month Treasury rate
        tnx = yf.Ticker("^IRX")  # 13-week Treasury Bill
        rate = tnx.history(period='1d')['Close'].iloc[-1] / 100
        return rate if rate > 0 else 0.045  # Fallback to 4.5%
    except:
        return 0.045  # Default 4.5%


def get_dividend_yield(ticker):
    """
    Get current dividend yield for stock
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Try to get dividend yield
        div_yield = info.get('dividendYield', 0)
        if div_yield and div_yield > 0:
            return div_yield

        # Calculate from dividend rate and price
        div_rate = info.get('dividendRate', 0)
        price = info.get('currentPrice', info.get('regularMarketPrice', 0))

        if div_rate > 0 and price > 0:
            return div_rate / price

        return 0.0
    except:
        return 0.0


def black_scholes_price(S, K, T, r, sigma, q, option_type='put'):
    """
    Black-Scholes-Merton option pricing (includes dividends)

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        q: Dividend yield (annual)
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    if T <= 0:
        return max(K - S, 0) if option_type == 'put' else max(S - K, 0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


def black_scholes_delta(S, K, T, r, sigma, q, option_type='put'):
    """
    Calculate exact Black-Scholes delta

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        q: Dividend yield (annual)
        option_type: 'call' or 'put'

    Returns:
        Delta (negative for puts)
    """
    if T <= 0:
        if option_type == 'put':
            return -1.0 if S < K else 0.0
        else:
            return 1.0 if S > K else 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:  # put
        delta = -np.exp(-q * T) * norm.cdf(-d1)

    return delta


def calculate_implied_volatility(market_price, S, K, T, r, q, option_type='put'):
    """
    Calculate implied volatility from market price using Brent's method
    This is what the exchanges do to get IV

    Args:
        market_price: Current market price of option
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        q: Dividend yield
        option_type: 'call' or 'put'

    Returns:
        Implied volatility (annual)
    """
    if T <= 0 or market_price <= 0:
        return 0.3  # Default 30%

    # Intrinsic value
    if option_type == 'put':
        intrinsic = max(K - S, 0)
    else:
        intrinsic = max(S - K, 0)

    # If option has no time value, return low IV
    if market_price <= intrinsic + 0.01:
        return 0.1

    def objective(sigma):
        """Difference between theoretical and market price"""
        try:
            theoretical = black_scholes_price(S, K, T, r, sigma, q, option_type)
            return theoretical - market_price
        except:
            return 1e10

    try:
        # Brent's method to find IV that matches market price
        # Search between 0.01 (1%) and 5.0 (500%) volatility
        iv = brentq(objective, 0.01, 5.0, maxiter=100)
        return iv
    except:
        # If optimization fails, use approximation
        # Brenner-Subrahmanyam approximation for ATM options
        try:
            iv = np.sqrt(2 * np.pi / T) * (market_price / S)
            return max(0.05, min(iv, 3.0))  # Clamp to reasonable range
        except:
            return 0.3  # Fallback


def binomial_american_option(S, K, T, r, sigma, q, option_type='put', steps=100):
    """
    Binomial tree for American options (more accurate than European BS)
    American options can be exercised early, so this is technically correct

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
        option_type: 'call' or 'put'
        steps: Number of time steps (more = more accurate, slower)

    Returns:
        (option_price, delta)
    """
    if T <= 0:
        intrinsic = max(K - S, 0) if option_type == 'put' else max(S - K, 0)
        delta = -1.0 if (option_type == 'put' and S < K) else 0.0
        return intrinsic, delta

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability
    discount = np.exp(-r * dt)

    # Initialize asset prices at maturity
    asset_prices = np.zeros(steps + 1)
    for i in range(steps + 1):
        asset_prices[i] = S * (u ** (steps - i)) * (d ** i)

    # Initialize option values at maturity
    option_values = np.zeros(steps + 1)
    for i in range(steps + 1):
        if option_type == 'put':
            option_values[i] = max(K - asset_prices[i], 0)
        else:
            option_values[i] = max(asset_prices[i] - K, 0)

    # Step back through tree
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            # Calculate stock price at this node
            stock_price = S * (u ** (step - i)) * (d ** i)

            # Expected option value (discounted)
            hold_value = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])

            # Exercise value
            if option_type == 'put':
                exercise_value = max(K - stock_price, 0)
            else:
                exercise_value = max(stock_price - K, 0)

            # American option: max of hold or exercise
            option_values[i] = max(hold_value, exercise_value)

    # Calculate delta from first two nodes
    price_up = S * u
    price_down = S * d
    delta = (option_values[0] - option_values[1]) / (price_up - price_down)

    return option_values[0], delta


def get_accurate_greeks(ticker, strike, expiration_date, option_type='put', use_binomial=True):
    """
    Get most accurate possible Greeks for an option

    Args:
        ticker: Stock symbol
        strike: Option strike price
        expiration_date: Expiration date string (YYYY-MM-DD)
        option_type: 'call' or 'put'
        use_binomial: Use binomial tree for American options (slower but more accurate)

    Returns:
        dict with price, delta, gamma, theta, vega, rho, iv
    """
    try:
        stock = yf.Ticker(ticker)

        # Get current stock price
        hist = stock.history(period='1d')
        S = hist['Close'].iloc[-1]

        # Calculate time to expiration (years)
        exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
        T = (exp_dt - datetime.now()).days / 365.0

        if T <= 0:
            return None

        # Get risk-free rate
        r = get_risk_free_rate()

        # Get dividend yield
        q = get_dividend_yield(ticker)

        # Get options chain to find market price and calculate IV
        opt_chain = stock.option_chain(expiration_date)
        options_df = opt_chain.puts if option_type == 'put' else opt_chain.calls

        # Find the specific strike
        option_data = options_df[options_df['strike'] == strike]

        if option_data.empty:
            return None

        # Get market price (midpoint of bid-ask)
        bid = option_data['bid'].iloc[0]
        ask = option_data['ask'].iloc[0]
        market_price = (bid + ask) / 2

        if market_price <= 0:
            return None

        # Calculate implied volatility from market price
        iv = calculate_implied_volatility(market_price, S, strike, T, r, q, option_type)

        # Calculate Greeks using best method
        if use_binomial and T > 1/365:  # Use binomial for options >1 day
            # Binomial tree (handles American options correctly)
            steps = min(200, max(50, int(T * 365)))  # More steps for longer DTE
            theoretical_price, delta = binomial_american_option(
                S, strike, T, r, iv, q, option_type, steps
            )
        else:
            # Black-Scholes (faster, good for near-expiration)
            theoretical_price = black_scholes_price(S, strike, T, r, iv, q, option_type)
            delta = black_scholes_delta(S, strike, T, r, iv, q, option_type)

        # Calculate other Greeks using Black-Scholes
        # (Greeks are similar for American/European)
        d1 = (np.log(S / strike) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        # Gamma (same for calls and puts)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * iv * np.sqrt(T))

        # Vega (same for calls and puts)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in IV

        # Theta (different for calls and puts)
        if option_type == 'put':
            theta = (-S * norm.pdf(d1) * iv * np.exp(-q * T) / (2 * np.sqrt(T))
                    + q * S * norm.cdf(-d1) * np.exp(-q * T)
                    - r * strike * np.exp(-r * T) * norm.cdf(-d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * iv * np.exp(-q * T) / (2 * np.sqrt(T))
                    - q * S * norm.cdf(d1) * np.exp(-q * T)
                    + r * strike * np.exp(-r * T) * norm.cdf(d2)) / 365

        # Rho (different for calls and puts)
        if option_type == 'put':
            rho = -strike * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        else:
            rho = strike * T * np.exp(-r * T) * norm.cdf(d2) / 100

        return {
            'strike': float(strike),
            'stock_price': float(S),
            'market_price': float(market_price),
            'theoretical_price': float(theoretical_price),
            'bid': float(bid),
            'ask': float(ask),
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega),
            'rho': float(rho),
            'implied_volatility': float(iv),
            'risk_free_rate': float(r),
            'dividend_yield': float(q),
            'dte': (exp_dt - datetime.now()).days,
            'time_to_expiration_years': float(T),
            'model_used': 'Binomial (American)' if use_binomial else 'Black-Scholes (European)'
        }

    except Exception as e:
        print(f"Error calculating Greeks: {e}")
        return None


if __name__ == "__main__":
    # Test with NVDA
    result = get_accurate_greeks('NVDA', 180.0, '2026-03-13', 'put')
    if result:
        print("Accurate Greeks Calculation:")
        print(f"Stock Price: ${result['stock_price']:.2f}")
        print(f"Strike: ${result['strike']:.2f}")
        print(f"Market Price: ${result['market_price']:.2f}")
        print(f"Theoretical Price: ${result['theoretical_price']:.2f}")
        print(f"\nGreeks:")
        print(f"  Delta: {result['delta']:.4f}")
        print(f"  Gamma: {result['gamma']:.4f}")
        print(f"  Theta: ${result['theta']:.4f}/day")
        print(f"  Vega: ${result['vega']:.4f}/IV%")
        print(f"\nInputs:")
        print(f"  IV: {result['implied_volatility']*100:.2f}%")
        print(f"  Risk-free rate: {result['risk_free_rate']*100:.2f}%")
        print(f"  Dividend yield: {result['dividend_yield']*100:.2f}%")
        print(f"  DTE: {result['dte']} days")
        print(f"  Model: {result['model_used']}")
