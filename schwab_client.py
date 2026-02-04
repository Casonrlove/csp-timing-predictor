"""
Schwab API Client
Handles market data requests including options chains and market hours
"""

import requests
from datetime import datetime, timedelta
from schwab_auth import get_valid_access_token, load_config

# API Base URL
API_BASE = "https://api.schwabapi.com/marketdata/v1"


def _make_request(endpoint, params=None):
    """Make authenticated request to Schwab API"""
    access_token = get_valid_access_token()

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    url = f"{API_BASE}/{endpoint}"
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 401:
        # Token might be invalid, try refreshing
        from schwab_auth import refresh_access_token
        access_token = refresh_access_token()
        headers['Authorization'] = f'Bearer {access_token}'
        response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    return response.json()


def get_quote(symbol):
    """Get quote for a single symbol"""
    return _make_request(f"{symbol}/quotes")


def get_quotes(symbols):
    """Get quotes for multiple symbols"""
    if isinstance(symbols, str):
        symbols = [symbols]
    params = {'symbols': ','.join(symbols)}
    return _make_request("quotes", params)


def get_stock_price(symbol):
    """
    Get current stock price and basic quote data

    Returns:
        dict with price info: {
            'symbol': str,
            'price': float (last price),
            'bid': float,
            'ask': float,
            'open': float,
            'high': float,
            'low': float,
            'close': float (previous close),
            'volume': int,
            'change': float,
            'change_percent': float,
            '52_week_high': float,
            '52_week_low': float
        }
    """
    try:
        data = get_quote(symbol.upper())

        if symbol.upper() not in data:
            raise Exception(f"No quote data for {symbol}")

        quote = data[symbol.upper()]['quote']

        return {
            'symbol': symbol.upper(),
            'price': quote.get('lastPrice', quote.get('mark', 0)),
            'bid': quote.get('bidPrice', 0),
            'ask': quote.get('askPrice', 0),
            'open': quote.get('openPrice', 0),
            'high': quote.get('highPrice', 0),
            'low': quote.get('lowPrice', 0),
            'close': quote.get('closePrice', 0),
            'volume': quote.get('totalVolume', 0),
            'change': quote.get('netChange', 0),
            'change_percent': quote.get('netPercentChangeInDouble', 0),
            '52_week_high': quote.get('52WkHigh', 0),
            '52_week_low': quote.get('52WkLow', 0),
            'pe_ratio': quote.get('peRatio', 0),
            'dividend_yield': quote.get('divYield', 0),
            'description': quote.get('description', ''),
            'exchange': quote.get('exchangeName', ''),
            'source': 'schwab'
        }
    except Exception as e:
        raise Exception(f"Failed to get quote for {symbol}: {e}")


def get_price_history(symbol, period_type='year', period=1, frequency_type='daily', frequency=1):
    """
    Get historical price data

    Args:
        symbol: Stock ticker
        period_type: 'day', 'month', 'year', 'ytd'
        period: Number of periods (1, 2, 3, 5, 10, 15, 20)
        frequency_type: 'minute', 'daily', 'weekly', 'monthly'
        frequency: Frequency interval (1, 5, 10, 15, 30 for minute; 1 for others)

    Returns:
        dict with candles: {'candles': [...], 'symbol': str}
    """
    params = {
        'symbol': symbol.upper(),
        'periodType': period_type,
        'period': period,
        'frequencyType': frequency_type,
        'frequency': frequency
    }

    return _make_request("pricehistory", params)


def get_movers(index='$SPX', sort='PERCENT_CHANGE_UP', frequency=0):
    """
    Get market movers for an index

    Args:
        index: '$SPX', '$DJI', '$COMPX'
        sort: 'VOLUME', 'TRADES', 'PERCENT_CHANGE_UP', 'PERCENT_CHANGE_DOWN'
        frequency: 0 (all day), 1, 5, 10, 30, 60 minutes

    Returns:
        List of top movers
    """
    return _make_request(f"movers/{index}", {'sort': sort, 'frequency': frequency})


def get_option_chain(symbol, contract_type='PUT', strike_count=20,
                     from_date=None, to_date=None,
                     min_dte=30, max_dte=45):
    """
    Get option chain with Greeks from Schwab API

    Args:
        symbol: Stock ticker (e.g., 'NVDA')
        contract_type: 'PUT', 'CALL', or 'ALL'
        strike_count: Number of strikes above/below ATM
        from_date: Start date for expirations (YYYY-MM-DD)
        to_date: End date for expirations (YYYY-MM-DD)
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration

    Returns:
        Option chain data including Greeks (delta, gamma, theta, vega)
    """
    # Calculate date range if not provided
    if not from_date:
        from_date = (datetime.now() + timedelta(days=min_dte)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = (datetime.now() + timedelta(days=max_dte)).strftime('%Y-%m-%d')

    params = {
        'symbol': symbol.upper(),
        'contractType': contract_type.upper(),
        'strikeCount': strike_count,
        'fromDate': from_date,
        'toDate': to_date,
        'includeUnderlyingQuote': 'true'
    }

    return _make_request("chains", params)


def get_market_hours(markets=None, date=None):
    """
    Get market hours for specified markets

    Args:
        markets: List of markets ['equity', 'option', 'bond', 'future', 'forex']
                 If None, returns all markets
        date: Date to check (YYYY-MM-DD), defaults to today

    Returns:
        Market hours information including open/close times
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    if markets:
        params = {'markets': ','.join(markets), 'date': date}
        return _make_request("markets", params)
    else:
        params = {'date': date}
        return _make_request("markets", params)


def get_single_market_hours(market, date=None):
    """
    Get market hours for a single market

    Args:
        market: Market type ('equity', 'option', 'bond', 'future', 'forex')
        date: Date to check (YYYY-MM-DD), defaults to today

    Returns:
        Market hours for the specified market
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    params = {'date': date}
    return _make_request(f"markets/{market}", params)


def is_market_open(market='equity'):
    """
    Check if a specific market is currently open

    Args:
        market: Market type ('equity', 'option', etc.)

    Returns:
        Boolean indicating if market is open
    """
    try:
        hours = get_single_market_hours(market)
        # Parse the response to determine if market is open
        if market in hours:
            market_data = hours[market]
            if isinstance(market_data, dict):
                for session_type, session_data in market_data.items():
                    if isinstance(session_data, dict) and session_data.get('isOpen'):
                        return True
        return False
    except Exception as e:
        print(f"Error checking market hours: {e}")
        return False


def parse_option_chain_for_csp(chain_data, min_delta=0.10, max_delta=0.40):
    """
    Parse Schwab option chain response into our CSP format

    Args:
        chain_data: Raw response from get_option_chain()
        min_delta: Minimum delta (absolute value)
        max_delta: Maximum delta (absolute value)

    Returns:
        List of options formatted for our CSP analyzer
    """
    options = []

    if not chain_data or 'putExpDateMap' not in chain_data:
        return options

    underlying_price = chain_data.get('underlyingPrice', 0)
    symbol = chain_data.get('symbol', '')

    put_map = chain_data.get('putExpDateMap', {})

    for exp_date, strikes in put_map.items():
        # exp_date format: "2025-02-21:37" (date:dte)
        parts = exp_date.split(':')
        expiration = parts[0]
        dte = int(parts[1]) if len(parts) > 1 else 0

        for strike_price, option_list in strikes.items():
            strike = float(strike_price)

            for opt in option_list:
                # Get Greeks directly from Schwab
                delta = opt.get('delta', 0)
                delta_abs = abs(delta)

                # Filter by delta range
                if delta_abs < min_delta or delta_abs > max_delta:
                    continue

                # Skip if no bid (can't sell it)
                bid = opt.get('bid', 0)
                if bid <= 0:
                    continue

                gamma = opt.get('gamma', 0)
                theta = opt.get('theta', 0)
                vega = opt.get('vega', 0)
                rho = opt.get('rho', 0)
                iv = opt.get('volatility', 0) / 100  # Schwab returns as percentage

                ask = opt.get('ask', 0)
                last = opt.get('last', 0)
                volume = opt.get('totalVolume', 0)
                open_interest = opt.get('openInterest', 0)

                # Use mid price for premium calculation
                premium = (bid + ask) / 2 if ask > 0 else bid

                # Calculate metrics
                ror = (premium / strike) * 100 if strike > 0 else 0
                annualized_ror = ror * (365 / dte) if dte > 0 else 0
                theta_efficiency = abs(theta / strike) * 100 if strike > 0 else 0
                moneyness = strike / underlying_price if underlying_price > 0 else 0

                options.append({
                    'ticker': symbol,
                    'strike': strike,
                    'premium': round(premium, 2),
                    'bid': round(bid, 2),
                    'ask': round(ask, 2),
                    'last': round(last, 2),
                    'delta': round(delta, 4),
                    'gamma': round(gamma, 5),
                    'theta': round(theta, 4),
                    'vega': round(vega, 4),
                    'rho': round(rho, 4),
                    'implied_volatility': round(iv, 4),
                    'volume': volume,
                    'open_interest': open_interest,
                    'dte': dte,
                    'expiration': expiration,
                    'ror': round(ror, 2),
                    'annualized_ror': round(annualized_ror, 2),
                    'theta_efficiency': round(theta_efficiency, 4),
                    'moneyness': round(moneyness, 4),
                    'model_used': 'Schwab',
                    'underlying_price': underlying_price
                })

    # Sort by distance from ideal (30 delta, 37 DTE)
    for opt in options:
        delta_score = abs(abs(opt['delta']) - 0.30)
        dte_score = abs(opt['dte'] - 37.5) / 37.5
        opt['score'] = delta_score + dte_score * 0.5

    options.sort(key=lambda x: x['score'])

    return options


def get_csp_options_schwab(ticker, min_delta=0.10, max_delta=0.40, min_dte=30, max_dte=45):
    """
    Get CSP options using Schwab API (replaces Yahoo Finance + Black-Scholes)

    Args:
        ticker: Stock symbol
        min_delta: Minimum delta
        max_delta: Maximum delta
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration

    Returns:
        List of options with Greeks from Schwab
    """
    try:
        # Fetch option chain from Schwab
        chain = get_option_chain(
            symbol=ticker,
            contract_type='PUT',
            strike_count=30,
            min_dte=min_dte,
            max_dte=max_dte
        )

        # Parse into our format
        options = parse_option_chain_for_csp(chain, min_delta, max_delta)

        print(f"[Schwab] Found {len(options)} options for {ticker}")
        return options

    except Exception as e:
        print(f"[Schwab] Error fetching options for {ticker}: {e}")
        return []


if __name__ == "__main__":
    # Test the client
    print("Testing Schwab API Client...")
    print("="*50)

    try:
        # Test market hours
        print("\n1. Checking market hours...")
        hours = get_market_hours(['equity', 'option'])
        print(f"   Market hours retrieved: {list(hours.keys())}")

        # Test if market is open
        print("\n2. Checking if equity market is open...")
        is_open = is_market_open('equity')
        print(f"   Equity market open: {is_open}")

        # Test option chain
        print("\n3. Fetching NVDA option chain...")
        options = get_csp_options_schwab('NVDA', min_delta=0.15, max_delta=0.35)
        print(f"   Found {len(options)} options")

        if options:
            print("\n   Top 3 options:")
            for i, opt in enumerate(options[:3]):
                print(f"   {i+1}. ${opt['strike']} | Delta: {opt['delta']:.3f} | "
                      f"Premium: ${opt['premium']:.2f} | DTE: {opt['dte']}")

        print("\n" + "="*50)
        print("✅ Schwab API client working!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've authorized first: python schwab_auth.py")
