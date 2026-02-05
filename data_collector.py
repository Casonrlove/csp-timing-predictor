"""
Data collection and feature engineering for CSP timing model
Focuses on technical support levels and optimal entry points
Uses Schwab API for accurate market data
"""

import yfinance as yf  # Fallback and for earnings dates
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from datetime import datetime, timedelta

# Try to import Schwab client
try:
    from schwab_client import get_price_history
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False


class CSPDataCollector:
    def __init__(self, ticker='NVDA', period='10y'):
        self.ticker = ticker
        self.period = period
        self.data = None

    def _parse_period(self, period):
        """Convert period string to Schwab API parameters"""
        # period can be '10y', '5y', '1y', '6mo', etc.
        period = period.lower()
        if period.endswith('y'):
            years = int(period[:-1])
            if years >= 10:
                return 'year', 10, 'daily', 1
            elif years >= 5:
                return 'year', 5, 'daily', 1
            elif years >= 3:
                return 'year', 3, 'daily', 1
            elif years >= 2:
                return 'year', 2, 'daily', 1
            else:
                return 'year', 1, 'daily', 1
        elif period.endswith('mo'):
            months = int(period[:-2])
            return 'month', months, 'daily', 1
        else:
            return 'year', 1, 'daily', 1

    def _schwab_candles_to_df(self, data):
        """Convert Schwab candles response to pandas DataFrame"""
        if not data or 'candles' not in data:
            return None

        candles = data['candles']
        if not candles:
            return None

        df = pd.DataFrame(candles)
        # Schwab returns: datetime (epoch ms), open, high, low, close, volume
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.index = df.index.tz_localize(None)  # Remove timezone

        # Rename columns to match yfinance format
        df.columns = [col.capitalize() for col in df.columns]
        df = df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})

        return df

    def fetch_data(self):
        """Fetch historical stock data from Schwab API (with Yahoo fallback)"""
        print(f"Fetching data for {self.ticker}...")

        data_source = "Yahoo"

        # Try Schwab first
        if SCHWAB_AVAILABLE:
            try:
                period_type, period_num, freq_type, freq = self._parse_period(self.period)
                print(f"[Schwab] Fetching {period_num} {period_type}(s) of daily data...")

                schwab_data = get_price_history(
                    self.ticker,
                    period_type=period_type,
                    period=period_num,
                    frequency_type=freq_type,
                    frequency=freq
                )

                self.data = self._schwab_candles_to_df(schwab_data)

                if self.data is not None and len(self.data) > 0:
                    data_source = "Schwab"
                    print(f"[Schwab] Loaded {len(self.data)} days of price history")
                else:
                    print("[Schwab] No data returned, falling back to Yahoo")
                    self.data = None

            except Exception as e:
                print(f"[Schwab] Error fetching price history: {e}")
                print("[Schwab] Falling back to Yahoo Finance")
                self.data = None

        # Fallback to Yahoo Finance
        if self.data is None:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            data_source = "Yahoo"
            print(f"[Yahoo] Loaded {len(self.data)} days of price history")

        # Get earnings dates from Yahoo (Schwab doesn't provide this)
        try:
            stock = yf.Ticker(self.ticker)
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and len(earnings_dates) > 0:
                earnings_list = earnings_dates.index.tolist()
                # Make earnings dates tz-naive to match price data
                self.earnings_dates = [pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d) for d in earnings_list]
                print(f"Loaded {len(self.earnings_dates)} earnings dates")
            else:
                self.earnings_dates = []
                print("No earnings dates available")
        except Exception as e:
            print(f"Could not fetch earnings dates: {e}")
            self.earnings_dates = []

        # Get VIX data - try Schwab first, then Yahoo
        vix_loaded = False
        if SCHWAB_AVAILABLE:
            try:
                period_type, period_num, freq_type, freq = self._parse_period(self.period)
                vix_data = get_price_history(
                    '$VIX',
                    period_type=period_type,
                    period=period_num,
                    frequency_type=freq_type,
                    frequency=freq
                )
                vix_df = self._schwab_candles_to_df(vix_data)
                if vix_df is not None and len(vix_df) > 0:
                    vix_df = vix_df[['Close']]
                    vix_df.columns = ['VIX']
                    self.data = self.data.join(vix_df, how='left')
                    self.data['VIX'] = self.data['VIX'].ffill().bfill()
                    if self.data['VIX'].isna().any():
                        self.data['VIX'] = self.data['VIX'].fillna(15.0)
                    vix_loaded = True
                    print(f"[Schwab] VIX data loaded: {self.data['VIX'].notna().sum()} values")
            except Exception as e:
                print(f"[Schwab] Could not fetch VIX: {e}")

        # Fallback to Yahoo for VIX
        if not vix_loaded:
            try:
                vix = yf.Ticker('^VIX')
                vix_data = vix.history(period=self.period)
                if len(vix_data) > 0 and 'Close' in vix_data.columns:
                    vix_data = vix_data[['Close']]
                    vix_data.columns = ['VIX']
                    self.data = self.data.join(vix_data, how='left')
                    self.data['VIX'] = self.data['VIX'].ffill().bfill()
                    if self.data['VIX'].isna().any():
                        self.data['VIX'] = self.data['VIX'].fillna(15.0)
                    vix_loaded = True
                    print(f"[Yahoo] VIX data loaded: {self.data['VIX'].notna().sum()} values")
            except Exception as e:
                print(f"Warning: Could not fetch VIX data ({e})")

        # If VIX not loaded, use constant value
        if not vix_loaded or 'VIX' not in self.data.columns or self.data['VIX'].isna().all():
            print("Using constant VIX value (15.0)")
            self.data['VIX'] = 15.0

        print(f"Fetched {len(self.data)} days of data (source: {data_source})")
        self.data_source = data_source
        return self.data

    def calculate_support_resistance(self, window=20):
        """Calculate support and resistance levels"""
        df = self.data.copy()

        # Rolling support (lowest low in window)
        df['Support'] = df['Low'].rolling(window=window).min()

        # Rolling resistance (highest high in window)
        df['Resistance'] = df['High'].rolling(window=window).max()

        # Distance from support (key feature for CSP timing)
        df['Distance_From_Support_Pct'] = ((df['Close'] - df['Support']) / df['Support']) * 100

        # Distance from resistance
        df['Distance_From_Resistance_Pct'] = ((df['Resistance'] - df['Close']) / df['Close']) * 100

        return df

    def calculate_technical_indicators(self):
        """Calculate technical indicators focused on entry timing"""
        df = self.data.copy()

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Moving Averages - identify trend and support
        df['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(close=close, window=200).sma_indicator()
        df['EMA_12'] = EMAIndicator(close=close, window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=close, window=26).ema_indicator()

        # Price relative to moving averages (support indicators)
        df['Price_to_SMA20'] = (close / df['SMA_20'] - 1) * 100
        df['Price_to_SMA50'] = (close / df['SMA_50'] - 1) * 100
        df['Price_to_SMA200'] = (close / df['SMA_200'] - 1) * 100

        # RSI - oversold conditions can indicate support
        rsi = RSIIndicator(close=close, window=14)
        df['RSI'] = rsi.rsi()

        # Stochastic - oversold/overbought
        stoch = StochasticOscillator(high=high, low=low, close=close)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # MACD - momentum and trend
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()

        # Bollinger Bands - identify when price is at lower band (support)
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Width'] = bb.bollinger_wband()
        df['BB_Position'] = ((close - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])) * 100

        # ATR - volatility measure
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df['ATR'] = atr.average_true_range()
        df['ATR_Pct'] = (df['ATR'] / close) * 100

        # ADX - trend strength
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        df['ADX'] = adx.adx()

        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        # OBV - volume momentum
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        df['OBV'] = obv.on_balance_volume()
        df['OBV_SMA_20'] = df['OBV'].rolling(window=20).mean()

        # Recent price action
        df['Return_1D'] = close.pct_change(1) * 100
        df['Return_5D'] = close.pct_change(5) * 100
        df['Return_20D'] = close.pct_change(20) * 100

        # Volatility (realized)
        df['Volatility_20D'] = close.pct_change().rolling(window=20).std() * np.sqrt(252) * 100

        # Distance from recent high/low
        df['High_20D'] = high.rolling(window=20).max()
        df['Low_20D'] = low.rolling(window=20).min()
        df['Distance_From_High_20D'] = ((df['High_20D'] - close) / close) * 100
        df['Distance_From_Low_20D'] = ((close - df['Low_20D']) / close) * 100

        # Support and resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Distance_From_Support_Pct'] = ((df['Close'] - df['Support']) / df['Support']) * 100
        df['Distance_From_Resistance_Pct'] = ((df['Resistance'] - df['Close']) / df['Close']) * 100

        # IV Rank (realized volatility percentile over past year)
        df['Volatility_252D'] = close.pct_change().rolling(window=252).std() * np.sqrt(252) * 100
        df['IV_Rank'] = df['Volatility_20D'].rolling(window=252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50,
            raw=False
        )

        # Volatility ratio (current vs 252-day average)
        df['Volatility_Ratio'] = df['Volatility_20D'] / df['Volatility_252D']

        # Earnings proximity feature
        if hasattr(self, 'earnings_dates') and len(self.earnings_dates) > 0:
            df['Days_To_Earnings'] = self._calculate_days_to_earnings(df.index)
            # Create binary feature: within 7 days of earnings
            df['Near_Earnings'] = (np.abs(df['Days_To_Earnings']) <= 7).astype(int)
        else:
            df['Days_To_Earnings'] = 999  # Large number means unknown
            df['Near_Earnings'] = 0

        self.data = df
        return df

    def _calculate_days_to_earnings(self, dates):
        """Calculate days until next earnings for each date"""
        days_to_earnings = []

        for date in dates:
            # Find nearest earnings date (past or future)
            # Ensure date is tz-naive
            date = pd.Timestamp(date)
            if date.tzinfo is not None:
                date = date.tz_localize(None)

            if len(self.earnings_dates) == 0:
                days_to_earnings.append(999)
                continue

            # Find closest earnings date
            time_diffs = [(e - date).days for e in self.earnings_dates]
            closest_idx = np.argmin(np.abs(time_diffs))
            days_to_earnings.append(time_diffs[closest_idx])

        return days_to_earnings

    def create_target_variable(self, forward_days=35, threshold_pct=-5):
        """
        Create target variable: Was it a good time to sell a CSP?
        Good time = stock doesn't drop significantly in next 30-45 days

        Args:
            forward_days: Days to look forward (default 35, middle of 30-45 range)
            threshold_pct: Max acceptable drawdown % (default -5%, relaxed from -3%)
        """
        df = self.data.copy()

        # Calculate forward returns and max drawdown
        df['Forward_Return'] = df['Close'].shift(-forward_days) / df['Close'] - 1

        # Calculate max drawdown over next forward_days
        max_drawdown_list = []
        for i in range(len(df)):
            if i + forward_days < len(df):
                future_prices = df['Close'].iloc[i:i+forward_days+1]
                current_price = df['Close'].iloc[i]
                drawdown = ((future_prices.min() - current_price) / current_price) * 100
                max_drawdown_list.append(drawdown)
            else:
                max_drawdown_list.append(np.nan)

        df['Max_Drawdown_35D'] = max_drawdown_list

        # Target: Good time to sell CSP
        # 1 if max drawdown > threshold (e.g., didn't drop more than 3%)
        # 0 if max drawdown <= threshold (bad time, would have been assigned or stressed)
        df['Good_CSP_Time'] = (df['Max_Drawdown_35D'] > threshold_pct).astype(int)

        # Also create a regression target for premium potential
        # Assume we could sell at 0.30 delta, approximate premium based on ATR
        df['Expected_Premium_Pct'] = df['ATR_Pct'] * 0.3  # Rough approximation

        self.data = df
        return df

    def prepare_features(self):
        """Prepare final feature set for modeling"""
        # List of features to use
        feature_cols = [
            # Price position indicators
            'Price_to_SMA20', 'Price_to_SMA50', 'Price_to_SMA200',
            'Distance_From_Support_Pct', 'Distance_From_Resistance_Pct',
            'Distance_From_High_20D', 'Distance_From_Low_20D',

            # Momentum indicators
            'RSI', 'Stoch_K', 'Stoch_D',
            'MACD', 'MACD_Signal', 'MACD_Diff',

            # Volatility indicators
            'BB_Position', 'BB_Width',
            'ATR_Pct', 'Volatility_20D', 'Volatility_252D',
            'IV_Rank', 'Volatility_Ratio',

            # Trend indicators
            'ADX',

            # Volume indicators
            'Volume_Ratio',

            # Returns
            'Return_1D', 'Return_5D', 'Return_20D',

            # Market context
            'VIX',

            # Earnings proximity
            'Days_To_Earnings', 'Near_Earnings'
        ]

        # Check for NaN values before dropping
        df_check = self.data[feature_cols + ['Good_CSP_Time', 'Max_Drawdown_35D']]

        print(f"\nBefore dropna: {len(df_check)} rows")
        nan_counts = df_check.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            print(f"Columns with NaN values:")
            for col, count in nan_cols.items():
                print(f"  {col}: {count} NaN values ({count/len(df_check)*100:.1f}%)")

        # Remove rows with NaN values
        df_clean = df_check.dropna()

        print(f"\nPrepared {len(df_clean)} samples with {len(feature_cols)} features")
        if len(df_clean) > 0:
            print(f"Target distribution: {df_clean['Good_CSP_Time'].value_counts().to_dict()}")
        else:
            print("ERROR: No samples remaining after dropna!")
            print("\nAll data columns:")
            print(self.data.columns.tolist())

        return df_clean, feature_cols

    def get_training_data(self):
        """Full pipeline to get training data"""
        self.fetch_data()

        # Check if we have enough data
        if len(self.data) < 250:
            raise ValueError(f"Not enough data: only {len(self.data)} days. Need at least 250 days.")

        self.calculate_technical_indicators()
        self.create_target_variable()

        df_clean, feature_cols = self.prepare_features()

        if len(df_clean) == 0:
            raise ValueError("No valid samples after feature engineering. This usually means all data has NaN values.")

        return df_clean, feature_cols


if __name__ == "__main__":
    collector = CSPDataCollector('NVDA', period='10y')
    df_clean, feature_cols = collector.get_training_data()

    print("\nSample of prepared data:")
    print(df_clean.head())

    print("\nFeature statistics:")
    print(df_clean[feature_cols].describe())

    # Save prepared data
    df_clean.to_csv('nvda_csp_training_data.csv', index=True)
    print("\nData saved to nvda_csp_training_data.csv")
