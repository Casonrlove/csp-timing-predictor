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
    # Class-level cache so all instances share the same market-context data
    # within a single process run (avoids redundant yfinance downloads).
    _spy_cache: dict = {}           # {period: DataFrame}
    _sector_cache: dict = {}        # {(ticker, period): DataFrame}
    _vix9d_cache: dict = {}         # {period: DataFrame}

    # Map each ticker to its closest sector / thematic ETF.
    # Unknown tickers fall back to SPY (market proxy).
    SECTOR_ETF_MAP: dict = {
        'NVDA': 'SMH', 'AMD': 'SMH', 'INTC': 'SMH', 'MRVL': 'SMH',
        'TSLA': 'XLY', 'AMZN': 'XLY',
        'META': 'XLC', 'GOOGL': 'XLC', 'GOOG': 'XLC', 'NFLX': 'XLC',
        'AAPL': 'XLK', 'MSFT': 'XLK', 'CRM': 'XLK',
        'SPY': 'SPY', 'QQQ': 'QQQ',
        'IWM': 'SPY',
        'V': 'XLF', 'MA': 'XLF',
        'MSTR': 'QQQ', 'COIN': 'XLF', 'PLTR': 'XLK',
    }

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

    # ------------------------------------------------------------------
    # Helper: fetch a reference series (SPY, sector ETF, VIX9D, PCR)
    # Uses a class-level cache keyed by (symbol, period) to avoid
    # redundant downloads when multiple instances share a process.
    # ------------------------------------------------------------------

    def _fetch_reference_series(self, symbol: str, column: str = 'Close') -> pd.Series | None:
        """Return a tz-naive daily Close series for ``symbol`` over self.period."""
        cache_key = (symbol, self.period)
        if cache_key not in CSPDataCollector._sector_cache:
            try:
                ref = yf.Ticker(symbol)
                df_ref = ref.history(period=self.period)
                if df_ref is not None and len(df_ref) > 0 and column in df_ref.columns:
                    series = df_ref[column].copy()
                    idx = pd.to_datetime(series.index)
                    # Convert tz-aware to tz-naive (UTC-strip) to match self.data index
                    if idx.tz is not None:
                        idx = idx.tz_convert('UTC').tz_localize(None)
                    series.index = idx
                    CSPDataCollector._sector_cache[cache_key] = series
                else:
                    CSPDataCollector._sector_cache[cache_key] = None
            except Exception as e:
                print(f"  [ref] Could not fetch {symbol}: {e}")
                CSPDataCollector._sector_cache[cache_key] = None
        return CSPDataCollector._sector_cache[cache_key]

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

        # Fetch VIX9D (short-term implied vol index) from Yahoo Finance
        try:
            vix9d_series = self._fetch_reference_series('^VIX9D')
            if vix9d_series is not None:
                vix9d_aligned = vix9d_series.reindex(self.data.index, method='nearest').ffill().bfill()
                self.data['VIX9D'] = vix9d_aligned
                vix_fallback = self.data['VIX'] if 'VIX' in self.data.columns else 15.0
                self.data['VIX9D'] = self.data['VIX9D'].fillna(vix_fallback)
            else:
                self.data['VIX9D'] = self.data['VIX'] if 'VIX' in self.data.columns else 15.0
        except Exception as e:
            print(f"  [VIX9D] Warning: {e}")
            self.data['VIX9D'] = self.data['VIX'] if 'VIX' in self.data.columns else 15.0

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

        # === IV-BASED FEATURES ===
        # These are critical for options pricing accuracy

        # 1. VIX Rank (VIX percentile over past year) - proxy for market IV
        if 'VIX' in df.columns:
            df['VIX_Rank'] = df['VIX'].rolling(window=252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50,
                raw=False
            )
            # Fill NaN with 50 (neutral)
            df['VIX_Rank'] = df['VIX_Rank'].fillna(50)

            # 2. IV vs RV Ratio (Volatility Risk Premium indicator)
            # VIX represents ~30-day implied vol, compare to realized
            # Ratio > 1 means IV > RV (options overpriced, good for selling)
            df['IV_RV_Ratio'] = df['VIX'] / df['Volatility_20D'].clip(lower=1)
            df['IV_RV_Ratio'] = df['IV_RV_Ratio'].clip(0.5, 3.0)  # Limit extremes
            df['IV_RV_Ratio'] = df['IV_RV_Ratio'].fillna(1.0)

            # 3. VIX Change features (IV momentum)
            df['VIX_Change_1D'] = df['VIX'].pct_change(1) * 100
            df['VIX_Change_5D'] = df['VIX'].pct_change(5) * 100
            df['VIX_SMA_10'] = df['VIX'].rolling(window=10).mean()
            df['VIX_vs_SMA'] = (df['VIX'] / df['VIX_SMA_10'] - 1) * 100

            # 4. VIX Level buckets (categorical via numeric)
            # Low VIX (<15) = complacent, High VIX (>25) = fearful
            df['VIX_Level'] = df['VIX'].apply(lambda x: 0 if x < 15 else (1 if x < 20 else (2 if x < 25 else 3)))

            # 5. Volatility term structure proxy
            # Compare short-term (10D) vs medium-term (20D) realized vol
            df['Volatility_10D'] = close.pct_change().rolling(window=10).std() * np.sqrt(252) * 100
            df['Vol_Term_Structure'] = df['Volatility_10D'] / df['Volatility_20D'].clip(lower=1)
            df['Vol_Term_Structure'] = df['Vol_Term_Structure'].fillna(1.0)
        else:
            # Default values if VIX not available
            df['VIX_Rank'] = 50
            df['IV_RV_Ratio'] = 1.0
            df['VIX_Change_1D'] = 0
            df['VIX_Change_5D'] = 0
            df['VIX_vs_SMA'] = 0
            df['VIX_Level'] = 1
            df['Volatility_10D'] = df['Volatility_20D']
            df['Vol_Term_Structure'] = 1.0

        # Earnings proximity feature
        if hasattr(self, 'earnings_dates') and len(self.earnings_dates) > 0:
            df['Days_To_Earnings'] = self._calculate_days_to_earnings(df.index)
            # Create binary feature: within 7 days of earnings
            df['Near_Earnings'] = (np.abs(df['Days_To_Earnings']) <= 7).astype(int)
        else:
            df['Days_To_Earnings'] = 999  # Large number means unknown
            df['Near_Earnings'] = 0

        # NEW: Advanced risk features
        # Return distribution features (tail risk indicators)
        df['Return_Skew_20D'] = close.pct_change().rolling(window=20).skew()
        df['Return_Kurt_20D'] = close.pct_change().rolling(window=20).kurt()

        # Drawdown from 52-week high (distinct from 20D high)
        df['High_252D'] = close.rolling(window=252).max()
        df['Drawdown_From_52W_High'] = ((close - df['High_252D']) / df['High_252D']) * 100

        # Consecutive down days (panic/capitulation indicator)
        down_days = (close.pct_change() < 0).astype(int)
        df['Consecutive_Down_Days'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()

        # Market regime indicators
        df['Regime_Trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)  # Bull=1, Bear=0

        # VIX acceleration (rate of change of VIX change)
        if 'VIX_Change_1D' in df.columns:
            df['VIX_Acceleration'] = df['VIX_Change_1D'] - df['VIX_Change_1D'].shift(1)
        else:
            df['VIX_Acceleration'] = 0

        # === MEAN REVERSION FEATURES (8 new features) ===
        # These capture "stock dropped hard → likely to bounce" patterns

        # 1. Recent drop magnitude (key for mean reversion)
        df['Recent_Drop_1D'] = df['Return_1D'].apply(lambda x: abs(x) if x < 0 else 0)
        df['Recent_Drop_3D'] = close.pct_change(3).apply(lambda x: abs(x) * 100 if x < 0 else 0)
        df['Recent_Drop_5D'] = close.pct_change(5).apply(lambda x: abs(x) * 100 if x < 0 else 0)

        # 2. Oversold indicators
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)  # Classic oversold level
        df['RSI_Extreme'] = (df['RSI'] < 25).astype(int)   # Extreme oversold

        # 3. Pullback from recent highs (already have High_20D, add percentage)
        df['Pullback_From_5D_High'] = ((close - df['High'].rolling(window=5).max()) / df['High'].rolling(window=5).max()) * 100
        df['Pullback_From_20D_High'] = ((close - df['High_20D']) / df['High_20D']) * 100

        # 4. Return acceleration (is the drop accelerating or decelerating?)
        df['Return_Acceleration'] = df['Return_1D'] - df['Return_1D'].shift(1)
        # Negative acceleration during drop = deceleration = potential bottom

        # 5. Volume spike on down days (capitulation indicator)
        df['Volume_Spike_Down'] = ((df['Volume_Ratio'] > 1.5) & (df['Return_1D'] < 0)).astype(int)

        # 6. Mean reversion z-score
        df['Return_Mean_20D'] = df['Return_1D'].rolling(window=20).mean()
        df['Return_Std_20D'] = df['Return_1D'].rolling(window=20).std()
        # Avoid division by zero
        df['Return_ZScore'] = np.where(
            df['Return_Std_20D'] > 0,
            (df['Return_1D'] - df['Return_Mean_20D']) / df['Return_Std_20D'],
            0
        )
        # Z-score < -2 = strong deviation below mean = reversion candidate

        # === REGIME & MEAN-REVERSION QUALITY FEATURES ===

        # 7. Rolling lag-1 autocorrelation of returns (20D window)
        # Negative = mean-reverting (good CSP entry after drops)
        # Positive = trending (dangerous to sell puts)
        daily_returns = close.pct_change()
        df['Return_Autocorr_20D'] = daily_returns.rolling(window=20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 6 else 0.0, raw=False
        ).fillna(0.0)

        # 8. Variance Ratio (simpler mean-reversion proxy, faster than Hurst)
        # VR < 1 = mean-reverting, VR > 1 = trending
        # VR = Var(k-period returns) / (k * Var(1-period returns))
        def _variance_ratio(returns, k=5):
            """Variance ratio test statistic"""
            if len(returns) < k + 5:
                return 1.0
            r = np.array(returns)
            var1 = np.var(np.diff(r), ddof=1) if len(r) > 1 else 1e-8
            if var1 < 1e-10:
                return 1.0
            # k-period log returns from overlapping windows
            log_prices = np.log(r + 1e-10)
            k_returns = np.diff(log_prices, n=k) if len(log_prices) > k else np.array([0.0])
            var_k = np.var(k_returns, ddof=1) if len(k_returns) > 1 else 1e-8
            return var_k / (k * var1)

        # Use price series for variance ratio (rolling 40D window)
        df['Variance_Ratio_5D'] = close.rolling(window=40).apply(
            lambda x: _variance_ratio(x, k=5), raw=True
        ).fillna(1.0).clip(0.2, 3.0)

        # 9. Hurst Exponent (rolling 60D) using simplified R/S analysis
        # H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending
        def _hurst_rs(prices):
            """Simplified R/S Hurst exponent on price series"""
            n = len(prices)
            if n < 20:
                return 0.5
            log_ret = np.diff(np.log(prices + 1e-10))
            if len(log_ret) < 10:
                return 0.5

            # Use two sub-lags to estimate slope
            half = max(len(log_ret) // 2, 8)
            rs_vals = []
            for lag in [half // 2, half]:
                sub = log_ret[:lag]
                mean = np.mean(sub)
                dev = np.cumsum(sub - mean)
                R = dev.max() - dev.min()
                S = np.std(sub, ddof=1)
                if S > 1e-10:
                    rs_vals.append((lag, R / S))

            if len(rs_vals) < 2:
                return 0.5
            lags_log = np.log([v[0] for v in rs_vals])
            rs_log = np.log([v[1] for v in rs_vals])
            H = np.polyfit(lags_log, rs_log, 1)[0]
            return float(np.clip(H, 0.1, 0.9))

        df['Hurst_Exponent_60D'] = close.rolling(window=60).apply(
            _hurst_rs, raw=True
        ).fillna(0.5)

        # 10. IV Percentile (more robust than IV Rank — uses median/IQR instead of min/max)
        # Less sensitive to outlier spikes; better calibrated probability
        if 'VIX' in df.columns:
            df['VIX_Percentile_252D'] = df['VIX'].rolling(window=252).apply(
                lambda x: float(np.mean(x[:-1] <= x[-1])) * 100 if len(x) > 1 else 50.0,
                raw=True
            ).fillna(50.0)
        else:
            df['VIX_Percentile_252D'] = 50.0

        # ================================================================
        # NEW TIER-1 MARKET-CONTEXT FEATURES
        # ================================================================

        # A. SPY-Relative Return — how did this stock do vs the market?
        spy_series = self._fetch_reference_series('SPY')
        if spy_series is not None:
            # reindex_like fills missing dates (holidays/weekends) via nearest-trading-day
            spy_aligned = spy_series.reindex(df.index, method='nearest').ffill().bfill()
            df['Return_5D_SPY'] = spy_aligned.pct_change(5) * 100
            df['Return_20D_SPY'] = spy_aligned.pct_change(20) * 100
            df['Stock_vs_SPY_5D'] = df['Return_5D'] - df['Return_5D_SPY']
            df['Stock_vs_SPY_20D'] = df['Return_20D'] - df['Return_20D_SPY']
        else:
            df['Stock_vs_SPY_5D'] = 0.0
            df['Stock_vs_SPY_20D'] = 0.0

        # B. Sector ETF Relative Strength
        sector_etf = self.SECTOR_ETF_MAP.get(self.ticker, 'SPY')
        sector_series = self._fetch_reference_series(sector_etf)
        if sector_series is not None:
            sector_aligned = sector_series.reindex(df.index, method='nearest').ffill().bfill()
            df['Return_5D_Sector'] = sector_aligned.pct_change(5) * 100
            df['Return_20D_Sector'] = sector_aligned.pct_change(20) * 100
            df['Sector_RS_5D'] = df['Return_5D'] - df['Return_5D_Sector']
            df['Sector_RS_20D'] = df['Return_20D'] - df['Return_20D_Sector']
        else:
            df['Sector_RS_5D'] = 0.0
            df['Sector_RS_20D'] = 0.0

        # C. VIX Term Structure using VIX9D (near-term fear vs medium-term)
        if 'VIX9D' in df.columns and 'VIX' in df.columns:
            df['VIX9D_Ratio'] = (df['VIX9D'] / df['VIX'].clip(lower=1)).clip(0.5, 3.0).fillna(1.0)
            vix9d_sma5 = df['VIX9D'].rolling(window=5).mean()
            df['VIX9D_vs_SMA5'] = ((df['VIX9D'] / vix9d_sma5.clip(lower=0.1)) - 1) * 100
            df['VIX9D_vs_SMA5'] = df['VIX9D_vs_SMA5'].fillna(0.0)
        else:
            df['VIX9D_Ratio'] = 1.0
            df['VIX9D_vs_SMA5'] = 0.0

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
            raw_value = time_diffs[closest_idx]
            # Clip to [-30, 90] so stale past earnings don't produce extreme
            # out-of-distribution values (e.g. -2991) that pollute the scaler.
            days_to_earnings.append(int(np.clip(raw_value, -30, 90)))

        return days_to_earnings

    def create_target_variable(self, forward_days=35, strike_otm_pct=0.10, use_csp_outcome=True):
        """
        Create target variable: Was it a good time to sell a CSP?

        NEW APPROACH (use_csp_outcome=True):
        Simulates actual CSP trade outcome - works correctly for ALL tickers including inverse ETFs!
        - Strike price = current_price * (1 - strike_otm_pct)  [typically 10% OTM for 0.30 delta]
        - CSP profitable if stock never drops below strike during holding period
        - This automatically handles TQQQ (bull) vs SQQQ (bear) correctly!

        OLD APPROACH (use_csp_outcome=False):
        Just checks if stock avoided X% drop - doesn't understand CSP directionality

        Args:
            forward_days: Days to hold CSP (default 35, middle of 30-45 DTE range)
            strike_otm_pct: How far OTM to set strike (default 0.10 = 10% below = ~0.30 delta)
            use_csp_outcome: Use actual CSP simulation (True) vs simple drawdown check (False)
        """
        df = self.data.copy()

        if use_csp_outcome:
            # SIMULATE ACTUAL CSP TRADES
            print(f"Using CSP outcome simulation (strike {strike_otm_pct*100:.0f}% OTM)")

            csp_profitable_list = []
            min_price_list = []
            strike_breach_pct_list = []

            for i in range(len(df)):
                if i + forward_days < len(df):
                    # Entry price and strike
                    entry_price = df['Close'].iloc[i]
                    strike_price = entry_price * (1 - strike_otm_pct)

                    # Future prices over holding period
                    future_prices = df['Close'].iloc[i:i+forward_days+1]
                    min_price = future_prices.min()

                    # CSP is profitable if stock never breached strike
                    csp_profitable = (min_price >= strike_price)

                    # How far below strike did it go (if breached)?
                    strike_breach_pct = ((min_price - strike_price) / strike_price) * 100 if min_price < strike_price else 0

                    csp_profitable_list.append(1 if csp_profitable else 0)
                    min_price_list.append(min_price)
                    strike_breach_pct_list.append(strike_breach_pct)
                else:
                    csp_profitable_list.append(np.nan)
                    min_price_list.append(np.nan)
                    strike_breach_pct_list.append(np.nan)

            df['CSP_Profitable'] = csp_profitable_list
            df['Min_Price_35D'] = min_price_list
            df['Strike_Breach_Pct'] = strike_breach_pct_list  # Negative if breached
            df['Good_CSP_Time'] = df['CSP_Profitable']

            # Also keep max drawdown for reference
            df['Max_Drawdown_35D'] = ((df['Min_Price_35D'] - df['Close']) / df['Close']) * 100

        else:
            # OLD METHOD: Just check drawdown (doesn't handle inverse ETFs correctly)
            print(f"Using old drawdown method (not recommended for inverse ETFs)")

            # Calculate forward returns and max drawdown
            df['Forward_Return'] = df['Close'].shift(-forward_days) / df['Close'] - 1

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

            # Use threshold to determine good timing
            threshold_pct = -strike_otm_pct * 100  # Convert to drawdown %
            df['Good_CSP_Time'] = (df['Max_Drawdown_35D'] > threshold_pct).astype(int)

        # Premium potential (for reference)
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

            # Market context & IV features
            'VIX',
            'VIX_Rank', 'IV_RV_Ratio',
            'VIX_Change_1D', 'VIX_Change_5D', 'VIX_vs_SMA',
            'VIX_Level', 'Volatility_10D', 'Vol_Term_Structure',
            'VIX_Acceleration',

            # Earnings proximity
            'Days_To_Earnings', 'Near_Earnings',

            # Advanced risk features
            'Return_Skew_20D', 'Return_Kurt_20D',
            'Drawdown_From_52W_High', 'Consecutive_Down_Days',
            'Regime_Trend',

            # Mean reversion features (8 features — pruned redundant ones)
            'Recent_Drop_3D', 'Recent_Drop_5D',
            'Pullback_From_5D_High',
            'Return_Acceleration',
            'Volume_Spike_Down',
            'Return_Mean_20D', 'Return_Std_20D', 'Return_ZScore',

            # Regime & mean-reversion quality (2 features — pruned noisy ones)
            'Return_Autocorr_20D',   # Lag-1 autocorrelation: negative = mean-reverting
            'Variance_Ratio_5D',     # < 1 = mean-reverting, > 1 = trending

            # Tier-1 market-context features (4 features — pruned VIX9D fallback noise)
            'Stock_vs_SPY_5D',       # A: Stock outperformance vs SPY (5D)
            'Stock_vs_SPY_20D',      # A: Stock outperformance vs SPY (20D)
            'Sector_RS_5D',          # B: Sector relative strength (5D)
            'Sector_RS_20D',         # B: Sector relative strength (20D)
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
