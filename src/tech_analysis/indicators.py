import pandas as pd
import talib
from ta.momentum import (
    rsi, stoch, stoch_signal, stochrsi, 
    stochrsi_d, stochrsi_k, awesome_oscillator, 
    williams_r, ultimate_oscillator
)
from ta.trend import (
    cci, adx, adx_neg, adx_pos, 
    ema_indicator, sma_indicator, ichimoku_a,
    ichimoku_b, ichimoku_base_line,
    ichimoku_conversion_line
)
from ta.volume import chaikin_money_flow


# =========== Trend Following Indicators ===============
def calc_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate Simple Moving Average and return as new DataFrame.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for SMA calculation
    """
    close_prices = df['close']
    result = pd.DataFrame(index=df.index)
    result[f'sma_{window}'] = sma_indicator(close=close_prices, window=window)
    return result

def calc_ema(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate Exponential Moving Average and return as new DataFrame.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for EMA calculation
    """
    close_prices = df['close']
    result = pd.DataFrame(index=df.index)
    result[f'ema_{window}'] = ema_indicator(close=close_prices, window=window)
    return result

def calc_adx(df: pd.DataFrame, adx_window: int, di_window: int) -> pd.DataFrame:
    """Calculate Average Directional Index and directional indicators.

    Args:
        df: DataFrame with OHLCV data
        adx_window: Window period for ADX calculation
        di_window: Window period for ADX positive/negative calculation
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    
    result = pd.DataFrame(index=df.index)
    result['adx'] = adx(high_prices, low_prices, close_prices, window=adx_window)
    result['adx_pos'] = adx_pos(high_prices, low_prices, close_prices, window=di_window)
    result['adx_neg'] = adx_neg(high_prices, low_prices, close_prices, window=di_window)
    return result
    
def calc_macd_diff(df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
    """Calculate MACD, signal line and histogram.

    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast period for MACD
        slow_period: Slow period for MACD
        signal_period: Signal period for MACD
    """
    close_prices = df['close']
    macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    
    result = pd.DataFrame(index=df.index)
    result['macd'] = macd
    result['macd_signal'] = macd_signal
    result['macd_hist'] = macd_hist
    return result

def calc_sar(df: pd.DataFrame, acceleration: float, maximum: float) -> pd.DataFrame:
    """Calculate Parabolic SAR indicator.

    Args:
        df: DataFrame with OHLCV data
        acceleration: Acceleration factor
        maximum: Maximum acceleration
    """
    high_prices = df['high']
    low_prices = df['low']
    
    result = pd.DataFrame(index=df.index)
    result['sar'] = talib.SAR(high_prices, low_prices, acceleration=acceleration, maximum=maximum)
    return result

def calc_ichimoku(df: pd.DataFrame, conversion_window: int, base_window: int, span_b_window: int) -> pd.DataFrame:
    """Calculate Ichimoku Cloud indicators.

    Args:
        df: DataFrame with OHLCV data
        conversion_window: Window period for conversion line
        base_window: Window period for base line
        span_b_window: Window period for span B
    """
    high_prices = df['high']
    low_prices = df['low']
    
    result = pd.DataFrame(index=df.index)
    result['ichimoku_base_line'] = ichimoku_base_line(
        high=high_prices, low=low_prices, window1=conversion_window, window2=base_window, visual=False
    )
    result['ichimoku_conversion_line'] = ichimoku_conversion_line(
        high=high_prices, low=low_prices, window1=conversion_window, window2=base_window, visual=False
    )
    result['ichimoku_a'] = ichimoku_a(
        high=high_prices, low=low_prices, window1=conversion_window, window2=base_window, visual=False
    )
    result['ichimoku_b'] = ichimoku_b(
        high=high_prices, low=low_prices, window2=base_window, window3=span_b_window, visual=False
    )
    return result

# ============== Momentum Indicators ===================
def calc_rsi(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate Relative Strength Index.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for RSI calculation
    """
    close_prices = df['close']
    result = pd.DataFrame(index=df.index)
    result['rsi'] = rsi(close_prices, window=window)
    return result

def calc_stoch(df: pd.DataFrame, window: int, smooth_window: int) -> pd.DataFrame:
    """Calculate Stochastic Oscillator.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for stochastic calculation
        smooth_window: Smoothing window period
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    
    result = pd.DataFrame(index=df.index)
    result['stoch_main'] = stoch(high_prices, low_prices, close_prices, window=window, smooth_window=smooth_window)
    result['stoch_side'] = stoch_signal(high_prices, low_prices, close_prices, window=window, smooth_window=smooth_window)
    return result

def calc_cci(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate Commodity Channel Index.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for CCI calculation
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    
    result = pd.DataFrame(index=df.index)
    result['cci'] = cci(high_prices, low_prices, close_prices, window=window)
    return result

def calc_momentum(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculate price momentum.

    Args:
        df: DataFrame with OHLCV data
        period: Period for momentum calculation
    """
    close_prices = df['close']
    result = pd.DataFrame(index=df.index)
    result['momentum'] = close_prices.diff(period)
    return result

# ============= Volatility Indicators ==================
def calc_bollinger_bands(df: pd.DataFrame, window: int, num_std: float) -> pd.DataFrame:
    """Calculate Bollinger Bands.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for moving average
        num_std: Number of standard deviations
    """
    close_prices = df['close']
    upper, middle, lower = talib.BBANDS(close_prices, window, num_std, num_std)
    
    result = pd.DataFrame(index=df.index)
    result['bbands_upper'] = upper
    result['bbands_middle'] = middle
    result['bbands_lower'] = lower
    return result

def calc_atr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate Average True Range.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for ATR calculation
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']

    atr = talib.ATR(high_prices, low_prices, close_prices, window)
    result = pd.DataFrame(index=df.index)
    result['ATR'] = atr
    return result

# =============== Volume Indicators ====================
def calc_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate On-Balance Volume.

    Args:
        df: DataFrame with OHLCV data
    """
    volumes = df['volume']
    close_prices = df['close']
    
    result = pd.DataFrame(index=df.index)
    result['obv'] = talib.OBV(close_prices, volumes)
    return result

def calc_vwap(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Calculate Volume Weighted Average Price.

    Args:
        df: DataFrame with OHLCV data
        period: Period for VWAP calculation
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    volumes = df['volume']
    typical_price = (high_prices + low_prices + close_prices) / 3
    
    result = pd.DataFrame(index=df.index)
    result['vwap'] = talib.SUM(typical_price * volumes, period) / talib.SUM(volumes, period)
    return result

# ================== Oscillators =======================
def calc_cmf(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate Chaikin Money Flow.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for CMF calculation
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    volumes = df['volume']
    
    result = pd.DataFrame(index=df.index)
    result['cmf'] = chaikin_money_flow(high_prices, low_prices, close_prices, volumes, window=window)
    return result

def calc_mfi(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate Money Flow Index.

    Args:
        df: DataFrame with OHLCV data
        window: Window period for MFI calculation
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    volumes = df['volume']
    
    result = pd.DataFrame(index=df.index)
    result['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=window)
    return result

def calc_williams_r(df: pd.DataFrame, lookback_period: int) -> pd.DataFrame:
    """Calculate Williams %R.

    Args:
        df: DataFrame with OHLCV data
        lookback_period: Lookback period for Williams %R
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    
    result = pd.DataFrame(index=df.index)
    result['wr'] = williams_r(high_prices, low_prices, close_prices, lbp=lookback_period)
    return result

def calc_awesome_oscillator(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Calculate Awesome Oscillator.

    Args:
        df: DataFrame with OHLCV data
        short_window: Short window period
        long_window: Long window period
    """
    high_prices = df['high']
    low_prices = df['low']
    
    result = pd.DataFrame(index=df.index)
    result['ao'] = awesome_oscillator(high_prices, low_prices, window1=short_window, window2=long_window)
    result['ao_prev'] = result['ao'].shift(1)
    return result

def calc_stochrsi(df: pd.DataFrame, smooth1: int, smooth2: int, rsi_window: int) -> pd.DataFrame:
    """Calculate Stochastic RSI.

    Args:
        df: DataFrame with OHLCV data
        smooth1: First smoothing parameter
        smooth2: Second smoothing parameter
        rsi_window: RSI window period
    """
    close_prices = df['close']
    
    result = pd.DataFrame(index=df.index)
    result['stoch_rsi'] = stochrsi(close_prices, window=rsi_window, smooth1=smooth1, smooth2=smooth2) * 100
    result['stoch_k'] = stochrsi_k(close_prices, window=rsi_window, smooth1=smooth1, smooth2=smooth2) * 100
    result['stoch_d'] = stochrsi_d(close_prices, window=rsi_window, smooth1=smooth1, smooth2=smooth2) * 100
    return result

def calc_ultimate_oscillator(df: pd.DataFrame, short_window: int, medium_window: int, long_window: int) -> pd.DataFrame:
    """Calculate Ultimate Oscillator.

    Args:
        df: DataFrame with OHLCV data
        short_window: Short window period
        medium_window: Medium window period
        long_window: Long window period
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    
    result = pd.DataFrame(index=df.index)
    result['uo'] = ultimate_oscillator(
        high_prices, low_prices, close_prices, 
        window1=short_window, window2=medium_window, window3=long_window
    )
    return result

# ============== Sentiment Indicators ====================
def calc_fear_greed_index(df: pd.DataFrame, rsi_period: int, volume_ma_period: int) -> pd.DataFrame:
    """Calculate Fear and Greed Index.

    Args:
        df: DataFrame with OHLCV data
        rsi_period: RSI period
        volume_ma_period: Volume moving average period for normalization
    """
    close_prices = df['close']
    volumes = df['volume']

    rsi = talib.RSI(close_prices, timeperiod=rsi_period)
    volume_ma = talib.SMA(volumes, timeperiod=volume_ma_period)
    
    normalized_rsi = (rsi - volume_ma_period) / volume_ma_period  
    normalized_volume_ma = (volumes / volume_ma) - 1 
    
    fear_greed_index = (normalized_rsi + normalized_volume_ma) / 2
    
    result = pd.DataFrame(index=df.index)
    result['fgi'] = (fear_greed_index + 1) * volume_ma_period 
    return result

# ================= Miscellaneous ======================
def calc_bull_bear_power(df: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    """Calculate Bull and Bear Power.

    Args:
        df: DataFrame with OHLCV data
        ema_period: EMA period for calculation
    """
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    ema_close = ema_indicator(close_prices, ema_period)
    
    result = pd.DataFrame(index=df.index)
    result['bull_power'] = high_prices - ema_close
    result['bear_power'] = low_prices - ema_close
    result['bull_bear_power'] = result['bull_power'] + result['bear_power']
    return result

def calc_fibonacci_retracement_levels(df: pd.DataFrame, level_236: float, level_382: float, level_500: float, level_618: float) -> pd.DataFrame:
    """Calculate Fibonacci Retracement Levels.

    Args:
        df: DataFrame with OHLCV data
        level_236: First Fibonacci level (typically 0.236)
        level_382: Second Fibonacci level (typically 0.382)
        level_500: Third Fibonacci level (typically 0.5)
        level_618: Fourth Fibonacci level (typically 0.618)
    """
    high_prices = df['high']
    low_prices = df['low']

    result = pd.DataFrame(index=df.index)
    
    for i in range(len(df)):
        high_price = high_prices.iloc[i]
        low_price = low_prices.iloc[i]
        price_range = high_price - low_price

        result.loc[result.index[i], ['frl_l1', 'frl_l2', 'frl_l3', 'frl_l4', 'frl_l5']] = [
            low_price + level_236 * price_range, 
            low_price + level_382 * price_range,
            low_price + level_500 * price_range,
            low_price + level_618 * price_range,
            high_price
        ]
    return result