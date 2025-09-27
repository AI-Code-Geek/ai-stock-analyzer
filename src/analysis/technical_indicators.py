import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import ta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
# Volume SMA will be calculated manually as VolumeSMAIndicator doesn't exist in ta
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self):
        self.ema_periods = settings.EMA_PERIODS
        self.rsi_period = settings.RSI_PERIOD
        self.macd_fast = settings.MACD_FAST
        self.macd_slow = settings.MACD_SLOW
        self.macd_signal = settings.MACD_SIGNAL
    
    def calculate_emas(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        for period in self.ema_periods:
            ema_indicator = EMAIndicator(close=data['close'], window=period)
            result[f'ema_{period}'] = ema_indicator.ema_indicator()
        
        return result
    
    def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        rsi_indicator = RSIIndicator(close=data['close'], window=self.rsi_period)
        result['rsi'] = rsi_indicator.rsi()
        
        return result
    
    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        macd_indicator = MACD(
            close=data['close'],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        
        result['macd'] = macd_indicator.macd()
        result['macd_signal'] = macd_indicator.macd_signal()
        result['macd_histogram'] = macd_indicator.macd_diff()
        
        return result
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        # Volume SMA (manual calculation)
        result['volume_sma'] = data['volume'].rolling(window=20).mean()
        
        # Volume ratio (current volume / average volume)
        result['volume_ratio'] = data['volume'] / result['volume_sma']
        
        # Volume weighted average price (VWAP)
        result['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
        
        return result
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        result = data.copy()
        
        # Pivot points
        result['pivot_high'] = data['high'].rolling(window=window, center=True).max() == data['high']
        result['pivot_low'] = data['low'].rolling(window=window, center=True).min() == data['low']
        
        # Support and resistance levels
        pivot_highs = data.loc[result['pivot_high'], 'high'].dropna()
        pivot_lows = data.loc[result['pivot_low'], 'low'].dropna()
        
        if len(pivot_highs) > 0:
            result['resistance'] = pivot_highs.iloc[-1] if len(pivot_highs) > 0 else np.nan
        
        if len(pivot_lows) > 0:
            result['support'] = pivot_lows.iloc[-1] if len(pivot_lows) > 0 else np.nan
        
        return result
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        result = data.copy()
        
        # True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        result['atr'] = true_range.rolling(window=window).mean()
        
        # Percentage volatility
        result['volatility'] = data['close'].pct_change().rolling(window=window).std() * np.sqrt(252)
        
        return result
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            result = data.copy()
            
            # Calculate all indicators
            result = self.calculate_emas(result)
            result = self.calculate_rsi(result)
            result = self.calculate_macd(result)
            result = self.calculate_volume_indicators(result)
            result = self.calculate_support_resistance(result)
            result = self.calculate_volatility(result)
            
            logger.info(f"Calculated all technical indicators for {len(result)} data points")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data

class TrendAnalyzer:
    def __init__(self):
        self.ema_periods = settings.EMA_PERIODS
    
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else latest
            
            # EMA trend analysis
            ema_trends = {}
            for period in self.ema_periods:
                ema_col = f'ema_{period}'
                if ema_col in latest:
                    current_ema = latest[ema_col]
                    prev_ema = prev[ema_col] if ema_col in prev else current_ema
                    
                    ema_trends[f'ema_{period}_trend'] = 'bullish' if current_ema > prev_ema else 'bearish'
                    ema_trends[f'ema_{period}_slope'] = (current_ema - prev_ema) / prev_ema if prev_ema != 0 else 0
            
            # Price vs EMA positioning
            price_positioning = {}
            current_price = latest['close']
            
            for period in self.ema_periods:
                ema_col = f'ema_{period}'
                if ema_col in latest:
                    ema_value = latest[ema_col]
                    price_positioning[f'price_vs_ema_{period}'] = 'above' if current_price > ema_value else 'below'
                    price_positioning[f'price_ema_{period}_distance'] = (current_price - ema_value) / ema_value if ema_value != 0 else 0
            
            # Overall trend determination
            bullish_signals = sum(1 for trend in ema_trends.values() if isinstance(trend, str) and trend == 'bullish')
            bearish_signals = sum(1 for trend in ema_trends.values() if isinstance(trend, str) and trend == 'bearish')
            
            if bullish_signals > bearish_signals:
                overall_trend = 'bullish'
            elif bearish_signals > bullish_signals:
                overall_trend = 'bearish'
            else:
                overall_trend = 'neutral'
            
            return {
                'overall_trend': overall_trend,
                'ema_trends': ema_trends,
                'price_positioning': price_positioning,
                'trend_strength': abs(bullish_signals - bearish_signals) / len(self.ema_periods)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'overall_trend': 'neutral', 'ema_trends': {}, 'price_positioning': {}, 'trend_strength': 0}
    
    def analyze_timeframe_alignment(self, daily_data: pd.DataFrame, weekly_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            daily_trend = self.analyze_trend(daily_data)
            weekly_trend = self.analyze_trend(weekly_data)
            
            # Check if daily and weekly trends align
            alignment = daily_trend['overall_trend'] == weekly_trend['overall_trend']
            
            # Calculate alignment score
            alignment_score = 0
            if alignment:
                alignment_score = min(daily_trend['trend_strength'], weekly_trend['trend_strength'])
            
            return {
                'daily_trend': daily_trend,
                'weekly_trend': weekly_trend,
                'trend_alignment': alignment,
                'alignment_score': alignment_score,
                'timeframe_sync': 'strong' if alignment_score > 0.6 else 'weak' if alignment else 'conflicting'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe alignment: {e}")
            return {
                'daily_trend': {'overall_trend': 'neutral'},
                'weekly_trend': {'overall_trend': 'neutral'},
                'trend_alignment': False,
                'alignment_score': 0,
                'timeframe_sync': 'unknown'
            }

class PatternRecognition:
    def __init__(self):
        pass
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        try:
            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else latest
            
            patterns = {}
            
            # Doji pattern
            body_size = abs(latest['close'] - latest['open'])
            candle_range = latest['high'] - latest['low']
            patterns['doji'] = body_size < (candle_range * 0.1) if candle_range > 0 else False
            
            # Hammer pattern
            lower_shadow = latest['open'] - latest['low'] if latest['open'] < latest['close'] else latest['close'] - latest['low']
            upper_shadow = latest['high'] - latest['open'] if latest['open'] > latest['close'] else latest['high'] - latest['close']
            patterns['hammer'] = lower_shadow > (body_size * 2) and upper_shadow < body_size
            
            # Shooting star pattern
            patterns['shooting_star'] = upper_shadow > (body_size * 2) and lower_shadow < body_size
            
            # Engulfing patterns
            if len(data) > 1:
                bullish_engulfing = (prev['close'] < prev['open'] and 
                                   latest['close'] > latest['open'] and
                                   latest['open'] < prev['close'] and
                                   latest['close'] > prev['open'])
                
                bearish_engulfing = (prev['close'] > prev['open'] and 
                                   latest['close'] < latest['open'] and
                                   latest['open'] > prev['close'] and
                                   latest['close'] < prev['open'])
                
                patterns['bullish_engulfing'] = bullish_engulfing
                patterns['bearish_engulfing'] = bearish_engulfing
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
            return {}
    
    def detect_chart_patterns(self, data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        try:
            patterns = {}
            
            if len(data) < window:
                return patterns
            
            recent_data = data.tail(window)
            highs = recent_data['high']
            lows = recent_data['low']
            closes = recent_data['close']
            
            # Breakout detection
            resistance_level = highs.iloc[:-5].max()
            support_level = lows.iloc[:-5].min()
            current_price = closes.iloc[-1]
            
            patterns['breakout'] = current_price > resistance_level
            patterns['breakdown'] = current_price < support_level
            
            # Trend channel
            price_trend = np.polyfit(range(len(closes)), closes.values, 1)[0]
            patterns['uptrend_channel'] = price_trend > 0
            patterns['downtrend_channel'] = price_trend < 0
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
            return {}