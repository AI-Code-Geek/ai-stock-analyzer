import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class EnhancedPatternRecognition:
    """Enhanced pattern recognition with support/resistance and demand/supply zones"""
    
    def __init__(self):
        pass
    
    def identify_support_resistance(self, data: pd.DataFrame, window: int = 20, min_touches: int = 2) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        try:
            if len(data) < window * 2:
                return [], []
            
            highs = data['high'].rolling(window=window, center=True).max()
            lows = data['low'].rolling(window=window, center=True).min()
            
            # Find peaks and troughs
            peaks = data[data['high'] == highs]['high'].dropna()
            troughs = data[data['low'] == lows]['low'].dropna()
            
            # Group similar levels
            resistance_levels = []
            support_levels = []
            
            tolerance = 0.02  # 2% tolerance for grouping levels
            
            for price in peaks.values:
                similar_count = len(peaks[(peaks >= price * (1 - tolerance)) & 
                                        (peaks <= price * (1 + tolerance))])
                if similar_count >= min_touches:
                    resistance_levels.append(price)
            
            for price in troughs.values:
                similar_count = len(troughs[(troughs >= price * (1 - tolerance)) & 
                                          (troughs <= price * (1 + tolerance))])
                if similar_count >= min_touches:
                    support_levels.append(price)
            
            return list(set(resistance_levels)), list(set(support_levels))
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return [], []
    
    def identify_demand_supply_zones(self, data: pd.DataFrame, strength_threshold: float = 0.05) -> List[Dict]:
        """Identify demand and supply zones"""
        try:
            if len(data) < 30:
                return []
            
            zones = []
            
            for i in range(20, len(data) - 5):
                # Look for strong moves
                current_close = data['close'].iloc[i]
                prev_close = data['close'].iloc[i-1]
                volume_current = data['volume'].iloc[i]
                volume_avg = data['volume'].iloc[i-20:i].mean()
                
                if volume_avg == 0:
                    continue
                
                # Strong bullish move (demand zone)
                if (current_close > prev_close * (1 + strength_threshold) and 
                    volume_current > volume_avg * 1.5):
                    zone_low = data['low'].iloc[i-5:i+1].min()
                    zone_high = data['high'].iloc[i-5:i+1].max()
                    zones.append({
                        'type': 'demand',
                        'low': zone_low,
                        'high': zone_high,
                        'date': data.index[i],
                        'strength': volume_current / volume_avg
                    })
                
                # Strong bearish move (supply zone)
                elif (current_close < prev_close * (1 - strength_threshold) and 
                      volume_current > volume_avg * 1.5):
                    zone_low = data['low'].iloc[i-5:i+1].min()
                    zone_high = data['high'].iloc[i-5:i+1].max()
                    zones.append({
                        'type': 'supply',
                        'low': zone_low,
                        'high': zone_high,
                        'date': data.index[i],
                        'strength': volume_current / volume_avg
                    })
            
            return zones
            
        except Exception as e:
            logger.error(f"Error identifying demand/supply zones: {e}")
            return []

class SignalGenerator:
    """Generate trading signals based on multi-timeframe analysis"""
    
    def __init__(self):
        pass
    
    def generate_trading_signals(self, daily_data: pd.DataFrame, weekly_data: pd.DataFrame, ai_score: float) -> List[Dict]:
        """Generate trading signals based on daily and weekly analysis"""
        try:
            signals = []
            
            if len(daily_data) < 50 or len(weekly_data) < 20:
                return [{'type': 'INSUFFICIENT_DATA', 'strength': 'N/A', 
                        'reason': 'Not enough data for analysis'}]
            
            # Get current values
            current_price = daily_data['close'].iloc[-1]
            
            # Calculate EMAs if not present
            if 'ema_20' not in daily_data.columns:
                daily_data['ema_20'] = daily_data['close'].ewm(span=20).mean()
            if 'ema_50' not in daily_data.columns:
                daily_data['ema_50'] = daily_data['close'].ewm(span=50).mean()
            
            daily_20_ema = daily_data['ema_20'].iloc[-1]
            daily_50_ema = daily_data['ema_50'].iloc[-1]
            
            # Calculate RSI if not present
            if 'rsi' not in daily_data.columns:
                delta = daily_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                daily_data['rsi'] = 100 - (100 / (1 + rs))
            
            daily_rsi = daily_data['rsi'].iloc[-1]
            
            # Determine trends
            weekly_trend = self._determine_trend(weekly_data)
            daily_trend = self._determine_trend(daily_data)
            
            # Enhanced signal generation logic with AI scoring
            if weekly_trend == "Bullish":
                if (current_price > daily_20_ema and 
                    daily_rsi < 70 and 
                    daily_trend in ["Bullish", "Neutral"] and
                    ai_score > 60):
                    signals.append({
                        'type': 'STRONG BUY',
                        'strength': f'AI Score: {ai_score:.0f}',
                        'reason': f'Weekly uptrend + Daily bullish setup + AI confirmation'
                    })
                elif (current_price > daily_20_ema and 
                      daily_rsi < 70 and 
                      ai_score > 45):
                    signals.append({
                        'type': 'BUY',
                        'strength': f'AI Score: {ai_score:.0f}',
                        'reason': 'Weekly uptrend + Daily above 20 EMA + Moderate AI score'
                    })
                elif (current_price < daily_20_ema and 
                      daily_rsi < 40 and
                      ai_score > 35):
                    signals.append({
                        'type': 'BUY DIP',
                        'strength': f'AI Score: {ai_score:.0f}',
                        'reason': 'Weekly uptrend + Daily pullback + Oversold RSI'
                    })
            
            elif weekly_trend == "Bearish":
                if (current_price < daily_20_ema and 
                    daily_rsi > 30 and 
                    daily_trend in ["Bearish", "Neutral"] and
                    ai_score > 60):
                    signals.append({
                        'type': 'STRONG SELL',
                        'strength': f'AI Score: {ai_score:.0f}',
                        'reason': 'Weekly downtrend + Daily bearish setup + AI confirmation'
                    })
                elif (current_price < daily_20_ema and 
                      daily_rsi > 30 and
                      ai_score > 45):
                    signals.append({
                        'type': 'SELL',
                        'strength': f'AI Score: {ai_score:.0f}',
                        'reason': 'Weekly downtrend + Daily below 20 EMA + Moderate AI score'
                    })
                elif (current_price > daily_20_ema and 
                      daily_rsi > 60 and
                      ai_score > 35):
                    signals.append({
                        'type': 'SELL RALLY',
                        'strength': f'AI Score: {ai_score:.0f}',
                        'reason': 'Weekly downtrend + Daily bounce + Overbought RSI'
                    })
            
            else:  # Neutral weekly trend
                if ai_score < 30:
                    signals.append({
                        'type': 'AVOID',
                        'strength': f'AI Score: {ai_score:.0f}',
                        'reason': 'Weekly trend unclear + Low AI confidence'
                    })
                else:
                    signals.append({
                        'type': 'WAIT',
                        'strength': f'AI Score: {ai_score:.0f}',
                        'reason': 'Weekly trend unclear - wait for direction'
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return [{'type': 'ERROR', 'strength': 'N/A', 'reason': f'Analysis error: {str(e)}'}]
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine trend direction"""
        try:
            if len(data) < 50:
                return "Neutral"
            
            current_price = data['close'].iloc[-1]
            
            # Calculate EMAs if not present
            if 'ema_20' not in data.columns:
                data['ema_20'] = data['close'].ewm(span=20).mean()
            if 'ema_50' not in data.columns:
                data['ema_50'] = data['close'].ewm(span=50).mean()
            
            ema_20 = data['ema_20'].iloc[-1]
            ema_50 = data['ema_50'].iloc[-1]
            
            if current_price > ema_20 > ema_50:
                return "Bullish"
            elif current_price < ema_20 < ema_50:
                return "Bearish"
            else:
                return "Neutral"
                
        except Exception as e:
            logger.error(f"Error determining trend: {e}")
            return "Neutral"