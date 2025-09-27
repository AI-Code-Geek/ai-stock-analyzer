import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

from src.analysis.enhanced_patterns import EnhancedPatternRecognition, SignalGenerator

logger = logging.getLogger(__name__)

class EnhancedChartCreator:
    """Create advanced interactive charts with technical analysis"""
    
    def __init__(self):
        self.pattern_recognition = EnhancedPatternRecognition()
        self.signal_generator = SignalGenerator()
    
    def create_comprehensive_chart(self, symbol: str, daily_data: pd.DataFrame, weekly_data: pd.DataFrame, ai_score: float = 0) -> go.Figure:
        """Create comprehensive technical analysis chart"""
        try:
            # Ensure proper column names (handle both upper and lower case)
            daily_data = self._normalize_column_names(daily_data)
            
            # Calculate technical indicators
            daily_data = self._calculate_indicators(daily_data)
            
            # Get support/resistance and zones
            support_levels, resistance_levels = self.pattern_recognition.identify_support_resistance(daily_data)
            zones = self.pattern_recognition.identify_demand_supply_zones(daily_data)
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} - Price & Moving Averages (AI Score: {ai_score:.2f})', 'Volume', 'RSI'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=daily_data.index,
                    open=daily_data['open'],
                    high=daily_data['high'],
                    low=daily_data['low'],
                    close=daily_data['close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'ema_20' in daily_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=daily_data.index, 
                        y=daily_data['ema_20'], 
                        line=dict(color='orange', width=2), 
                        name='EMA 20'
                    ),
                    row=1, col=1
                )
            
            if 'ema_50' in daily_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=daily_data.index, 
                        y=daily_data['ema_50'], 
                        line=dict(color='blue', width=2), 
                        name='EMA 50'
                    ),
                    row=1, col=1
                )
            
            if 'ema_200' in daily_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=daily_data.index, 
                        y=daily_data['ema_200'], 
                        line=dict(color='red', width=2), 
                        name='EMA 200'
                    ),
                    row=1, col=1
                )
            
            # Add support and resistance levels
            for level in support_levels[-5:]:  # Show last 5 levels
                fig.add_hline(
                    y=level, 
                    line_dash="dash", 
                    line_color="green", 
                    annotation_text=f"Support: ${level:.2f}", 
                    row=1
                )
            
            for level in resistance_levels[-5:]:  # Show last 5 levels
                fig.add_hline(
                    y=level, 
                    line_dash="dash", 
                    line_color="red", 
                    annotation_text=f"Resistance: ${level:.2f}", 
                    row=1
                )
            
            # Add demand and supply zones
            for zone in zones[-10:]:  # Show last 10 zones
                color = 'rgba(0,255,0,0.2)' if zone['type'] == 'demand' else 'rgba(255,0,0,0.2)'
                fig.add_shape(
                    type="rect",
                    x0=zone['date'],
                    y0=zone['low'],
                    x1=daily_data.index[-1],
                    y1=zone['high'],
                    fillcolor=color,
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=1,
                    col=1
                )
            
            # Add volume bars
            colors = ['red' if daily_data['close'].iloc[i] < daily_data['open'].iloc[i] else 'green' 
                      for i in range(len(daily_data))]
            
            fig.add_trace(
                go.Bar(
                    x=daily_data.index, 
                    y=daily_data['volume'], 
                    marker_color=colors, 
                    name='Volume', 
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Add volume average
            if 'volume_sma' in daily_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=daily_data.index, 
                        y=daily_data['volume_sma'], 
                        line=dict(color='purple', width=2), 
                        name='Volume SMA'
                    ),
                    row=2, col=1
                )
            
            # Add RSI
            if 'rsi' in daily_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=daily_data.index, 
                        y=daily_data['rsi'], 
                        line=dict(color='purple', width=2), 
                        name='RSI'
                    ),
                    row=3, col=1
                )
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, 
                              annotation_text="Overbought (70)")
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, 
                              annotation_text="Oversold (30)")
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3)
            
            # Update layout
            fig.update_layout(
                title=f"AI-Enhanced Technical Analysis - {symbol}",
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comprehensive chart for {symbol}: {e}")
            # Return a simple fallback chart
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def _normalize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase"""
        data = data.copy()
        data.columns = [col.lower() for col in data.columns]
        return data
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators if not present"""
        try:
            result = data.copy()
            
            # Calculate EMAs
            for period in [20, 50, 200]:
                if f'ema_{period}' not in result.columns:
                    result[f'ema_{period}'] = result['close'].ewm(span=period).mean()
            
            # Calculate RSI
            if 'rsi' not in result.columns:
                delta = result['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                result['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate volume SMA
            if 'volume_sma' not in result.columns:
                result['volume_sma'] = result['volume'].rolling(window=20).mean()
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return data