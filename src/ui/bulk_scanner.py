import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from typing import List, Dict, Any
import time
import logging

from src.data.yahoo_finance import YahooFinanceClient
from src.analysis.technical_indicators import TechnicalIndicators
from src.ai.ollama_client import AIAnalyzer
from src.utils.custom_css import get_signal_emoji, get_trend_color, format_ai_score_class

logger = logging.getLogger(__name__)

class BulkStockScanner:
    """Advanced bulk scanning functionality"""
    
    def __init__(self):
        self.yahoo_client = YahooFinanceClient()
        self.technical_indicators = TechnicalIndicators()
        self.ai_analyzer = AIAnalyzer()
    
    def scan_single_stock(self, symbol: str) -> Dict[str, Any]:
        """Scan a single stock and return analysis results"""
        try:
            # Fetch data
            daily_data = self.yahoo_client.get_stock_data(symbol, period="1y", interval="1d")
            if daily_data is None or len(daily_data) < 100:
                return None
            
            weekly_data = self.yahoo_client.get_weekly_data(symbol, period="2y")
            if weekly_data is None or len(weekly_data) < 20:
                return None
            
            # Calculate indicators
            daily_with_indicators = self.technical_indicators.calculate_all_indicators(daily_data)
            weekly_with_indicators = self.technical_indicators.calculate_all_indicators(weekly_data)
            
            # Get current price info
            current_price = daily_with_indicators['close'].iloc[-1]
            prev_close = daily_with_indicators['close'].iloc[-2]
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # Get latest indicators
            latest = daily_with_indicators.iloc[-1]
            rsi = latest.get('rsi', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            volatility = latest.get('volatility', 0.2)
            
            # Determine trends
            weekly_trend = self._determine_trend(weekly_with_indicators)
            daily_trend = self._determine_trend(daily_with_indicators)
            
            # Calculate proper alignment score
            alignment_score = 0
            if daily_trend == weekly_trend and daily_trend != 'Neutral':
                alignment_score = 0.8  # Strong alignment
            elif daily_trend != 'Neutral' and weekly_trend != 'Neutral':
                alignment_score = 0.3  # Some alignment
            
            # Get price positioning
            ema_20 = daily_with_indicators.get('ema_20', daily_with_indicators['close'].ewm(span=20).mean()).iloc[-1]
            ema_50 = daily_with_indicators.get('ema_50', daily_with_indicators['close'].ewm(span=50).mean()).iloc[-1]
            
            price_vs_ema20 = 'above' if current_price > ema_20 else 'below'
            price_vs_ema50 = 'above' if current_price > ema_50 else 'below'
            
            # Get MACD signal (simplified)
            macd_signal = 'neutral'
            if 'macd' in daily_with_indicators.columns and 'macd_signal' in daily_with_indicators.columns:
                macd_val = daily_with_indicators['macd'].iloc[-1]
                macd_sig = daily_with_indicators['macd_signal'].iloc[-1]
                if macd_val > macd_sig:
                    macd_signal = 'bullish'
                elif macd_val < macd_sig:
                    macd_signal = 'bearish'
            
            # Calculate AI score with complete data (convert trends to lowercase for AI scoring)
            analysis_data = {
                'current_price': current_price,
                'daily_trend': {'overall_trend': daily_trend.lower()},
                'weekly_trend': {'overall_trend': weekly_trend.lower()},
                'alignment_score': alignment_score,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'price_positioning': {
                    'price_vs_ema_20': price_vs_ema20,
                    'price_vs_ema_50': price_vs_ema50
                },
                'macd_signal': macd_signal,
                'patterns': {}  # Empty patterns for now
            }
            
            ai_results = self.ai_analyzer.calculate_ai_score(analysis_data)
            ai_score = ai_results.get('ai_score', 0)
            
            # Filter low-scoring stocks
            if ai_score < 30:
                return None
            
            # Determine if penny stock
            is_penny_stock = current_price < 5.0
            
            # Get signal strength
            signal_strength = ai_results.get('signal_classification', 'Unknown')
            signal_emoji = get_signal_emoji(signal_strength) if hasattr(st, 'session_state') else '📊'
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'change_pct': change_pct,
                'is_penny_stock': is_penny_stock,
                'ai_score': ai_score,
                'signal_strength': signal_strength,
                'signal_emoji': signal_emoji,
                'weekly_trend': weekly_trend,
                'daily_trend': daily_trend,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'components': ai_results.get('component_scores', {})
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None
    
    def parallel_stock_scan(self, symbols: List[str], max_workers: int = 10) -> List[Dict]:
        """Scan multiple stocks in parallel with progress tracking"""
        results = []
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.scan_single_stock, symbol): symbol 
                for symbol in symbols
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        # Show real-time results
                        self._update_live_results(results_container, results)
                    
                    completed += 1
                    
                    # Update progress
                    progress = completed / len(symbols)
                    progress_bar.progress(progress)
                    status_text.text(f"Scanning: {symbol} ({completed}/{len(symbols)}) - Found {len(results)} qualifying stocks")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    completed += 1
                    progress = completed / len(symbols)
                    progress_bar.progress(progress)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return sorted(results, key=lambda x: x['ai_score'], reverse=True)
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine trend direction"""
        try:
            if len(data) < 50:
                return "Neutral"
            
            current_price = data['close'].iloc[-1]
            ema_20 = data.get('ema_20', data['close'].ewm(span=20).mean()).iloc[-1]
            ema_50 = data.get('ema_50', data['close'].ewm(span=50).mean()).iloc[-1]
            
            if current_price > ema_20 > ema_50:
                return "Bullish"
            elif current_price < ema_20 < ema_50:
                return "Bearish"
            else:
                return "Neutral"
                
        except Exception:
            return "Neutral"
    
    def _update_live_results(self, container, results: List[Dict]):
        """Update live results display during scanning"""
        try:
            if not results:
                return
            
            # Show top 10 results so far
            top_results = sorted(results, key=lambda x: x['ai_score'], reverse=True)[:10]
            
            with container.container():
                st.subheader(f"📊 Live Results ({len(results)} found)")
                
                # Create quick summary
                cols = st.columns(4)
                
                with cols[0]:
                    avg_score = np.mean([r['ai_score'] for r in results])
                    st.metric("Avg AI Score", f"{avg_score:.1f}")
                
                with cols[1]:
                    strong_signals = len([r for r in results if r['ai_score'] >= 75])
                    st.metric("Strong Signals", strong_signals)
                
                with cols[2]:
                    bullish_count = len([r for r in results if r['daily_trend'] == 'Bullish'])
                    st.metric("Bullish Stocks", bullish_count)
                
                with cols[3]:
                    penny_count = len([r for r in results if r['is_penny_stock']])
                    st.metric("Penny Stocks", penny_count)
                
                # Show top results table
                if top_results:
                    df_data = []
                    for result in top_results:
                        df_data.append({
                            'Symbol': result['symbol'],
                            'Price': f"${result['current_price']:.2f}",
                            'Change %': f"{result['change_pct']:+.1f}%",
                            'AI Score': f"{result['ai_score']:.2f}",
                            'Signal': f"{result['signal_emoji']} {result['signal_strength']}",
                            'Trend': result['daily_trend']
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error updating live results: {e}")
    
    def create_scan_results_table(self, results: List[Dict]) -> pd.DataFrame:
        """Create a formatted results table"""
        if not results:
            return pd.DataFrame()
        
        df_data = []
        for result in results:
            df_data.append({
                'Symbol': result['symbol'],
                'Price': round(result['current_price'], 2),
                'Change %': round(result['change_pct'], 2),
                'AI Score': round(result['ai_score'], 2),
                'Signal': f"{result['signal_emoji']} {result['signal_strength']}",
                'Weekly Trend': result['weekly_trend'],
                'Daily Trend': result['daily_trend'],
                'RSI': round(result['rsi'], 1),
                'Volume Ratio': round(result['volume_ratio'], 2),
                'Volatility': round(result['volatility'] * 100, 1),  # Convert to percentage
                'Type': 'Penny' if result['is_penny_stock'] else 'Regular'
            })
        
        df = pd.DataFrame(df_data)
        return df.sort_values('AI Score', ascending=False)
    
    def create_results_visualization(self, results: List[Dict]):
        """Create visualization charts for scan results"""
        if not results:
            st.warning("No results to visualize")
            return
        
        df = pd.DataFrame(results)
        
        # Create subplot layout
        col1, col2 = st.columns(2)
        
        with col1:
            # AI Score Distribution
            fig = px.histogram(
                df, 
                x='ai_score', 
                nbins=20,
                title="AI Score Distribution",
                labels={'ai_score': 'AI Score', 'count': 'Number of Stocks'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trend Distribution
            trend_counts = df['daily_trend'].value_counts()
            fig = px.pie(
                values=trend_counts.values,
                names=trend_counts.index,
                title="Daily Trend Distribution",
                color_discrete_map={
                    'Bullish': '#28a745',
                    'Bearish': '#dc3545',
                    'Neutral': '#ffc107'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top performers bar chart
        st.subheader("🏆 Top 20 Performers by AI Score")
        top_20 = df.nlargest(20, 'ai_score')
        
        fig = px.bar(
            top_20,
            x='symbol',
            y='ai_score',
            color='ai_score',
            title="Top 20 Stocks by AI Score",
            labels={'symbol': 'Stock Symbol', 'ai_score': 'AI Score'},
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("📊 Correlation Analysis")
        
        # Prepare numeric data for correlation
        numeric_cols = ['ai_score', 'current_price', 'change_pct', 'rsi', 'volume_ratio', 'volatility']
        corr_data = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_data,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix of Key Metrics",
            color_continuous_scale='RdBu'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)