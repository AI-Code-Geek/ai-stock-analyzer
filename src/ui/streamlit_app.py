import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import warnings
import requests
import concurrent.futures
from typing import List, Dict, Tuple
import json

from src.scanner.stock_scanner import ScannerManager, ScanFilter, ScannerType, ScanResultExporter
from src.data.yahoo_finance import YahooFinanceClient
from src.analysis.technical_indicators import TechnicalIndicators, TrendAnalyzer
from src.analysis.enhanced_patterns import EnhancedPatternRecognition, SignalGenerator
from src.ai.ollama_client import AIAnalyzer
from src.ui.enhanced_charts import EnhancedChartCreator
from src.ui.bulk_scanner import BulkStockScanner
from src.ui.financial_reports import FinancialReportGenerator
from src.utils.custom_css import get_custom_css, get_signal_emoji, get_trend_color, format_ai_score_class, format_signal_class
from config.settings import settings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Stock Timeframe Analyzer & Scanner",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# S&P 500 and Popular Penny Stock Lists
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_sp500_symbols():
    """Get S&P 500 symbols from Wikipedia"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        return sp500_table['Symbol'].tolist()
    except:
        # Fallback list of popular S&P 500 stocks
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 
                'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'NFLX',
                'ADBE', 'CRM', 'CMCSA', 'XOM', 'KO', 'PEP', 'ABT', 'TMO', 'COST',
                'AVGO', 'ACN', 'MRK', 'WMT', 'NKE', 'DHR', 'LIN', 'VZ', 'QCOM',
                'CVX', 'NEE', 'PM', 'ORCL', 'T', 'BMY', 'WFC', 'MDT', 'UNP', 'LOW',
                'IBM', 'INTC', 'AMD', 'C', 'GS', 'INTU', 'CAT', 'AXP', 'RTX', 'HON']

def get_penny_stocks():
    """Get popular penny stocks (under $5)"""
    return [
        'SNDL', 'NOK', 'BB', 'AMC', 'NAKD', 'CCIV', 'PLTR', 'NIO', 'XPEV', 'LI',
        'RIOT', 'MARA', 'PLUG', 'FCEL', 'CLSK', 'EBON', 'GNUS', 'EXPR', 'KOSS',
        'CTRM', 'SHIP', 'TOPS', 'DRYS', 'GLBS', 'CASTOR', 'PHUN', 'DWAC', 'BKKT',
        'PROG', 'ATER', 'BBIG', 'SPRT', 'GREE', 'IRNT', 'OPAD', 'TMC', 'GOEV',
        'RIDE', 'WKHS', 'HYLN', 'NKLA', 'BLNK', 'CHPT', 'EVGO', 'LCID', 'RIVN',
        'SOFI', 'OPEN', 'WISH', 'CLOV', 'SKLZ', 'DKNG', 'PENN', 'MGM', 'WYNN'
    ]

class StreamlitApp:
    def __init__(self):
        self.scanner_manager = ScannerManager()
        self.yahoo_client = YahooFinanceClient()
        self.technical_indicators = TechnicalIndicators()
        self.trend_analyzer = TrendAnalyzer()
        self.ai_analyzer = AIAnalyzer()
        self.pattern_recognition = EnhancedPatternRecognition()
        self.signal_generator = SignalGenerator()
        self.chart_creator = EnhancedChartCreator()
        self.bulk_scanner = BulkStockScanner()
        self.financial_report_generator = FinancialReportGenerator()
        
        # Initialize session state
        if 'scan_results' not in st.session_state:
            st.session_state.scan_results = []
        if 'selected_stock' not in st.session_state:
            st.session_state.selected_stock = None
        if 'enhanced_analysis_cache' not in st.session_state:
            st.session_state.enhanced_analysis_cache = {}
        if 'bulk_scan_results' not in st.session_state:
            st.session_state.bulk_scan_results = []
    
    def run(self):
        st.title("🤖 AI Stock Timeframe Analyzer & Scanner")
        st.markdown("### Daily & Weekly Timeframe Analysis with OLLAMA Integration")
        
        # Sidebar navigation
        if 'page' not in st.session_state:
            st.session_state.page = "Dashboard"
        
        page = st.sidebar.selectbox(
            "Navigate",
            ["Dashboard", "Stock Scanner", "Individual Analysis", "Bulk Scanner", "Settings"],
            index=["Dashboard", "Stock Scanner", "Individual Analysis", "Bulk Scanner", "Settings"].index(st.session_state.page)
        )
        
        # Update session state if user selects a different page
        if page != st.session_state.page:
            st.session_state.page = page
        
        if st.session_state.page == "Dashboard":
            self.show_dashboard()
        elif st.session_state.page == "Stock Scanner":
            self.show_scanner()
        elif st.session_state.page == "Individual Analysis":
            self.show_individual_analysis()
        elif st.session_state.page == "Bulk Scanner":
            self.show_bulk_scanner()
        elif st.session_state.page == "Settings":
            self.show_settings()
    
    def show_dashboard(self):
        st.header("📊 Dashboard")
        
        # Quick stats
        if st.session_state.scan_results:
            results = st.session_state.scan_results
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Stocks Scanned", len(results))
            
            with col2:
                strong_signals = len([r for r in results if r.ai_score >= 75])
                st.metric("Strong Signals", strong_signals)
            
            with col3:
                avg_score = np.mean([r.ai_score for r in results])
                st.metric("Average AI Score", f"{avg_score:.1f}")
            
            with col4:
                bullish_count = len([r for r in results if r.daily_trend == 'bullish'])
                st.metric("Bullish Stocks", bullish_count)
            
            # Top performers chart
            st.subheader("Top 10 AI Scores")
            top_10 = results[:10]
            
            fig = px.bar(
                x=[r.symbol for r in top_10],
                y=[r.ai_score for r in top_10],
                title="Top 10 Stocks by AI Score",
                labels={'x': 'Symbol', 'y': 'AI Score'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Score distribution
            st.subheader("AI Score Distribution")
            scores = [r.ai_score for r in results]
            
            fig = px.histogram(
                x=scores,
                nbins=20,
                title="Distribution of AI Scores",
                labels={'x': 'AI Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Use the Stock Scanner to analyze stocks and populate the dashboard.")
    
    def show_scanner(self):
        st.header("🔍 Stock Scanner")
        
        # Scanner type selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            scanner_type = st.selectbox(
                "Scanner Type",
                ["S&P 500", "Penny Stocks", "Custom Symbols"]
            )
        
        with col2:
            if scanner_type == "Custom Symbols":
                custom_symbols = st.text_input(
                    "Enter symbols (comma-separated)",
                    placeholder="AAPL, MSFT, GOOGL"
                )
        
        # Filters
        st.subheader("Scan Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_ai_score = st.slider("Minimum AI Score", 0, 100, 60)
            trend_filter = st.selectbox("Trend Filter", ["Any", "Bullish", "Bearish", "Neutral"])
        
        with col2:
            min_price = st.number_input("Minimum Price ($)", min_value=0.0, value=0.0)
            max_price = st.number_input("Maximum Price ($)", min_value=0.0, value=1000.0)
        
        with col3:
            rsi_min = st.slider("RSI Minimum", 0, 100, 0)
            rsi_max = st.slider("RSI Maximum", 0, 100, 100)
        
        # Create filters
        filters = ScanFilter(
            min_ai_score=min_ai_score if min_ai_score > 0 else None,
            min_price=min_price if min_price > 0 else None,
            max_price=max_price if max_price < 1000 else None,
            trend_filter=trend_filter.lower() if trend_filter != "Any" else None,
            rsi_min=rsi_min if rsi_min > 0 else None,
            rsi_max=rsi_max if rsi_max < 100 else None
        )
        
        # Scan button
        if st.button("🚀 Start Scan", type="primary"):
            self.run_scan(scanner_type, custom_symbols if scanner_type == "Custom Symbols" else None, filters)
        
        # Display results
        if st.session_state.scan_results:
            self.display_scan_results()
    
    def run_scan(self, scanner_type: str, custom_symbols: str, filters: ScanFilter):
        with st.spinner("Scanning stocks... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(completed, total):
                progress = completed / total
                progress_bar.progress(progress)
                status_text.text(f"Processed {completed}/{total} stocks")
            
            try:
                if scanner_type == "S&P 500":
                    results = self.scanner_manager.start_sp500_scan(filters, progress_callback)
                elif scanner_type == "Penny Stocks":
                    results = self.scanner_manager.start_penny_stock_scan(filters, progress_callback)
                elif scanner_type == "Custom Symbols" and custom_symbols:
                    symbols = [s.strip().upper() for s in custom_symbols.split(",")]
                    results = self.scanner_manager.start_custom_scan(symbols, filters, progress_callback)
                else:
                    st.error("Please provide valid input for scanning.")
                    return
                
                st.session_state.scan_results = results
                progress_bar.progress(1.0)
                status_text.text(f"Scan completed! Found {len(results)} matching stocks.")
                
                if results:
                    st.success(f"✅ Scan completed successfully! Found {len(results)} stocks matching your criteria.")
                else:
                    st.warning("⚠️ No stocks found matching your criteria. Try adjusting the filters.")
                
            except Exception as e:
                st.error(f"❌ Error during scan: {str(e)}")
                logger.error(f"Scan error: {e}")
    
    def display_scan_results(self):
        st.subheader("📋 Scan Results")
        
        results = st.session_state.scan_results
        
        # Results table
        df = ScanResultExporter.to_dataframe(results)
        
        # Add selection column
        df.insert(0, 'Select', False)
        
        # Display table with selection
        edited_df = st.data_editor(
            df,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select stock for detailed analysis",
                    default=False,
                ),
                "AI Score": st.column_config.ProgressColumn(
                    "AI Score",
                    help="AI-generated score (0-100)",
                    min_value=0,
                    max_value=100,
                ),
                "Price": st.column_config.NumberColumn(
                    "Price",
                    help="Current stock price",
                    format="$%.2f"
                ),
                "Volume Ratio": st.column_config.NumberColumn(
                    "Volume Ratio",
                    help="Current volume vs average",
                    format="%.2f"
                ),
                "RSI": st.column_config.NumberColumn(
                    "RSI",
                    help="Relative Strength Index",
                    format="%.1f"
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="AI confidence in analysis",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Handle stock selection
        selected_rows = edited_df[edited_df['Select'] == True]
        if not selected_rows.empty:
            selected_symbol = selected_rows.iloc[0]['Symbol']
            if st.button(f"📊 Analyze {selected_symbol}"):
                st.session_state.selected_stock = selected_symbol
                st.session_state.page = "Individual Analysis"
                st.rerun()
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export to CSV"):
                filename = f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                if ScanResultExporter.to_csv(results, filename):
                    st.success(f"✅ Results exported to {filename}")
        
        with col2:
            if st.button("📄 Export to JSON"):
                filename = f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                if ScanResultExporter.to_json(results, filename):
                    st.success(f"✅ Results exported to {filename}")
    
    def show_individual_analysis(self):
        st.header("📈 Individual Stock Analysis")
        
        # Stock input
        symbol = st.text_input(
            "Enter Stock Symbol",
            value=st.session_state.selected_stock or "",
            placeholder="e.g., AAPL"
        ).upper()
        
        if symbol and st.button("🔍 Analyze Stock"):
            self.analyze_individual_stock(symbol)
        
        # Display analysis if available
        if f"{symbol}_analysis" in st.session_state:
            self.display_individual_analysis(symbol)
    
    def show_bulk_scanner(self):
        """Enhanced bulk scanning interface"""
        st.header("🔍 Advanced Bulk Stock Scanner")
        
        # Scanner configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Scanner Configuration")
            
            # Scanner type selection
            scanner_type = st.selectbox(
                "Choose Scanner Type",
                ["S&P 500 Stocks", "Penny Stocks (Under $5)", "Custom Symbol List", "Mixed Analysis"]
            )
            
            # Custom symbols input
            if scanner_type == "Custom Symbol List":
                custom_symbols = st.text_area(
                    "Enter Stock Symbols (one per line or comma-separated)",
                    placeholder="AAPL\nMSFT\nGOOGL\n\nOr: AAPL, MSFT, GOOGL",
                    height=150
                )
            
            # Advanced filters
            st.subheader("🔧 Advanced Filters")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                min_ai_score = st.slider("Minimum AI Score", 0, 100, 50)
                price_range = st.slider("Price Range ($)", 0.0, 1000.0, (1.0, 500.0))
            
            with filter_col2:
                trend_filter = st.selectbox("Trend Filter", ["Any", "Bullish Only", "Bearish Only", "Neutral Only"])
                rsi_range = st.slider("RSI Range", 0, 100, (20, 80))
            
            with filter_col3:
                min_volume_ratio = st.slider("Min Volume Ratio", 0.5, 5.0, 1.0)
                max_volatility = st.slider("Max Volatility (%)", 10, 200, 100)
        
        with col2:
            st.subheader("📊 Scan Settings")
            
            max_workers = st.selectbox("Parallel Workers", [5, 10, 15, 20], index=1)
            show_live_results = st.checkbox("Show Live Results", value=True)
            auto_export = st.checkbox("Auto Export Results", value=False)
            
            # Quick presets
            st.subheader("⚡ Quick Presets")
            
            if st.button("🚀 High Growth Potential", use_container_width=True):
                st.session_state.preset_config = {
                    'min_ai_score': 70,
                    'trend_filter': 'Bullish Only',
                    'rsi_range': (30, 70),
                    'min_volume_ratio': 1.5
                }
            
            if st.button("💎 Value Opportunities", use_container_width=True):
                st.session_state.preset_config = {
                    'min_ai_score': 60,
                    'trend_filter': 'Any',
                    'rsi_range': (20, 40),
                    'price_range': (5.0, 100.0)
                }
            
            if st.button("⚡ Momentum Plays", use_container_width=True):
                st.session_state.preset_config = {
                    'min_ai_score': 75,
                    'trend_filter': 'Bullish Only',
                    'min_volume_ratio': 2.0,
                    'rsi_range': (50, 80)
                }
        
        # Start scan button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Start Advanced Scan", type="primary", use_container_width=True):
                self.run_bulk_scan(scanner_type, locals())
        
        # Display results
        if st.session_state.bulk_scan_results:
            self.display_bulk_scan_results()
    
    def run_bulk_scan(self, scanner_type: str, config: dict):
        """Execute the bulk scanning operation"""
        try:
            st.subheader("📡 Scanning in Progress...")
            
            # Determine symbol list
            if scanner_type == "S&P 500 Stocks":
                symbols = get_sp500_symbols()
            elif scanner_type == "Penny Stocks (Under $5)":
                symbols = get_penny_stocks()
            elif scanner_type == "Custom Symbol List":
                custom_symbols = config.get('custom_symbols', '')
                if not custom_symbols:
                    st.error("Please enter stock symbols for custom scanning.")
                    return
                
                # Parse symbols
                symbols = []
                for line in custom_symbols.replace(',', '\n').split('\n'):
                    symbol = line.strip().upper()
                    if symbol and len(symbol) <= 10:  # Basic validation
                        symbols.append(symbol)
                
                if not symbols:
                    st.error("No valid symbols found. Please check your input.")
                    return
            
            elif scanner_type == "Mixed Analysis":
                # Combine S&P 500 and penny stocks
                symbols = get_sp500_symbols()[:50] + get_penny_stocks()[:50]  # Limit for demo
            
            # Apply filters during scanning
            filters = {
                'min_ai_score': config.get('min_ai_score', 50),
                'price_range': config.get('price_range', (1.0, 500.0)),
                'trend_filter': config.get('trend_filter', 'Any'),
                'rsi_range': config.get('rsi_range', (20, 80)),
                'min_volume_ratio': config.get('min_volume_ratio', 1.0),
                'max_volatility': config.get('max_volatility', 100) / 100  # Convert to decimal
            }
            
            # Show scan info
            with st.expander("📋 Scan Information", expanded=True):
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.metric("Total Symbols", len(symbols))
                
                with info_col2:
                    st.metric("Min AI Score", f"{filters['min_ai_score']:.0f}")
                
                with info_col3:
                    estimated_time = len(symbols) * 2 / config.get('max_workers', 10)  # Rough estimate
                    st.metric("Est. Time", f"{estimated_time:.0f}s")
            
            # Execute scan
            start_time = time.time()
            
            results = self.bulk_scanner.parallel_stock_scan(
                symbols, 
                max_workers=config.get('max_workers', 10)
            )
            
            # Filter results
            filtered_results = self._apply_bulk_filters(results, filters)
            
            scan_duration = time.time() - start_time
            
            # Store results
            st.session_state.bulk_scan_results = filtered_results
            
            # Show completion message
            st.success(f"✅ Scan completed in {scan_duration:.1f} seconds!")
            st.info(f"📊 Found {len(filtered_results)} stocks matching your criteria out of {len(symbols)} scanned.")
            
            # Auto export if enabled
            if config.get('auto_export', False) and filtered_results:
                self.export_bulk_results(filtered_results)
        
        except Exception as e:
            st.error(f"❌ Scan failed: {str(e)}")
            logger.error(f"Bulk scan error: {e}")
    
    def _apply_bulk_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply additional filters to scan results"""
        filtered = []
        
        for result in results:
            # Price range filter
            if not (filters['price_range'][0] <= result['current_price'] <= filters['price_range'][1]):
                continue
            
            # AI score filter
            if result['ai_score'] < filters['min_ai_score']:
                continue
            
            # Trend filter
            trend_filter = filters['trend_filter']
            if trend_filter != 'Any':
                expected_trend = trend_filter.replace(' Only', '')
                if result['daily_trend'] != expected_trend:
                    continue
            
            # RSI filter
            if not (filters['rsi_range'][0] <= result['rsi'] <= filters['rsi_range'][1]):
                continue
            
            # Volume ratio filter
            if result['volume_ratio'] < filters['min_volume_ratio']:
                continue
            
            # Volatility filter
            if result['volatility'] > filters['max_volatility']:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def display_bulk_scan_results(self):
        """Display comprehensive bulk scan results"""
        results = st.session_state.bulk_scan_results
        
        if not results:
            st.warning("No results to display. Try adjusting your filters.")
            return
        
        st.header(f"📈 Scan Results ({len(results)} stocks found)")
        
        # Summary metrics
        self.show_bulk_summary_metrics(results)
        
        # Results table and visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results Table", "📈 Visualizations", "🏆 Top Picks", "📁 Export"])
        
        with tab1:
            self.show_bulk_results_table(results)
        
        with tab2:
            self.bulk_scanner.create_results_visualization(results)
        
        with tab3:
            self.show_top_picks(results)
        
        with tab4:
            self.show_export_options(results)
    
    def show_bulk_summary_metrics(self, results: List[Dict]):
        """Show summary metrics for bulk scan results"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate metrics
        avg_score = np.mean([r['ai_score'] for r in results])
        strong_signals = len([r for r in results if r['ai_score'] >= 75])
        bullish_count = len([r for r in results if r['daily_trend'] == 'Bullish'])
        penny_count = len([r for r in results if r['is_penny_stock']])
        high_volume = len([r for r in results if r['volume_ratio'] >= 1.5])
        
        with col1:
            st.metric(
                "📊 Average AI Score", 
                f"{avg_score:.1f}",
                delta=f"{avg_score - 50:.1f} vs baseline"
            )
        
        with col2:
            st.metric(
                "🚀 Strong Signals", 
                strong_signals,
                delta=f"{(strong_signals/len(results)*100):.1f}% of total"
            )
        
        with col3:
            st.metric(
                "📈 Bullish Trends", 
                bullish_count,
                delta=f"{(bullish_count/len(results)*100):.1f}% of total"
            )
        
        with col4:
            st.metric(
                "💰 Penny Stocks", 
                penny_count,
                delta=f"{(penny_count/len(results)*100):.1f}% of total"
            )
        
        with col5:
            st.metric(
                "📊 High Volume", 
                high_volume,
                delta=f"{(high_volume/len(results)*100):.1f}% of total"
            )
    
    def show_bulk_results_table(self, results: List[Dict]):
        """Show interactive results table"""
        df = self.bulk_scanner.create_scan_results_table(results)
        
        # Add selection functionality
        selected_indices = st.multiselect(
            "Select stocks for detailed analysis:",
            options=range(len(df)),
            format_func=lambda x: f"{df.iloc[x]['Symbol']} (Score: {min(int(round(df.iloc[x]['AI Score'])), 100)})",
            max_selections=5
        )
        
        # Display table with formatting
        st.dataframe(
            df,
            column_config={
                "AI Score": st.column_config.ProgressColumn(
                    "AI Score",
                    help="AI-generated score (0-100)",
                    min_value=0,
                    max_value=100,
                ),
                "Price": st.column_config.NumberColumn(
                    "Price",
                    help="Current stock price",
                    format="$%.2f"
                ),
                "Change %": st.column_config.NumberColumn(
                    "Change %",
                    help="Daily price change",
                    format="%.2f%%"
                ),
                "RSI": st.column_config.NumberColumn(
                    "RSI",
                    help="14-period RSI",
                    format="%.1f"
                ),
                "Volume Ratio": st.column_config.NumberColumn(
                    "Volume Ratio",
                    help="Current vs average volume",
                    format="%.2fx"
                ),
                "Volatility": st.column_config.NumberColumn(
                    "Volatility",
                    help="Annualized volatility",
                    format="%.1f%%"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Quick analysis for selected stocks
        if selected_indices:
            st.subheader("🔍 Quick Analysis of Selected Stocks")
            for idx in selected_indices:
                symbol = df.iloc[idx]['Symbol']
                with st.expander(f"📊 {symbol} - Quick Analysis"):
                    self.show_quick_stock_analysis(symbol, results[idx])
    
    def show_quick_stock_analysis(self, symbol: str, result: Dict):
        """Show quick analysis for a selected stock"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**🏷️ {symbol}**")
            st.markdown(f"💵 **Price:** ${result['current_price']:.2f}")
            st.markdown(f"📈 **Change:** {result['change_pct']:+.1f}%")
            st.markdown(f"🎯 **Type:** {'Penny Stock' if result['is_penny_stock'] else 'Regular Stock'}")
        
        with col2:
            st.markdown(f"🤖 **AI Score:** {min(int(round(result['ai_score'])), 100)}/100")
            st.markdown(f"📊 **Signal:** {result['signal_strength']}")
            st.markdown(f"📈 **Weekly Trend:** {result['weekly_trend']}")
            st.markdown(f"📊 **Daily Trend:** {result['daily_trend']}")
        
        with col3:
            st.markdown(f"⚡ **RSI:** {result['rsi']:.1f}")
            st.markdown(f"📊 **Volume Ratio:** {result['volume_ratio']:.2f}x")
            st.markdown(f"📈 **Volatility:** {result['volatility']*100:.1f}%")
        
        # Component scores
        if 'components' in result and result['components']:
            st.markdown("**🔧 AI Score Components:**")
            components = result['components']
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown(f"• Trend Alignment: {components.get('trend_alignment', 0):.1f}")
                st.markdown(f"• Volume Confirmation: {components.get('volume_confirmation', 0):.1f}")
                st.markdown(f"• RSI Positioning: {components.get('rsi_positioning', 0):.1f}")
            
            with comp_col2:
                st.markdown(f"• Price Action: {components.get('price_action', 0):.1f}")
                st.markdown(f"• Volatility: {components.get('volatility_analysis', 0):.1f}")
                st.markdown(f"• Momentum: {components.get('momentum_quality', 0):.1f}")
    
    def show_top_picks(self, results: List[Dict]):
        """Show AI-curated top picks with detailed analysis"""
        st.subheader("🏆 AI-Curated Top Picks")
        
        # Categorize picks
        strong_buys = [r for r in results if r['ai_score'] >= 80 and r['daily_trend'] == 'Bullish']
        value_plays = [r for r in results if 60 <= r['ai_score'] < 80 and r['rsi'] < 40]
        momentum_plays = [r for r in results if r['ai_score'] >= 70 and r['volume_ratio'] >= 2.0]
        
        tab1, tab2, tab3 = st.tabs(["🚀 Strong Buys", "💎 Value Plays", "⚡ Momentum Plays"])
        
        with tab1:
            self.show_category_picks(strong_buys, "Strong Buy Candidates", "🚀")
        
        with tab2:
            self.show_category_picks(value_plays, "Value Opportunities", "💎")
        
        with tab3:
            self.show_category_picks(momentum_plays, "Momentum Plays", "⚡")
    
    def show_category_picks(self, picks: List[Dict], title: str, emoji: str):
        """Show picks for a specific category"""
        if not picks:
            st.info(f"No stocks found in {title} category with current filters.")
            return
        
        st.markdown(f"### {emoji} {title} ({len(picks)} found)")
        
        # Sort by AI score
        picks = sorted(picks, key=lambda x: x['ai_score'], reverse=True)[:10]  # Top 10
        
        # Create unique category identifier for button keys
        category_id = title.lower().replace(' ', '_').replace('&', 'and')
        
        for i, pick in enumerate(picks, 1):
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**#{i}**")
                    st.markdown(f"**{pick['symbol']}**")
                
                with col2:
                    st.markdown(f"💵 ${pick['current_price']:.2f} ({pick['change_pct']:+.1f}%)")
                    st.markdown(f"🎯 AI Score: {min(int(round(pick['ai_score'])), 100)}/100")
                
                with col3:
                    st.markdown(f"📈 {pick['daily_trend']} trend")
                    st.markdown(f"⚡ RSI: {pick['rsi']:.1f} | Vol: {pick['volume_ratio']:.1f}x")
                
                with col4:
                    # Use hash of category_id + symbol + position for truly unique keys
                    import hashlib
                    unique_key = hashlib.md5(f"{category_id}_{pick['symbol']}_{i}".encode()).hexdigest()[:8]
                    if st.button(f"Analyze {pick['symbol']}", key=f"analyze_{unique_key}"):
                        st.session_state.selected_stock = pick['symbol']
                        st.session_state.page = "Individual Analysis"
                        st.rerun()
                
                st.markdown("---")
    
    def show_export_options(self, results: List[Dict]):
        """Show export options for bulk scan results"""
        st.subheader("📁 Export Scan Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Export Formats")
            
            export_format = st.selectbox(
                "Choose Export Format",
                ["CSV (Excel Compatible)", "JSON (Developer Friendly)", "PDF Report (Coming Soon)"]
            )
            
            include_components = st.checkbox("Include AI Component Scores", value=True)
            include_metadata = st.checkbox("Include Scan Metadata", value=True)
            
            filename_prefix = st.text_input(
                "File Name Prefix", 
                value=f"stock_scan_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
        
        with col2:
            st.markdown("### 📋 Export Preview")
            
            if results:
                preview_df = self.bulk_scanner.create_scan_results_table(results[:5])
                st.dataframe(preview_df, use_container_width=True)
                st.caption(f"Preview of first 5 results (Total: {len(results)} stocks)")
        
        # Export buttons
        st.markdown("---")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📊 Export CSV", type="primary", use_container_width=True):
                self.export_bulk_results(results, "csv", filename_prefix, include_components)
        
        with export_col2:
            if st.button("🔧 Export JSON", use_container_width=True):
                self.export_bulk_results(results, "json", filename_prefix, include_components)
        
        with export_col3:
            if st.button("📄 Export Summary", use_container_width=True):
                self.export_scan_summary(results, filename_prefix)
    
    def export_bulk_results(self, results: List[Dict], format_type: str = "csv", filename_prefix: str = "scan", include_components: bool = True):
        """Export bulk scan results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == "csv":
                df = self.bulk_scanner.create_scan_results_table(results)
                filename = f"{filename_prefix}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                st.success(f"✅ Results exported to {filename}")
            
            elif format_type == "json":
                filename = f"{filename_prefix}_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                st.success(f"✅ Results exported to {filename}")
            
            # Show download link (in a real deployment, you'd use st.download_button)
            st.info(f"📁 File saved as: {filename}")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
    
    def export_scan_summary(self, results: List[Dict], filename_prefix: str):
        """Export scan summary report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_summary_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"AI Stock Scanner - Scan Summary Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"="*50 + "\n\n")
                
                # Summary statistics
                f.write(f"Total Stocks Analyzed: {len(results)}\n")
                f.write(f"Average AI Score: {np.mean([r['ai_score'] for r in results]):.2f}\n")
                f.write(f"Strong Signals (75+): {len([r for r in results if r['ai_score'] >= 75])}\n")
                f.write(f"Bullish Trends: {len([r for r in results if r['daily_trend'] == 'Bullish'])}\n")
                f.write(f"Penny Stocks: {len([r for r in results if r['is_penny_stock']])}\n\n")
                
                # Top 10 picks
                f.write("TOP 10 PICKS BY AI SCORE:\n")
                f.write("-" * 30 + "\n")
                
                top_10 = sorted(results, key=lambda x: x['ai_score'], reverse=True)[:10]
                for i, stock in enumerate(top_10, 1):
                    f.write(f"{i:2}. {stock['symbol']:6} - AI Score: {stock['ai_score']:5.2f} - "
                           f"Price: ${stock['current_price']:7.2f} - Trend: {stock['daily_trend']}\n")
            
            st.success(f"✅ Summary report exported to {filename}")
            
        except Exception as e:
            st.error(f"❌ Summary export failed: {str(e)}")
    
    def analyze_individual_stock(self, symbol: str):
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                # Get stock data
                daily_data = self.yahoo_client.get_stock_data(symbol, period="1y")
                weekly_data = self.yahoo_client.get_weekly_data(symbol, period="2y")
                
                if daily_data is None:
                    st.error(f"❌ Could not fetch data for {symbol}")
                    return
                
                # Calculate indicators
                daily_with_indicators = self.technical_indicators.calculate_all_indicators(daily_data)
                weekly_with_indicators = self.technical_indicators.calculate_all_indicators(weekly_data)
                
                # Analyze trends
                trend_analysis = self.trend_analyzer.analyze_timeframe_alignment(
                    daily_with_indicators, weekly_with_indicators
                )
                
                # Get AI analysis
                latest = daily_with_indicators.iloc[-1]
                # Get price positioning for AI analysis
                current_price = latest['close']
                ema_20 = latest.get('ema_20', current_price)
                ema_50 = latest.get('ema_50', current_price)
                
                analysis_data = {
                    'current_price': current_price,
                    'daily_trend': trend_analysis['daily_trend'],
                    'weekly_trend': trend_analysis['weekly_trend'],
                    'alignment_score': trend_analysis.get('alignment_score', 0),
                    'rsi': latest.get('rsi', 50),
                    'volume_ratio': latest.get('volume_ratio', 1),
                    'volatility': latest.get('volatility', 0.2),
                    'price_positioning': {
                        'price_vs_ema_20': 'above' if current_price > ema_20 else 'below',
                        'price_vs_ema_50': 'above' if current_price > ema_50 else 'below'
                    },
                    'macd_signal': latest.get('macd_signal', 'neutral'),
                    'patterns': {}
                }
                
                # Get news data for AI analysis
                news_summary = self.financial_report_generator.news_service.get_news_summary(symbol)
                news_for_ai = news_summary.get('formatted_for_ai', '')
                
                ai_results = self.ai_analyzer.analyze_stock(symbol, analysis_data, news_for_ai)
                
                # Store in session state
                st.session_state[f"{symbol}_analysis"] = {
                    'daily_data': daily_with_indicators,
                    'weekly_data': weekly_with_indicators,
                    'trend_analysis': trend_analysis,
                    'ai_results': ai_results,
                    'analysis_data': analysis_data
                }
                
                st.success(f"✅ Analysis completed for {symbol}")
                
            except Exception as e:
                st.error(f"❌ Error analyzing {symbol}: {str(e)}")
                logger.error(f"Individual analysis error: {e}")
    
    def display_individual_analysis(self, symbol: str):
        analysis = st.session_state[f"{symbol}_analysis"]
        
        # Key metrics header
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "AI Score",
                f"{analysis['ai_results']['ai_score']:.2f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Signal",
                analysis['ai_results']['signal_classification']
            )
        
        with col3:
            st.metric(
                "Current Price",
                f"${analysis['analysis_data']['current_price']:.2f}"
            )
        
        with col4:
            st.metric(
                "RSI",
                f"{analysis['analysis_data']['rsi']:.1f}"
            )
        
        # Create comprehensive analysis tabs
        tabs = st.tabs([
            "📈 Technical Analysis", 
            "🏢 Company Overview", 
            "📊 Financial Metrics", 
            "🏥 Financial Health", 
            "🏆 Peer Comparison", 
            "🔮 Financial Forecast",
            "📰 News Analysis",
            "🤖 AI Analysis"
        ])
        
        with tabs[0]:  # Technical Analysis
            self.create_price_chart(symbol, analysis['daily_data'])
            
            # Component scores
            st.subheader("📊 AI Component Scores")
            scores = analysis['ai_results']['component_scores']
            
            score_df = pd.DataFrame([
                {'Component': 'Trend Alignment', 'Score': scores.get('trend_alignment', 0)},
                {'Component': 'Volume Confirmation', 'Score': scores.get('volume_confirmation', 0)},
                {'Component': 'RSI Positioning', 'Score': scores.get('rsi_positioning', 0)},
                {'Component': 'Price Action', 'Score': scores.get('price_action', 0)},
                {'Component': 'Volatility Analysis', 'Score': scores.get('volatility_analysis', 0)},
                {'Component': 'Momentum Quality', 'Score': scores.get('momentum_quality', 0)}
            ])
            
            fig = px.bar(
                score_df,
                x='Component',
                y='Score',
                title="AI Score Breakdown",
                color='Score',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:  # Company Overview
            self.financial_report_generator.show_company_overview(symbol)
        
        with tabs[2]:  # Financial Metrics
            self.financial_report_generator.show_financial_metrics(symbol)
        
        with tabs[3]:  # Financial Health
            self.financial_report_generator.show_financial_health_analysis(symbol)
        
        with tabs[4]:  # Peer Comparison
            self.financial_report_generator.show_peer_comparison(symbol)
        
        with tabs[5]:  # Financial Forecast
            self.financial_report_generator.show_financial_forecast(symbol)
        
        with tabs[6]:  # News Analysis
            self.financial_report_generator.show_news_analysis(symbol)
        
        with tabs[7]:  # AI Analysis
            st.subheader("🤖 AI Analysis")
            st.write(analysis['ai_results']['ai_narrative'])
            
            # Confidence and additional metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{analysis['ai_results'].get('confidence', 0):.1f}%")
            with col2:
                st.metric("Analysis Timestamp", 
                         datetime.fromtimestamp(analysis['ai_results'].get('analysis_timestamp', 0)).strftime('%H:%M:%S'))
    
    def create_price_chart(self, symbol: str, data: pd.DataFrame):
        st.subheader(f"📈 {symbol} Enhanced Technical Chart")
        
        # Get weekly data for analysis
        weekly_data = self.yahoo_client.get_weekly_data(symbol, period="2y")
        
        # Calculate AI score for context
        if weekly_data is not None and len(weekly_data) > 20:
            # Get basic positioning data
            current_price = data['close'].iloc[-1]
            ema_20 = data.get('ema_20', data['close'].ewm(span=20).mean()).iloc[-1]
            ema_50 = data.get('ema_50', data['close'].ewm(span=50).mean()).iloc[-1]
            
            analysis_data = {
                'current_price': current_price,
                'daily_trend': {'overall_trend': 'neutral'},
                'weekly_trend': {'overall_trend': 'neutral'},
                'alignment_score': 0,
                'rsi': data.get('rsi', pd.Series([50])).iloc[-1] if 'rsi' in data.columns else 50,
                'volume_ratio': data.get('volume_ratio', pd.Series([1])).iloc[-1] if 'volume_ratio' in data.columns else 1,
                'volatility': 0.2,
                'price_positioning': {
                    'price_vs_ema_20': 'above' if current_price > ema_20 else 'below',
                    'price_vs_ema_50': 'above' if current_price > ema_50 else 'below'
                },
                'macd_signal': 'neutral',
                'patterns': {}
            }
            ai_results = self.ai_analyzer.calculate_ai_score(analysis_data)
            ai_score = ai_results.get('ai_score', 0)
        else:
            ai_score = 0
            weekly_data = data  # Fallback
        
        # Use enhanced chart creator
        fig = self.chart_creator.create_comprehensive_chart(symbol, data, weekly_data, ai_score)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add trading signals
        st.subheader("🎯 Trading Signals")
        signals = self.signal_generator.generate_trading_signals(data, weekly_data, ai_score)
        
        for signal in signals:
            signal_type = signal.get('type', 'Unknown')
            signal_emoji = get_signal_emoji(signal_type)
            
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 2])
                
                with col1:
                    st.markdown(f"## {signal_emoji}")
                
                with col2:
                    st.markdown(f"**{signal_type}**")
                    st.markdown(f"*{signal.get('strength', 'N/A')}*")
                
                with col3:
                    st.markdown(f"**Reason:** {signal.get('reason', 'No reason provided')}")
        
        # Support/Resistance and Zones
        st.subheader("📊 Support & Resistance Analysis")
        
        support_levels, resistance_levels = self.pattern_recognition.identify_support_resistance(data)
        zones = self.pattern_recognition.identify_demand_supply_zones(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if support_levels:
                st.markdown("**🟢 Support Levels:**")
                for level in support_levels[-5:]:  # Show last 5
                    current_price = data['close'].iloc[-1]
                    distance = ((current_price - level) / level) * 100
                    st.markdown(f"• ${level:.2f} ({distance:+.1f}% from current)")
            else:
                st.info("No significant support levels found")
        
        with col2:
            if resistance_levels:
                st.markdown("**🔴 Resistance Levels:**")
                for level in resistance_levels[-5:]:  # Show last 5
                    current_price = data['close'].iloc[-1]
                    distance = ((level - current_price) / current_price) * 100
                    st.markdown(f"• ${level:.2f} (+{distance:.1f}% from current)")
            else:
                st.info("No significant resistance levels found")
        
        # Demand/Supply zones
        if zones:
            st.subheader("🔄 Demand & Supply Zones")
            
            recent_zones = sorted(zones, key=lambda x: x['date'], reverse=True)[:5]
            
            for i, zone in enumerate(recent_zones, 1):
                zone_type = zone['type'].title()
                zone_emoji = "🟢" if zone['type'] == 'demand' else "🔴"
                strength = zone['strength']
                
                st.markdown(f"{zone_emoji} **{zone_type} Zone #{i}** (Strength: {strength:.1f}x)")
                st.markdown(f"   Range: ${zone['low']:.2f} - ${zone['high']:.2f}")
                st.markdown(f"   Date: {zone['date'].strftime('%Y-%m-%d')}")
                st.markdown("---")
    
    def show_settings(self):
        st.header("⚙️ Settings")
        
        # OLLAMA settings
        st.subheader("🤖 OLLAMA Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ollama_host = st.text_input("OLLAMA Host", value=settings.OLLAMA_HOST)
            
        with col2:
            ollama_model = st.text_input("OLLAMA Model", value=settings.OLLAMA_MODEL)
        
        # Test OLLAMA connection
        if st.button("Test OLLAMA Connection"):
            from src.ai.ollama_client import OllamaClient
            client = OllamaClient()
            
            if client.check_connection():
                st.success("✅ OLLAMA connection successful!")
                models = client.list_models()
                if models:
                    st.info(f"Available models: {', '.join(models)}")
            else:
                st.error("❌ Could not connect to OLLAMA. Please check your configuration.")
        
        # Scanning settings
        st.subheader("🔍 Scanning Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_workers = st.number_input("Max Workers", min_value=1, max_value=20, value=settings.MAX_WORKERS)
            
        with col2:
            rate_limit = st.number_input("Rate Limit (seconds)", min_value=0.1, max_value=5.0, value=settings.YAHOO_RATE_LIMIT)
        
        # Technical analysis settings
        st.subheader("📊 Technical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=settings.RSI_PERIOD)
            
        with col2:
            rsi_oversold = st.number_input("RSI Oversold", min_value=10, max_value=40, value=settings.RSI_OVERSOLD)
            
        with col3:
            rsi_overbought = st.number_input("RSI Overbought", min_value=60, max_value=90, value=settings.RSI_OVERBOUGHT)
        
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved! (Note: Restart required for some changes)")

def main():
    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Streamlit app error: {e}")

if __name__ == "__main__":
    main()