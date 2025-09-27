import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List
import logging

from src.analysis.financial_analyzer import FinancialAnalyzer, FinancialMetrics
from src.data.news_service import NewsService
from src.ai.ollama_client import AIAnalyzer
from src.ai.news_analyzer import AdvancedNewsAnalyzer, TradingSignal, MarketImpact

logger = logging.getLogger(__name__)

class FinancialReportGenerator:
    """Generate comprehensive financial reports and visualizations"""
    
    def __init__(self):
        self.financial_analyzer = FinancialAnalyzer()
        self.news_service = NewsService()
        self.ai_analyzer = AIAnalyzer()
        self.advanced_news_analyzer = AdvancedNewsAnalyzer()
    
    def show_company_overview(self, symbol: str) -> None:
        """Display company overview section"""
        try:
            company_info = self.financial_analyzer.get_company_info(symbol)
            
            if not company_info:
                st.error("Unable to fetch company information")
                return
            
            st.subheader("🏢 Company Overview")
            
            # Create columns for company info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Company Name:** {company_info.get('company_name', 'N/A')}
                
                **Sector:** {company_info.get('sector', 'N/A')}
                
                **Industry:** {company_info.get('industry', 'N/A')}
                
                **Country:** {company_info.get('country', 'N/A')}
                
                **Exchange:** {company_info.get('exchange', 'N/A')}
                """)
            
            with col2:
                market_cap = company_info.get('market_cap', 0)
                if market_cap > 0:
                    market_cap_formatted = self._format_large_number(market_cap)
                    st.metric("Market Cap", market_cap_formatted)
                
                enterprise_value = company_info.get('enterprise_value', 0)
                if enterprise_value > 0:
                    ev_formatted = self._format_large_number(enterprise_value)
                    st.metric("Enterprise Value", ev_formatted)
                
                employees = company_info.get('employees', 0)
                if employees > 0:
                    st.metric("Employees", f"{employees:,}")
            
            # Company description
            description = company_info.get('description', '')
            if description:
                st.markdown("**Business Description:**")
                st.write(description[:500] + "..." if len(description) > 500 else description)
            
            if company_info.get('website'):
                st.markdown(f"**Website:** [{company_info['website']}]({company_info['website']})")
                
        except Exception as e:
            logger.error(f"Error showing company overview: {e}")
            st.error("Error displaying company overview")
    
    def show_financial_metrics(self, symbol: str) -> None:
        """Display financial metrics dashboard"""
        try:
            st.subheader("📊 Financial Metrics Dashboard")
            
            metrics = self.financial_analyzer.calculate_financial_metrics(symbol)
            
            if not any(vars(metrics).values()):
                st.warning("Limited financial data available for this stock")
                return
            
            # Create tabs for different metric categories
            tabs = st.tabs(["📈 Profitability", "💧 Liquidity", "⚖️ Leverage", "📊 Valuation", "🚀 Growth"])
            
            with tabs[0]:  # Profitability
                self._show_profitability_metrics(metrics)
            
            with tabs[1]:  # Liquidity
                self._show_liquidity_metrics(metrics)
            
            with tabs[2]:  # Leverage
                self._show_leverage_metrics(metrics)
            
            with tabs[3]:  # Valuation
                self._show_valuation_metrics(metrics)
            
            with tabs[4]:  # Growth
                self._show_growth_metrics(metrics)
                
        except Exception as e:
            logger.error(f"Error showing financial metrics: {e}")
            st.error("Error displaying financial metrics")
    
    def show_financial_health_analysis(self, symbol: str) -> None:
        """Display financial health analysis"""
        try:
            st.subheader("🏥 Financial Health Analysis")
            
            metrics = self.financial_analyzer.calculate_financial_metrics(symbol)
            health_analysis = self.financial_analyzer.analyze_financial_health(metrics)
            
            if not health_analysis:
                st.warning("Unable to perform financial health analysis")
                return
            
            overall = health_analysis.get('overall', {})
            
            # Overall health score
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                score = overall.get('score', 0)
                max_score = overall.get('max_score', 100)
                percentage = overall.get('percentage', 0)
                rating = overall.get('rating', 'Unknown')
                
                # Create gauge chart for overall health
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = percentage,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Financial Health Score"},
                    delta = {'reference': 50},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': self._get_health_color(percentage)},
                             'steps': [
                                 {'range': [0, 20], 'color': "lightgray"},
                                 {'range': [20, 40], 'color': "lightblue"},
                                 {'range': [40, 60], 'color': "lightyellow"},
                                 {'range': [60, 80], 'color': "lightgreen"},
                                 {'range': [80, 100], 'color': "green"}],
                             'threshold': {'line': {'color': "red", 'width': 4},
                                          'thickness': 0.75, 'value': 90}}))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Overall Rating", rating)
                st.metric("Score", f"{score}/{max_score}")
            
            with col3:
                st.metric("Percentage", f"{percentage}%")
            
            # Category breakdown
            st.markdown("### Category Breakdown")
            
            categories = ['profitability', 'liquidity', 'leverage', 'valuation', 'growth']
            category_names = ['Profitability', 'Liquidity', 'Leverage', 'Valuation', 'Growth']
            
            cols = st.columns(len(categories))
            
            for i, (category, name) in enumerate(zip(categories, category_names)):
                with cols[i]:
                    cat_data = health_analysis.get(category, {})
                    cat_score = cat_data.get('score', 0)
                    cat_max = cat_data.get('max_score', 1)
                    cat_rating = cat_data.get('rating', 'Unknown')
                    cat_percentage = (cat_score / cat_max) * 100 if cat_max > 0 else 0
                    
                    st.metric(
                        name,
                        f"{cat_score}/{cat_max}",
                        delta=f"{cat_percentage:.1f}%"
                    )
                    st.caption(cat_rating)
            
            # Create radar chart for category comparison
            self._create_health_radar_chart(health_analysis)
                
        except Exception as e:
            logger.error(f"Error showing financial health analysis: {e}")
            st.error("Error displaying financial health analysis")
    
    def show_peer_comparison(self, symbol: str) -> None:
        """Display peer comparison analysis"""
        try:
            st.subheader("🏆 Peer Comparison")
            
            company_info = self.financial_analyzer.get_company_info(symbol)
            sector = company_info.get('sector', 'Unknown')
            
            peer_data = self.financial_analyzer.get_peer_comparison(symbol, sector)
            
            if not peer_data or not peer_data.get('peers'):
                st.warning(f"No peer data available for {sector} sector")
                return
            
            peers = peer_data['peers']
            sector_avg = peer_data.get('sector_average', {})
            
            # Create comparison table
            df_data = []
            
            # Add current company data
            current_metrics = self.financial_analyzer.calculate_financial_metrics(symbol)
            df_data.append({
                'Company': symbol + ' (Current)',
                'Market Cap': company_info.get('market_cap', 0),
                'P/E Ratio': current_metrics.pe_ratio,
                'Profit Margin': current_metrics.net_margin,
                'ROE': current_metrics.roe
            })
            
            # Add peer data
            for peer in peers:
                df_data.append({
                    'Company': peer['symbol'],
                    'Market Cap': peer['market_cap'],
                    'P/E Ratio': peer['pe_ratio'],
                    'Profit Margin': peer['profit_margin'],
                    'ROE': peer['roe']
                })
            
            df = pd.DataFrame(df_data)
            
            # Format the dataframe
            df['Market Cap'] = df['Market Cap'].apply(lambda x: self._format_large_number(x) if x else 'N/A')
            df['P/E Ratio'] = df['P/E Ratio'].apply(lambda x: f"{x:.2f}" if x else 'N/A')
            df['Profit Margin'] = df['Profit Margin'].apply(lambda x: f"{x*100:.1f}%" if x else 'N/A')
            df['ROE'] = df['ROE'].apply(lambda x: f"{x*100:.1f}%" if x else 'N/A')
            
            st.dataframe(df, use_container_width=True)
            
            # Sector averages
            if sector_avg:
                st.markdown("### Sector Averages")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'pe_ratio' in sector_avg:
                        st.metric("Avg P/E Ratio", f"{sector_avg['pe_ratio']:.2f}")
                
                with col2:
                    if 'profit_margin' in sector_avg:
                        st.metric("Avg Profit Margin", f"{sector_avg['profit_margin']*100:.1f}%")
                
                with col3:
                    if 'roe' in sector_avg:
                        st.metric("Avg ROE", f"{sector_avg['roe']*100:.1f}%")
                
        except Exception as e:
            logger.error(f"Error showing peer comparison: {e}")
            st.error("Error displaying peer comparison")
    
    def show_financial_forecast(self, symbol: str) -> None:
        """Display financial forecasts"""
        try:
            st.subheader("🔮 Financial Forecast")
            
            forecast = self.financial_analyzer.generate_forecast(symbol)
            
            if not forecast:
                st.warning("Unable to generate financial forecast")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'revenue_projection' in forecast:
                    st.markdown("### Revenue Projection")
                    rev_proj = forecast['revenue_projection']
                    
                    years = ['Current', 'Year 1', 'Year 2', 'Year 3']
                    revenues = [
                        rev_proj['year_1'] / (1 + rev_proj['growth_rate']),  # Back-calculate current
                        rev_proj['year_1'],
                        rev_proj['year_2'],
                        rev_proj['year_3']
                    ]
                    
                    fig = px.bar(
                        x=years,
                        y=revenues,
                        title=f"Revenue Growth ({rev_proj['growth_rate']*100:.1f}% annually)",
                        labels={'x': 'Period', 'y': 'Revenue ($)'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Growth Rate", f"{rev_proj['growth_rate']*100:.1f}% annually")
            
            with col2:
                if 'eps_projection' in forecast:
                    st.markdown("### EPS Projection")
                    eps_proj = forecast['eps_projection']
                    
                    years = ['Current', 'Year 1', 'Year 2', 'Year 3']
                    eps_values = [
                        eps_proj['year_1'] / (1 + eps_proj['growth_rate']),  # Back-calculate current
                        eps_proj['year_1'],
                        eps_proj['year_2'],
                        eps_proj['year_3']
                    ]
                    
                    fig = px.line(
                        x=years,
                        y=eps_values,
                        title=f"EPS Growth ({eps_proj['growth_rate']*100:.1f}% annually)",
                        labels={'x': 'Period', 'y': 'EPS ($)'},
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Growth Rate", f"{eps_proj['growth_rate']*100:.1f}% annually")
            
            # Price targets
            if 'price_target' in forecast:
                st.markdown("### Price Targets")
                price_targets = forecast['price_target']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Conservative", f"${price_targets['conservative']:.2f}")
                
                with col2:
                    st.metric("Base Case", f"${price_targets['base_case']:.2f}")
                
                with col3:
                    st.metric("Optimistic", f"${price_targets['optimistic']:.2f}")
                
        except Exception as e:
            logger.error(f"Error showing financial forecast: {e}")
            st.error("Error displaying financial forecast")
    
    def _show_profitability_metrics(self, metrics: FinancialMetrics) -> None:
        """Display profitability metrics"""
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics.roe is not None:
                st.metric("Return on Equity (ROE)", f"{metrics.roe*100:.2f}%")
            if metrics.roa is not None:
                st.metric("Return on Assets (ROA)", f"{metrics.roa*100:.2f}%")
            if metrics.gross_margin is not None:
                st.metric("Gross Margin", f"{metrics.gross_margin*100:.2f}%")
        
        with col2:
            if metrics.net_margin is not None:
                st.metric("Net Margin", f"{metrics.net_margin*100:.2f}%")
            if metrics.operating_margin is not None:
                st.metric("Operating Margin", f"{metrics.operating_margin*100:.2f}%")
    
    def _show_liquidity_metrics(self, metrics: FinancialMetrics) -> None:
        """Display liquidity metrics"""
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics.current_ratio is not None:
                st.metric("Current Ratio", f"{metrics.current_ratio:.2f}")
            if metrics.quick_ratio is not None:
                st.metric("Quick Ratio", f"{metrics.quick_ratio:.2f}")
        
        with col2:
            if metrics.cash_ratio is not None:
                st.metric("Cash Ratio", f"{metrics.cash_ratio:.2f}")
    
    def _show_leverage_metrics(self, metrics: FinancialMetrics) -> None:
        """Display leverage metrics"""
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics.debt_to_equity is not None:
                st.metric("Debt-to-Equity", f"{metrics.debt_to_equity:.2f}")
            if metrics.debt_to_assets is not None:
                st.metric("Debt-to-Assets", f"{metrics.debt_to_assets:.2f}")
        
        with col2:
            if metrics.interest_coverage is not None:
                st.metric("Interest Coverage", f"{metrics.interest_coverage:.2f}x")
    
    def _show_valuation_metrics(self, metrics: FinancialMetrics) -> None:
        """Display valuation metrics"""
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics.pe_ratio is not None:
                st.metric("P/E Ratio", f"{metrics.pe_ratio:.2f}")
            if metrics.peg_ratio is not None:
                st.metric("PEG Ratio", f"{metrics.peg_ratio:.2f}")
            if metrics.price_to_book is not None:
                st.metric("Price-to-Book", f"{metrics.price_to_book:.2f}")
        
        with col2:
            if metrics.price_to_sales is not None:
                st.metric("Price-to-Sales", f"{metrics.price_to_sales:.2f}")
            if metrics.ev_ebitda is not None:
                st.metric("EV/EBITDA", f"{metrics.ev_ebitda:.2f}")
    
    def _show_growth_metrics(self, metrics: FinancialMetrics) -> None:
        """Display growth metrics"""
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics.revenue_growth is not None:
                st.metric("Revenue Growth", f"{metrics.revenue_growth*100:.2f}%")
            if metrics.earnings_growth is not None:
                st.metric("Earnings Growth", f"{metrics.earnings_growth*100:.2f}%")
        
        with col2:
            if metrics.book_value_growth is not None:
                st.metric("Book Value Growth", f"{metrics.book_value_growth*100:.2f}%")
    
    def _create_health_radar_chart(self, health_analysis: Dict[str, Any]) -> None:
        """Create radar chart for financial health categories"""
        try:
            categories = ['Profitability', 'Liquidity', 'Leverage', 'Valuation', 'Growth']
            values = []
            
            for category in ['profitability', 'liquidity', 'leverage', 'valuation', 'growth']:
                cat_data = health_analysis.get(category, {})
                score = cat_data.get('score', 0)
                max_score = cat_data.get('max_score', 1)
                percentage = (score / max_score) * 100 if max_score > 0 else 0
                values.append(percentage)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Financial Health'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Financial Health Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
    
    def _format_large_number(self, number: float) -> str:
        """Format large numbers with appropriate suffixes"""
        if number >= 1e12:
            return f"${number/1e12:.2f}T"
        elif number >= 1e9:
            return f"${number/1e9:.2f}B"
        elif number >= 1e6:
            return f"${number/1e6:.2f}M"
        elif number >= 1e3:
            return f"${number/1e3:.2f}K"
        else:
            return f"${number:.2f}"
    
    def _get_health_color(self, percentage: float) -> str:
        """Get color based on health percentage"""
        if percentage >= 80:
            return "green"
        elif percentage >= 60:
            return "lightgreen"
        elif percentage >= 40:
            return "yellow"
        elif percentage >= 20:
            return "orange"
        else:
            return "red"
    
    def show_news_analysis(self, symbol: str) -> None:
        """Display comprehensive news analysis with AI-powered trading recommendations"""
        try:
            st.subheader(f"📰 Advanced News Analysis & Trading Recommendations for {symbol}")
            
            # Get comprehensive news analysis
            with st.spinner("Performing comprehensive news analysis..."):
                analysis_result = self.advanced_news_analyzer.analyze_stock_news_comprehensive(symbol)
            
            if analysis_result.articles_analyzed == 0:
                st.warning(f"AI analysis found no articles to analyze for {symbol}, but showing available news below")
            
            # Trading Recommendation Dashboard
            st.markdown("### 🎯 AI Trading Recommendation")
            
            # Create recommendation header with color coding
            signal_colors = {
                TradingSignal.STRONG_BUY: "🟢",
                TradingSignal.BUY: "🟢", 
                TradingSignal.HOLD: "🟡",
                TradingSignal.SELL: "🔴",
                TradingSignal.STRONG_SELL: "🔴"
            }
            
            signal_emoji = signal_colors.get(analysis_result.trading_signal, "⚪")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Trading Signal", 
                    f"{signal_emoji} {analysis_result.trading_signal.value.replace('_', ' ').title()}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Confidence", 
                    f"{analysis_result.confidence:.0%}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Market Impact", 
                    analysis_result.market_impact.value.title(),
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Articles Analyzed", 
                    analysis_result.articles_analyzed,
                    delta=None
                )
            
            # Action Recommendation
            st.markdown("#### 📋 Recommended Action")
            st.info(analysis_result.target_action)
            
            # Detailed Analysis Tabs
            analysis_tabs = st.tabs([
                "📊 Summary", "🧠 AI Analysis", "📈 Sentiment", "⚡ Key Factors", "📰 News Articles"
            ])
            
            with analysis_tabs[0]:  # Summary
                st.markdown("#### 📊 Analysis Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Overall Assessment:**")
                    st.write(analysis_result.recommendation_reasoning)
                    
                    st.markdown("**Key Themes:**")
                    if analysis_result.key_themes:
                        for theme in analysis_result.key_themes:
                            st.write(f"• {theme.replace('_', ' ').title()}")
                    else:
                        st.write("No specific themes identified")
                
                with col2:
                    # Sentiment gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = analysis_result.sentiment_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "News Sentiment Score"},
                        gauge = {
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "blue"},
                            'steps': [
                                {'range': [-1, -0.2], 'color': "red"},
                                {'range': [-0.2, 0.2], 'color': "yellow"},
                                {'range': [0.2, 1], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': analysis_result.sentiment_score
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250, font={'size': 10})
                    st.plotly_chart(fig, use_container_width=True)
            
            with analysis_tabs[1]:  # AI Analysis
                st.markdown("#### 🧠 Comprehensive AI Analysis")
                
                if analysis_result.ai_summary and "unavailable" not in analysis_result.ai_summary.lower():
                    st.write(analysis_result.ai_summary)
                else:
                    st.warning("AI analysis unavailable - OLLAMA not connected or analysis failed")
                
                st.markdown("#### 🎯 Investment Reasoning")
                st.info(analysis_result.recommendation_reasoning)
            
            with analysis_tabs[2]:  # Sentiment
                st.markdown("#### 📈 Sentiment Analysis")
                
                sentiment_col1, sentiment_col2 = st.columns(2)
                
                with sentiment_col1:
                    st.metric("Overall Sentiment", analysis_result.overall_sentiment.title())
                    st.metric("Sentiment Score", f"{analysis_result.sentiment_score:.3f}")
                    
                    # Create sentiment breakdown chart
                    if hasattr(analysis_result, 'sentiment_breakdown'):
                        sentiment_data = analysis_result.sentiment_breakdown
                        fig = px.pie(
                            values=list(sentiment_data.values()),
                            names=list(sentiment_data.keys()),
                            title="Sentiment Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with sentiment_col2:
                    st.markdown("**Sentiment Interpretation:**")
                    if analysis_result.sentiment_score > 0.4:
                        st.success("🟢 Strong positive sentiment - Bullish news flow")
                    elif analysis_result.sentiment_score > 0.1:
                        st.info("🟡 Moderately positive sentiment - Cautiously optimistic")
                    elif analysis_result.sentiment_score > -0.1:
                        st.warning("🟡 Neutral sentiment - Mixed signals")
                    elif analysis_result.sentiment_score > -0.4:
                        st.warning("🟠 Moderately negative sentiment - Cautious approach")
                    else:
                        st.error("🔴 Strong negative sentiment - Bearish news flow")
            
            with analysis_tabs[3]:  # Key Factors
                st.markdown("#### ⚡ Key Factors Analysis")
                
                factor_col1, factor_col2 = st.columns(2)
                
                with factor_col1:
                    st.markdown("**🚨 Risk Factors:**")
                    if analysis_result.risk_factors:
                        for risk in analysis_result.risk_factors:
                            st.warning(f"⚠️ {risk}")
                    else:
                        st.success("✅ No significant risks identified")
                
                with factor_col2:
                    st.markdown("**🚀 Potential Catalysts:**")
                    if analysis_result.catalysts:
                        for catalyst in analysis_result.catalysts:
                            st.success(f"📈 {catalyst}")
                    else:
                        st.info("ℹ️ No specific catalysts identified")
                
                # Market Impact Assessment
                st.markdown("**📊 Market Impact Assessment:**")
                impact_color = {
                    MarketImpact.HIGH: "🔴",
                    MarketImpact.MEDIUM: "🟡", 
                    MarketImpact.LOW: "🟢",
                    MarketImpact.NEUTRAL: "⚪"
                }
                
                impact_description = {
                    MarketImpact.HIGH: "Significant market-moving events detected. Expect high volatility.",
                    MarketImpact.MEDIUM: "Moderate impact events. Watch for price movement.",
                    MarketImpact.LOW: "Limited impact expected. Minor price influence.",
                    MarketImpact.NEUTRAL: "Minimal market impact anticipated."
                }
                
                st.info(f"{impact_color[analysis_result.market_impact]} {impact_description[analysis_result.market_impact]}")
            
            with analysis_tabs[4]:  # News Articles
                st.markdown("#### 📰 Source Articles")
                
                # Get articles from news service
                articles = self.news_service.get_stock_news(symbol, days_back=7, max_articles=10)
                
                # News source diversity
                st.markdown("### 📊 News Source Diversity")
                if articles:
                    source_counts = {}
                    for article in articles:
                        source = article.publisher.split(' - ')[0] if ' - ' in article.publisher else article.publisher
                        source_counts[source] = source_counts.get(source, 0) + 1
                    
                    source_cols = st.columns(min(len(source_counts), 5))
                    for i, (source, count) in enumerate(source_counts.items()):
                        if i < len(source_cols):
                            with source_cols[i]:
                                st.metric(source, f"{count} articles")
                
                # Recent news articles
                st.markdown("### 📋 Recent News Articles")
                
                if not articles:
                    st.info(f"No articles found directly from news service for {symbol}. This may indicate a data fetching issue.")
                    st.write("Debug: Articles list is empty")
                else:
                    st.success(f"Found {len(articles)} articles to display")
                
                for i, article in enumerate(articles, 1):
                    try:
                        # Color code by source
                        source_colors = {
                            'Yahoo Finance': '🟠',
                            'Google News': '🔵', 
                            'Finviz': '🟢',
                            'MarketWatch': '🟡',
                            'Investing.com': '🟣'
                        }
                        
                        source_key = article.publisher.split(' - ')[0] if ' - ' in article.publisher else article.publisher
                        source_emoji = source_colors.get(source_key, '⚪')
                        
                        with st.expander(f"{source_emoji} {i}. {article.title} ({article.published_date.strftime('%Y-%m-%d %H:%M')})"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Publisher:** {article.publisher}")
                                st.markdown(f"**Published:** {article.published_date.strftime('%Y-%m-%d %H:%M UTC')}")
                                st.markdown(f"**Summary:** {article.summary}")
                                
                                if article.link:
                                    st.markdown(f"[Read Full Article]({article.link})")
                            
                            with col2:
                                if article.sentiment:
                                    sentiment_emoji = {
                                        'positive': '🟢 Positive',
                                        'negative': '🔴 Negative',
                                        'neutral': '🟡 Neutral'
                                    }
                                    st.metric("Sentiment", sentiment_emoji.get(article.sentiment, '🟡 Neutral'))
                    except Exception as e:
                        st.error(f"Error displaying article {i}: {e}")
                        logger.error(f"Error displaying article {i}: {e}")
                        continue
                
                # Market context
                st.markdown("### 📊 Market Context")
                market_news = self.news_service.get_market_news(max_articles=3)
                
                if market_news:
                    st.markdown("**Recent Market Headlines:**")
                    for i, article in enumerate(market_news, 1):
                        st.markdown(f"{i}. **{article.title}** - {article.publisher} ({article.published_date.strftime('%m/%d')})")
                else:
                    st.info("No recent market news available")
                
        except Exception as e:
            logger.error(f"Error showing news analysis: {e}")
            st.error("Error displaying news analysis")