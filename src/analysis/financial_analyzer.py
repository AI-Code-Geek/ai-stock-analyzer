import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FinancialMetrics:
    # Profitability Ratios
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    
    # Liquidity Ratios
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    
    # Leverage Ratios
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    # Efficiency Ratios
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    
    # Valuation Metrics
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_ebitda: Optional[float] = None
    
    # Growth Metrics
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    book_value_growth: Optional[float] = None

class FinancialAnalyzer:
    """Comprehensive financial analysis engine"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'description': info.get('longBusinessSummary', 'No description available'),
                'website': info.get('website', ''),
                'country': info.get('country', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {e}")
            return {}
    
    def calculate_financial_metrics(self, symbol: str) -> FinancialMetrics:
        """Calculate comprehensive financial metrics"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            metrics = FinancialMetrics()
            
            # Profitability Ratios
            metrics.roe = self._safe_divide(info.get('returnOnEquity'), 100) if info.get('returnOnEquity') else None
            metrics.roa = self._safe_divide(info.get('returnOnAssets'), 100) if info.get('returnOnAssets') else None
            metrics.gross_margin = self._safe_divide(info.get('grossMargins'), 1) if info.get('grossMargins') else None
            metrics.net_margin = self._safe_divide(info.get('profitMargins'), 1) if info.get('profitMargins') else None
            metrics.operating_margin = self._safe_divide(info.get('operatingMargins'), 1) if info.get('operatingMargins') else None
            
            # Liquidity Ratios
            metrics.current_ratio = info.get('currentRatio')
            metrics.quick_ratio = info.get('quickRatio')
            
            # Calculate cash ratio if balance sheet data available
            if not balance_sheet.empty:
                try:
                    latest_bs = balance_sheet.iloc[:, 0]
                    cash = latest_bs.get('Cash And Cash Equivalents', 0)
                    current_liabilities = latest_bs.get('Current Liabilities', 1)
                    metrics.cash_ratio = self._safe_divide(cash, current_liabilities)
                except:
                    pass
            
            # Leverage Ratios
            metrics.debt_to_equity = info.get('debtToEquity')
            metrics.interest_coverage = info.get('interestCoverage')
            
            # Valuation Metrics
            metrics.pe_ratio = info.get('trailingPE')
            metrics.peg_ratio = info.get('pegRatio')
            metrics.price_to_book = info.get('priceToBook')
            metrics.price_to_sales = info.get('priceToSalesTrailing12Months')
            metrics.ev_ebitda = info.get('enterpriseToEbitda')
            
            # Growth Metrics
            metrics.revenue_growth = info.get('revenueGrowth')
            metrics.earnings_growth = info.get('earningsGrowth')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating financial metrics for {symbol}: {e}")
            return FinancialMetrics()
    
    def analyze_financial_health(self, metrics: FinancialMetrics) -> Dict[str, Any]:
        """Analyze overall financial health based on metrics"""
        try:
            health_score = 0
            max_score = 0
            analysis = {}
            
            # Profitability Analysis
            profitability_score = 0
            profitability_max = 25
            
            if metrics.roe and metrics.roe > 0.15:  # ROE > 15%
                profitability_score += 5
            if metrics.roa and metrics.roa > 0.05:  # ROA > 5%
                profitability_score += 5
            if metrics.gross_margin and metrics.gross_margin > 0.3:  # Gross margin > 30%
                profitability_score += 5
            if metrics.net_margin and metrics.net_margin > 0.1:  # Net margin > 10%
                profitability_score += 5
            if metrics.operating_margin and metrics.operating_margin > 0.15:  # Operating margin > 15%
                profitability_score += 5
            
            analysis['profitability'] = {
                'score': profitability_score,
                'max_score': profitability_max,
                'rating': self._get_rating(profitability_score, profitability_max)
            }
            
            # Liquidity Analysis
            liquidity_score = 0
            liquidity_max = 15
            
            if metrics.current_ratio and metrics.current_ratio > 1.5:
                liquidity_score += 5
            if metrics.quick_ratio and metrics.quick_ratio > 1.0:
                liquidity_score += 5
            if metrics.cash_ratio and metrics.cash_ratio > 0.2:
                liquidity_score += 5
            
            analysis['liquidity'] = {
                'score': liquidity_score,
                'max_score': liquidity_max,
                'rating': self._get_rating(liquidity_score, liquidity_max)
            }
            
            # Leverage Analysis
            leverage_score = 0
            leverage_max = 15
            
            if metrics.debt_to_equity and metrics.debt_to_equity < 0.5:
                leverage_score += 5
            elif metrics.debt_to_equity and metrics.debt_to_equity < 1.0:
                leverage_score += 3
            
            if metrics.interest_coverage and metrics.interest_coverage > 5:
                leverage_score += 5
            elif metrics.interest_coverage and metrics.interest_coverage > 2:
                leverage_score += 3
            
            if metrics.debt_to_assets and metrics.debt_to_assets < 0.3:
                leverage_score += 5
            
            analysis['leverage'] = {
                'score': leverage_score,
                'max_score': leverage_max,
                'rating': self._get_rating(leverage_score, leverage_max)
            }
            
            # Valuation Analysis
            valuation_score = 0
            valuation_max = 20
            
            if metrics.pe_ratio and 10 <= metrics.pe_ratio <= 25:
                valuation_score += 5
            if metrics.peg_ratio and metrics.peg_ratio < 1.5:
                valuation_score += 5
            if metrics.price_to_book and metrics.price_to_book < 3:
                valuation_score += 5
            if metrics.ev_ebitda and metrics.ev_ebitda < 15:
                valuation_score += 5
            
            analysis['valuation'] = {
                'score': valuation_score,
                'max_score': valuation_max,
                'rating': self._get_rating(valuation_score, valuation_max)
            }
            
            # Growth Analysis
            growth_score = 0
            growth_max = 25
            
            if metrics.revenue_growth and metrics.revenue_growth > 0.1:
                growth_score += 8
            elif metrics.revenue_growth and metrics.revenue_growth > 0.05:
                growth_score += 5
            
            if metrics.earnings_growth and metrics.earnings_growth > 0.15:
                growth_score += 8
            elif metrics.earnings_growth and metrics.earnings_growth > 0.05:
                growth_score += 5
            
            if metrics.book_value_growth and metrics.book_value_growth > 0.1:
                growth_score += 9
            
            analysis['growth'] = {
                'score': growth_score,
                'max_score': growth_max,
                'rating': self._get_rating(growth_score, growth_max)
            }
            
            # Overall Health Score
            total_score = (profitability_score + liquidity_score + 
                          leverage_score + valuation_score + growth_score)
            total_max = (profitability_max + liquidity_max + 
                        leverage_max + valuation_max + growth_max)
            
            overall_rating = self._get_rating(total_score, total_max)
            
            analysis['overall'] = {
                'score': total_score,
                'max_score': total_max,
                'percentage': round((total_score / total_max) * 100, 1),
                'rating': overall_rating
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing financial health: {e}")
            return {}
    
    def get_peer_comparison(self, symbol: str, sector: str) -> Dict[str, Any]:
        """Compare company metrics with sector peers"""
        try:
            # This is a simplified peer comparison
            # In production, you'd want to use a comprehensive peer list
            sector_peers = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
                'Consumer Cyclical': ['TSLA', 'HD', 'MCD', 'NKE', 'SBUX'],
                'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
            }
            
            peers = sector_peers.get(sector, [])
            if symbol in peers:
                peers.remove(symbol)
            
            peer_data = []
            for peer in peers[:5]:  # Limit to top 5 peers
                try:
                    peer_info = yf.Ticker(peer).info
                    peer_data.append({
                        'symbol': peer,
                        'name': peer_info.get('longName', peer),
                        'market_cap': peer_info.get('marketCap', 0),
                        'pe_ratio': peer_info.get('trailingPE'),
                        'profit_margin': peer_info.get('profitMargins'),
                        'roe': peer_info.get('returnOnEquity')
                    })
                except:
                    continue
            
            return {
                'peers': peer_data,
                'sector_average': self._calculate_sector_averages(peer_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting peer comparison for {symbol}: {e}")
            return {}
    
    def generate_forecast(self, symbol: str) -> Dict[str, Any]:
        """Generate basic financial forecasts"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get historical data for trend analysis
            hist_data = ticker.history(period="3y")
            financials = ticker.financials
            
            forecast = {}
            
            # Revenue forecast based on growth rate
            revenue_growth = info.get('revenueGrowth', 0.05)  # Default 5%
            current_revenue = info.get('totalRevenue', 0)
            
            if current_revenue > 0:
                forecast['revenue_projection'] = {
                    'year_1': current_revenue * (1 + revenue_growth),
                    'year_2': current_revenue * (1 + revenue_growth) ** 2,
                    'year_3': current_revenue * (1 + revenue_growth) ** 3,
                    'growth_rate': revenue_growth
                }
            
            # EPS forecast
            current_eps = info.get('trailingEps', 0)
            earnings_growth = info.get('earningsGrowth', 0.1)  # Default 10%
            
            if current_eps > 0:
                forecast['eps_projection'] = {
                    'year_1': current_eps * (1 + earnings_growth),
                    'year_2': current_eps * (1 + earnings_growth) ** 2,
                    'year_3': current_eps * (1 + earnings_growth) ** 3,
                    'growth_rate': earnings_growth
                }
            
            # Price target based on forward P/E
            forward_pe = info.get('forwardPE', info.get('trailingPE', 15))
            if current_eps > 0 and forward_pe:
                target_eps = current_eps * (1 + earnings_growth)
                forecast['price_target'] = {
                    'conservative': target_eps * (forward_pe * 0.8),
                    'base_case': target_eps * forward_pe,
                    'optimistic': target_eps * (forward_pe * 1.2)
                }
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast for {symbol}: {e}")
            return {}
    
    def _safe_divide(self, numerator, denominator):
        """Safely divide two numbers"""
        try:
            if denominator == 0:
                return None
            return numerator / denominator
        except:
            return None
    
    def _get_rating(self, score: int, max_score: int) -> str:
        """Convert score to rating"""
        percentage = (score / max_score) * 100
        if percentage >= 80:
            return "Excellent"
        elif percentage >= 60:
            return "Good"
        elif percentage >= 40:
            return "Fair"
        elif percentage >= 20:
            return "Poor"
        else:
            return "Very Poor"
    
    def _calculate_sector_averages(self, peer_data: list) -> Dict[str, float]:
        """Calculate sector averages from peer data"""
        averages = {}
        if not peer_data:
            return averages
        
        metrics = ['pe_ratio', 'profit_margin', 'roe']
        for metric in metrics:
            values = [p[metric] for p in peer_data if p[metric] is not None]
            if values:
                averages[metric] = sum(values) / len(values)
        
        return averages