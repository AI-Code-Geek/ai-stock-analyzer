import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import re
from enum import Enum

from src.data.news_service import NewsService, NewsArticle
from src.ai.ollama_client import AIAnalyzer

logger = logging.getLogger(__name__)

class MarketImpact(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEUTRAL = "neutral"

class TradingSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class NewsAnalysisResult:
    symbol: str
    overall_sentiment: str
    sentiment_score: float
    market_impact: MarketImpact
    trading_signal: TradingSignal
    confidence: float
    key_themes: List[str]
    risk_factors: List[str]
    catalysts: List[str]
    ai_summary: str
    recommendation_reasoning: str
    target_action: str
    articles_analyzed: int
    analysis_timestamp: datetime

class AdvancedNewsAnalyzer:
    """Advanced news aggregation and AI analysis system for trading decisions"""
    
    def __init__(self):
        self.news_service = NewsService()
        self.ai_analyzer = AIAnalyzer()
        
        # Market-moving keywords and their impact weights
        self.impact_keywords = {
            'high': [
                'earnings', 'bankruptcy', 'merger', 'acquisition', 'takeover',
                'fda approval', 'clinical trial', 'lawsuit', 'investigation',
                'dividend', 'stock split', 'guidance', 'forecast', 'outlook',
                'ceo', 'resignation', 'fired', 'appointed', 'management change'
            ],
            'medium': [
                'revenue', 'sales', 'growth', 'expansion', 'partnership',
                'contract', 'deal', 'investment', 'funding', 'ipo',
                'analyst', 'upgrade', 'downgrade', 'rating', 'target price'
            ],
            'low': [
                'conference', 'presentation', 'interview', 'comment',
                'statement', 'announcement', 'update', 'news', 'report'
            ]
        }
        
        # Sentiment keywords for financial context
        self.sentiment_keywords = {
            'positive': [
                'beat', 'exceeded', 'strong', 'growth', 'profit', 'gain',
                'success', 'bullish', 'optimistic', 'breakthrough', 'rally',
                'surge', 'outperform', 'record', 'milestone', 'achievement'
            ],
            'negative': [
                'miss', 'disappointing', 'weak', 'decline', 'loss', 'bearish',
                'pessimistic', 'concern', 'risk', 'challenge', 'problem',
                'fall', 'drop', 'crash', 'underperform', 'warning', 'alert'
            ]
        }
    
    def analyze_stock_news_comprehensive(self, symbol: str, days_back: int = 7) -> NewsAnalysisResult:
        """Perform comprehensive news analysis for trading decisions"""
        try:
            logger.info(f"Starting comprehensive news analysis for {symbol}")
            
            # Get news from multiple sources
            news_summary = self.news_service.get_news_summary(symbol)
            articles = news_summary['articles']
            
            if not articles:
                return self._create_default_result(symbol, "No news available")
            
            # Analyze each article
            article_analyses = []
            for article in articles:
                analysis = self._analyze_single_article(article, symbol)
                article_analyses.append(analysis)
            
            # Aggregate analysis results
            aggregated_result = self._aggregate_analyses(symbol, articles, article_analyses)
            
            # Get AI-powered comprehensive analysis
            ai_analysis = self._get_ai_comprehensive_analysis(symbol, articles, aggregated_result)
            
            # Generate trading recommendation
            trading_recommendation = self._generate_trading_recommendation(
                aggregated_result, ai_analysis
            )
            
            # Create final result
            result = NewsAnalysisResult(
                symbol=symbol,
                overall_sentiment=aggregated_result['sentiment'],
                sentiment_score=aggregated_result['sentiment_score'],
                market_impact=aggregated_result['market_impact'],
                trading_signal=trading_recommendation['signal'],
                confidence=trading_recommendation['confidence'],
                key_themes=aggregated_result['themes'],
                risk_factors=aggregated_result['risks'],
                catalysts=aggregated_result['catalysts'],
                ai_summary=ai_analysis.get('summary', 'AI analysis unavailable'),
                recommendation_reasoning=trading_recommendation['reasoning'],
                target_action=trading_recommendation['action'],
                articles_analyzed=len(articles),
                analysis_timestamp=datetime.now()
            )
            
            logger.info(f"Completed news analysis for {symbol}: {result.trading_signal.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive news analysis for {symbol}: {e}")
            return self._create_default_result(symbol, f"Analysis error: {str(e)}")
    
    def _analyze_single_article(self, article: NewsArticle, symbol: str) -> Dict[str, Any]:
        """Analyze a single news article for market impact and sentiment"""
        try:
            content = f"{article.title} {article.summary}".lower()
            
            # Calculate market impact
            impact_score = 0
            for impact_level, keywords in self.impact_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in content)
                if impact_level == 'high':
                    impact_score += matches * 3
                elif impact_level == 'medium':
                    impact_score += matches * 2
                else:
                    impact_score += matches * 1
            
            # Determine market impact level
            if impact_score >= 5:
                market_impact = MarketImpact.HIGH
            elif impact_score >= 3:
                market_impact = MarketImpact.MEDIUM
            elif impact_score >= 1:
                market_impact = MarketImpact.LOW
            else:
                market_impact = MarketImpact.NEUTRAL
            
            # Calculate sentiment
            pos_score = sum(1 for word in self.sentiment_keywords['positive'] if word in content)
            neg_score = sum(1 for word in self.sentiment_keywords['negative'] if word in content)
            
            if pos_score > neg_score:
                sentiment = 'positive'
                sentiment_score = min(pos_score / (pos_score + neg_score + 1), 1.0)
            elif neg_score > pos_score:
                sentiment = 'negative'
                sentiment_score = -min(neg_score / (pos_score + neg_score + 1), 1.0)
            else:
                sentiment = 'neutral'
                sentiment_score = 0.0
            
            # Extract themes and keywords
            themes = self._extract_themes(content)
            
            return {
                'article': article,
                'market_impact': market_impact,
                'impact_score': impact_score,
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'themes': themes,
                'recency_weight': self._calculate_recency_weight(article.published_date)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing article: {e}")
            return {
                'article': article,
                'market_impact': MarketImpact.NEUTRAL,
                'impact_score': 0,
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'themes': [],
                'recency_weight': 0.5
            }
    
    def _aggregate_analyses(self, symbol: str, articles: List[NewsArticle], 
                          analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual article analyses into overall assessment"""
        try:
            if not analyses:
                return self._get_neutral_aggregation()
            
            # Calculate weighted sentiment
            total_weight = 0
            weighted_sentiment = 0
            
            for analysis in analyses:
                weight = analysis['recency_weight'] * (analysis['impact_score'] + 1)
                weighted_sentiment += analysis['sentiment_score'] * weight
                total_weight += weight
            
            overall_sentiment_score = weighted_sentiment / total_weight if total_weight > 0 else 0
            
            # Determine overall sentiment
            if overall_sentiment_score > 0.2:
                overall_sentiment = 'positive'
            elif overall_sentiment_score < -0.2:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            # Determine overall market impact
            high_impact_count = sum(1 for a in analyses if a['market_impact'] == MarketImpact.HIGH)
            medium_impact_count = sum(1 for a in analyses if a['market_impact'] == MarketImpact.MEDIUM)
            
            if high_impact_count >= 2 or (high_impact_count >= 1 and len(analyses) <= 3):
                overall_impact = MarketImpact.HIGH
            elif high_impact_count >= 1 or medium_impact_count >= 2:
                overall_impact = MarketImpact.MEDIUM
            elif medium_impact_count >= 1:
                overall_impact = MarketImpact.LOW
            else:
                overall_impact = MarketImpact.NEUTRAL
            
            # Extract key themes
            all_themes = []
            for analysis in analyses:
                all_themes.extend(analysis['themes'])
            
            # Count theme frequency and get top themes
            theme_counts = {}
            for theme in all_themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            key_themes = sorted(theme_counts.keys(), key=lambda x: theme_counts[x], reverse=True)[:5]
            
            # Identify risks and catalysts
            risks = self._identify_risks(analyses)
            catalysts = self._identify_catalysts(analyses)
            
            return {
                'sentiment': overall_sentiment,
                'sentiment_score': overall_sentiment_score,
                'market_impact': overall_impact,
                'themes': key_themes,
                'risks': risks,
                'catalysts': catalysts,
                'article_count': len(analyses)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating analyses: {e}")
            return self._get_neutral_aggregation()
    
    def _get_ai_comprehensive_analysis(self, symbol: str, articles: List[NewsArticle], 
                                     aggregated_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered comprehensive analysis of news"""
        try:
            if not self.ai_analyzer.ollama_client.check_connection():
                return {'summary': 'AI analysis unavailable - OLLAMA not connected'}
            
            # Create comprehensive analysis prompt
            news_context = self._format_news_for_ai_analysis(articles, aggregated_result)
            
            analysis_prompt = f"""
            You are a professional financial analyst with expertise in news analysis and trading decisions.
            Analyze the following comprehensive news data for {symbol} and provide a detailed investment assessment.

            NEWS ANALYSIS SUMMARY:
            - Overall Sentiment: {aggregated_result['sentiment']} (Score: {aggregated_result['sentiment_score']:.2f})
            - Market Impact Level: {aggregated_result['market_impact'].value}
            - Key Themes: {', '.join(aggregated_result['themes'])}
            - Risk Factors: {', '.join(aggregated_result['risks'])}
            - Potential Catalysts: {', '.join(aggregated_result['catalysts'])}
            - Articles Analyzed: {aggregated_result['article_count']}

            DETAILED NEWS CONTENT:
            {news_context}

            REQUIRED ANALYSIS:
            Provide a comprehensive investment analysis covering:

            1. **News Impact Assessment**: How will this news likely affect {symbol}'s stock price?
            2. **Market Sentiment**: What is the overall market sentiment and confidence level?
            3. **Risk Analysis**: What are the key risks and potential negative catalysts?
            4. **Opportunity Analysis**: What opportunities and positive catalysts exist?
            5. **Timeline Considerations**: Short-term vs long-term implications
            6. **Trading Recommendation**: Specific buy/sell/hold recommendation with reasoning

            Format your response as a clear, actionable analysis for investment decision-making.
            Focus on concrete, data-driven insights rather than generic statements.
            """
            
            ai_response = self.ai_analyzer.ollama_client.generate_response(analysis_prompt)
            
            return {
                'summary': ai_response or 'AI analysis failed',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in AI comprehensive analysis: {e}")
            return {'summary': f'AI analysis error: {str(e)}'}
    
    def _generate_trading_recommendation(self, aggregated_result: Dict[str, Any], 
                                       ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific trading recommendation based on analysis"""
        try:
            sentiment_score = aggregated_result['sentiment_score']
            market_impact = aggregated_result['market_impact']
            
            # Base recommendation logic
            if sentiment_score > 0.6 and market_impact in [MarketImpact.HIGH, MarketImpact.MEDIUM]:
                signal = TradingSignal.STRONG_BUY
                confidence = 0.85
                action = "Consider strong buy position - highly positive news with significant market impact"
            elif sentiment_score > 0.3 and market_impact in [MarketImpact.MEDIUM, MarketImpact.HIGH]:
                signal = TradingSignal.BUY
                confidence = 0.75
                action = "Consider buy position - positive sentiment with notable impact"
            elif sentiment_score < -0.6 and market_impact in [MarketImpact.HIGH, MarketImpact.MEDIUM]:
                signal = TradingSignal.STRONG_SELL
                confidence = 0.85
                action = "Consider strong sell position - highly negative news with significant impact"
            elif sentiment_score < -0.3 and market_impact in [MarketImpact.MEDIUM, MarketImpact.HIGH]:
                signal = TradingSignal.SELL
                confidence = 0.75
                action = "Consider sell position - negative sentiment with notable impact"
            elif abs(sentiment_score) < 0.2 or market_impact == MarketImpact.NEUTRAL:
                signal = TradingSignal.HOLD
                confidence = 0.65
                action = "Hold current position - neutral sentiment or limited market impact"
            else:
                signal = TradingSignal.HOLD
                confidence = 0.60
                action = "Hold and monitor - mixed signals require further observation"
            
            # Adjust confidence based on article count and recency
            if aggregated_result['article_count'] >= 5:
                confidence += 0.1
            elif aggregated_result['article_count'] <= 2:
                confidence -= 0.1
            
            confidence = max(0.3, min(0.95, confidence))
            
            # Create reasoning
            reasoning = self._create_recommendation_reasoning(
                signal, sentiment_score, market_impact, aggregated_result
            )
            
            return {
                'signal': signal,
                'confidence': confidence,
                'action': action,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error generating trading recommendation: {e}")
            return {
                'signal': TradingSignal.HOLD,
                'confidence': 0.5,
                'action': 'Hold - unable to generate recommendation due to analysis error',
                'reasoning': f'Recommendation error: {str(e)}'
            }
    
    def _create_recommendation_reasoning(self, signal: TradingSignal, sentiment_score: float,
                                       market_impact: MarketImpact, aggregated_result: Dict[str, Any]) -> str:
        """Create detailed reasoning for trading recommendation"""
        reasoning_parts = []
        
        # Sentiment reasoning
        if sentiment_score > 0.5:
            reasoning_parts.append("Strong positive sentiment from recent news coverage")
        elif sentiment_score > 0.2:
            reasoning_parts.append("Moderately positive sentiment trend")
        elif sentiment_score < -0.5:
            reasoning_parts.append("Strong negative sentiment from recent developments")
        elif sentiment_score < -0.2:
            reasoning_parts.append("Moderately negative sentiment trend")
        else:
            reasoning_parts.append("Neutral sentiment with mixed signals")
        
        # Impact reasoning
        if market_impact == MarketImpact.HIGH:
            reasoning_parts.append("High market impact events detected")
        elif market_impact == MarketImpact.MEDIUM:
            reasoning_parts.append("Moderate market impact expected")
        else:
            reasoning_parts.append("Limited market impact anticipated")
        
        # Theme-based reasoning
        if aggregated_result['themes']:
            key_theme = aggregated_result['themes'][0]
            reasoning_parts.append(f"Key focus area: {key_theme}")
        
        # Risk/catalyst reasoning
        if aggregated_result['risks']:
            reasoning_parts.append(f"Key risks: {', '.join(aggregated_result['risks'][:2])}")
        
        if aggregated_result['catalysts']:
            reasoning_parts.append(f"Potential catalysts: {', '.join(aggregated_result['catalysts'][:2])}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _extract_themes(self, content: str) -> List[str]:
        """Extract key themes from article content"""
        themes = []
        
        # Financial themes
        financial_themes = {
            'earnings': ['earnings', 'eps', 'profit', 'revenue', 'income'],
            'growth': ['growth', 'expansion', 'scaling', 'increase'],
            'merger_acquisition': ['merger', 'acquisition', 'takeover', 'deal'],
            'regulatory': ['fda', 'sec', 'regulatory', 'compliance', 'approval'],
            'management': ['ceo', 'cfo', 'management', 'leadership', 'executive'],
            'product': ['product', 'launch', 'release', 'innovation', 'technology'],
            'financial_health': ['debt', 'cash', 'liquidity', 'financing', 'funding']
        }
        
        for theme, keywords in financial_themes.items():
            if any(keyword in content for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _identify_risks(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify key risk factors from news analyses"""
        risks = []
        
        risk_indicators = [
            'lawsuit', 'investigation', 'regulatory', 'competition',
            'debt', 'loss', 'decline', 'warning', 'concern', 'challenge'
        ]
        
        for analysis in analyses:
            if analysis['sentiment'] == 'negative':
                content = f"{analysis['article'].title} {analysis['article'].summary}".lower()
                for indicator in risk_indicators:
                    if indicator in content and indicator not in risks:
                        risks.append(indicator.replace('_', ' ').title())
        
        return risks[:3]  # Return top 3 risks
    
    def _identify_catalysts(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify potential positive catalysts"""
        catalysts = []
        
        catalyst_indicators = [
            'earnings', 'approval', 'partnership', 'expansion', 'growth',
            'breakthrough', 'success', 'milestone', 'achievement', 'upgrade'
        ]
        
        for analysis in analyses:
            if analysis['sentiment'] == 'positive':
                content = f"{analysis['article'].title} {analysis['article'].summary}".lower()
                for indicator in catalyst_indicators:
                    if indicator in content and indicator not in catalysts:
                        catalysts.append(indicator.replace('_', ' ').title())
        
        return catalysts[:3]  # Return top 3 catalysts
    
    def _calculate_recency_weight(self, published_date: datetime) -> float:
        """Calculate recency weight for article importance"""
        try:
            hours_ago = (datetime.now() - published_date).total_seconds() / 3600
            
            if hours_ago <= 6:
                return 1.0  # Very recent
            elif hours_ago <= 24:
                return 0.8  # Recent
            elif hours_ago <= 72:
                return 0.6  # Moderately recent
            else:
                return 0.4  # Older news
                
        except Exception:
            return 0.5  # Default weight
    
    def _format_news_for_ai_analysis(self, articles: List[NewsArticle], 
                                    aggregated_result: Dict[str, Any]) -> str:
        """Format news articles for AI analysis"""
        formatted_news = ""
        
        for i, article in enumerate(articles[:5], 1):  # Limit to top 5 articles
            formatted_news += f"""
            Article {i}:
            Title: {article.title}
            Publisher: {article.publisher}
            Published: {article.published_date.strftime('%Y-%m-%d %H:%M')}
            Summary: {article.summary}
            ---
            """
        
        return formatted_news
    
    def _create_default_result(self, symbol: str, reason: str) -> NewsAnalysisResult:
        """Create default result when analysis fails"""
        return NewsAnalysisResult(
            symbol=symbol,
            overall_sentiment='neutral',
            sentiment_score=0.0,
            market_impact=MarketImpact.NEUTRAL,
            trading_signal=TradingSignal.HOLD,
            confidence=0.5,
            key_themes=[],
            risk_factors=[],
            catalysts=[],
            ai_summary=f"Analysis unavailable: {reason}",
            recommendation_reasoning="Unable to generate recommendation due to insufficient data",
            target_action="Hold and monitor for more news",
            articles_analyzed=0,
            analysis_timestamp=datetime.now()
        )
    
    def _get_neutral_aggregation(self) -> Dict[str, Any]:
        """Get neutral aggregation result"""
        return {
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'market_impact': MarketImpact.NEUTRAL,
            'themes': [],
            'risks': [],
            'catalysts': [],
            'article_count': 0
        }