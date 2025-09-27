import yfinance as yf
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
import feedparser
from bs4 import BeautifulSoup
import urllib.parse

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    title: str
    summary: str
    link: str
    publisher: str
    published_date: datetime
    sentiment: Optional[str] = None
    relevance_score: Optional[float] = None

class NewsService:
    """Service to fetch and process stock-related news from multiple free sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 1800  # 30 minutes cache
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Free news sources configuration
        self.news_sources = {
            'yahoo_finance': {
                'name': 'Yahoo Finance',
                'method': self._get_yahoo_news
            },
            'google_finance': {
                'name': 'Google Finance',
                'method': self._get_google_news
            },
            'finviz': {
                'name': 'Finviz',
                'method': self._get_finviz_news
            },
            'marketwatch': {
                'name': 'MarketWatch',
                'method': self._get_marketwatch_news
            },
            'investing_com': {
                'name': 'Investing.com',
                'method': self._get_investing_news
            }
        }
    
    def get_stock_news(self, symbol: str, days_back: int = 7, max_articles: int = 10) -> List[NewsArticle]:
        """Get recent news articles for a stock symbol from multiple sources"""
        try:
            cache_key = f"{symbol}_{days_back}_{max_articles}"
            
            # Check cache
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if time.time() - cached_time < self.cache_duration:
                    return cached_data
            
            all_articles = []
            
            # Fetch from multiple sources
            for source_name, source_config in self.news_sources.items():
                try:
                    source_articles = source_config['method'](symbol, days_back)
                    all_articles.extend(source_articles)
                except Exception as e:
                    logger.warning(f"Error fetching from {source_name}: {e}")
                    continue
            
            # Remove duplicates based on title similarity
            unique_articles = self._remove_duplicate_articles(all_articles)
            
            # Sort by date and limit results
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_articles = [
                article for article in unique_articles 
                if article.published_date >= cutoff_date
            ]
            
            # Sort by date (newest first) and limit
            filtered_articles.sort(key=lambda x: x.published_date, reverse=True)
            final_articles = filtered_articles[:max_articles]
            
            # Cache the results
            self.cache[cache_key] = (time.time(), final_articles)
            
            return final_articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_market_news(self, max_articles: int = 5) -> List[NewsArticle]:
        """Get general market news"""
        try:
            # Use SPY as proxy for market news
            return self.get_stock_news("SPY", days_back=3, max_articles=max_articles)
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []
    
    def format_news_for_ai(self, articles: List[NewsArticle], symbol: str) -> str:
        """Format news articles for AI analysis"""
        if not articles:
            return f"No recent news available for {symbol}."
        
        formatted_news = f"RECENT NEWS FOR {symbol} (Last 7 days):\n\n"
        
        for i, article in enumerate(articles, 1):
            formatted_news += f"{i}. **{article.title}**\n"
            formatted_news += f"   Publisher: {article.publisher}\n"
            formatted_news += f"   Date: {article.published_date.strftime('%Y-%m-%d %H:%M')}\n"
            formatted_news += f"   Summary: {article.summary[:200]}{'...' if len(article.summary) > 200 else ''}\n\n"
        
        return formatted_news
    
    def analyze_news_sentiment(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Basic news sentiment analysis"""
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Simple keyword-based sentiment analysis
        positive_keywords = [
            'growth', 'profit', 'beat', 'strong', 'rise', 'gain', 'up', 'bullish',
            'positive', 'increase', 'success', 'win', 'breakthrough', 'expansion',
            'upgrade', 'outperform', 'buy', 'optimistic', 'rally', 'surge'
        ]
        
        negative_keywords = [
            'loss', 'down', 'fall', 'decline', 'drop', 'bear', 'negative',
            'concern', 'risk', 'challenge', 'problem', 'issue', 'weakness',
            'downgrade', 'sell', 'pessimistic', 'crash', 'plunge', 'struggle'
        ]
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            text = (article.title + " " + article.summary).lower()
            
            pos_score = sum(1 for keyword in positive_keywords if keyword in text)
            neg_score = sum(1 for keyword in negative_keywords if keyword in text)
            
            if pos_score > neg_score:
                positive_count += 1
                article.sentiment = 'positive'
            elif neg_score > pos_score:
                negative_count += 1
                article.sentiment = 'negative'
            else:
                neutral_count += 1
                article.sentiment = 'neutral'
        
        total_articles = len(articles)
        sentiment_score = (positive_count - negative_count) / total_articles if total_articles > 0 else 0
        
        if sentiment_score > 0.2:
            overall_sentiment = 'positive'
        elif sentiment_score < -0.2:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': total_articles
        }
    
    def get_news_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive news summary for a stock"""
        try:
            articles = self.get_stock_news(symbol)
            sentiment_analysis = self.analyze_news_sentiment(articles)
            
            return {
                'symbol': symbol,
                'articles': articles,
                'sentiment_analysis': sentiment_analysis,
                'news_count': len(articles),
                'last_updated': datetime.now(),
                'formatted_for_ai': self.format_news_for_ai(articles, symbol)
            }
            
        except Exception as e:
            logger.error(f"Error getting news summary for {symbol}: {e}")
            return {
                'symbol': symbol,
                'articles': [],
                'sentiment_analysis': self.analyze_news_sentiment([]),
                'news_count': 0,
                'last_updated': datetime.now(),
                'formatted_for_ai': f"No news data available for {symbol}."
            }
    
    def _get_yahoo_news(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            articles = []
            for item in news_data:
                try:
                    # Yahoo Finance has a nested structure: {'id': ..., 'content': {...}}
                    content = item.get('content', item)  # Fallback to item if no content key
                    
                    # Extract title from content object
                    title = (content.get('title') or 
                            content.get('headline') or 
                            item.get('title') or
                            'Yahoo Finance News Update')
                    
                    # Extract summary from content object
                    summary = (content.get('summary') or 
                              content.get('description') or 
                              item.get('summary') or
                              f"Latest news about {symbol} from Yahoo Finance")
                    
                    # Handle publisher information
                    provider_info = content.get('provider', {})
                    if isinstance(provider_info, dict):
                        publisher_name = provider_info.get('displayName', 'Yahoo Finance')
                    else:
                        publisher_name = 'Yahoo Finance'
                    
                    # Handle timestamp - try multiple fields
                    published_date = datetime.now()
                    timestamp_fields = ['providerPublishTime', 'pubDate', 'publishedAt', 'timestamp']
                    
                    for field in timestamp_fields:
                        timestamp = content.get(field) or item.get(field)
                        if timestamp:
                            try:
                                if isinstance(timestamp, str):
                                    # Try parsing ISO format timestamp
                                    if 'T' in timestamp and 'Z' in timestamp:
                                        published_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).replace(tzinfo=None)
                                    else:
                                        published_date = datetime.fromisoformat(timestamp)
                                else:
                                    published_date = datetime.fromtimestamp(timestamp)
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    # Get link from content or canonical URL
                    link = (content.get('canonicalUrl', {}).get('url') or
                           content.get('clickThroughUrl', {}).get('url') or
                           content.get('link') or
                           item.get('link', ''))
                    
                    article = NewsArticle(
                        title=title,
                        summary=summary[:500] if summary else f"Latest news about {symbol}",
                        link=link,
                        publisher=f"Yahoo Finance - {publisher_name}",
                        published_date=published_date
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Error processing Yahoo news item: {e}")
                    continue
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {e}")
            return []
    
    def _get_google_news(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from Google Finance (via RSS)"""
        try:
            # Google Finance RSS feed
            url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries:
                try:
                    # Parse published date
                    published_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    
                    # Extract and clean summary from description
                    summary = entry.get('summary', 'No summary available')
                    if hasattr(entry, 'content'):
                        summary = entry.content[0].value if entry.content else summary
                    
                    # Clean HTML markup from summary
                    summary = self._clean_html_markup(summary)
                    
                    # Extract publisher from source or title
                    source_info = entry.get('source', {})
                    if isinstance(source_info, dict):
                        publisher_name = source_info.get('title', 'Unknown')
                    else:
                        # Try to extract publisher from title (format: "Title - Publisher")
                        title_parts = entry.title.split(' - ')
                        if len(title_parts) > 1:
                            publisher_name = title_parts[-1]
                        else:
                            publisher_name = 'Unknown'
                    
                    article = NewsArticle(
                        title=entry.title,
                        summary=summary[:500] if summary else f"Latest news about {symbol}",
                        link=entry.link,
                        publisher=f"Google News - {publisher_name}",
                        published_date=published_date
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Error processing Google news item: {e}")
                    continue
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching Google Finance news: {e}")
            return []
    
    def _get_finviz_news(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from Finviz (fallback implementation)"""
        try:
            # Create a reference link to Finviz
            article = NewsArticle(
                title=f"Finviz Analysis for {symbol}",
                summary=f"View comprehensive stock analysis and news for {symbol} on Finviz",
                link=f"https://finviz.com/quote.ashx?t={symbol}",
                publisher="Finviz",
                published_date=datetime.now()
            )
            return [article]
        except Exception as e:
            logger.error(f"Error creating Finviz reference: {e}")
            return []
    
    def _get_marketwatch_news(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from MarketWatch (reference implementation)"""
        try:
            # Create a reference link to MarketWatch
            article = NewsArticle(
                title=f"MarketWatch Coverage for {symbol}",
                summary=f"Visit MarketWatch for the latest news and analysis on {symbol}",
                link=f"https://www.marketwatch.com/investing/stock/{symbol}",
                publisher="MarketWatch",
                published_date=datetime.now()
            )
            return [article]
        except Exception as e:
            logger.error(f"Error creating MarketWatch reference: {e}")
            return []
    
    def _get_investing_news(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from Investing.com (reference implementation)"""
        try:
            # Create a reference link to Investing.com
            article = NewsArticle(
                title=f"Investing.com Coverage for {symbol}",
                summary=f"Check Investing.com for comprehensive news and analysis on {symbol}",
                link=f"https://www.investing.com/search/?q={symbol}",
                publisher="Investing.com",
                published_date=datetime.now()
            )
            return [article]
        except Exception as e:
            logger.error(f"Error creating Investing.com reference: {e}")
            return []
    
    def _clean_html_markup(self, text: str) -> str:
        """Clean HTML markup and special characters from text"""
        try:
            if not text:
                return ""
            
            # Remove HTML tags using BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            clean_text = soup.get_text()
            
            # Remove extra whitespace and newlines
            clean_text = ' '.join(clean_text.split())
            
            # Remove common RSS/HTML artifacts
            clean_text = clean_text.replace('&nbsp;', ' ')
            clean_text = clean_text.replace('&amp;', '&')
            clean_text = clean_text.replace('&lt;', '<')
            clean_text = clean_text.replace('&gt;', '>')
            clean_text = clean_text.replace('&quot;', '"')
            clean_text = clean_text.replace('&#39;', "'")
            
            return clean_text.strip()
        except Exception as e:
            logger.warning(f"Error cleaning HTML markup: {e}")
            return text
    
    def _remove_duplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        try:
            unique_articles = []
            seen_titles = set()
            
            for article in articles:
                # Normalize title for comparison
                normalized_title = article.title.lower().strip()
                
                # Simple deduplication based on title
                is_duplicate = False
                for seen_title in seen_titles:
                    # Check for similar titles (>80% similarity in words)
                    title_words = set(normalized_title.split())
                    seen_words = set(seen_title.split())
                    
                    if title_words and seen_words:
                        similarity = len(title_words & seen_words) / len(title_words | seen_words)
                        if similarity > 0.8:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_articles.append(article)
                    seen_titles.add(normalized_title)
            
            return unique_articles
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            return articles