#!/usr/bin/env python3
"""
Debug script to test news data fetching from various sources
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.news_service import NewsService
from src.ai.news_analyzer import AdvancedNewsAnalyzer
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_individual_news_sources():
    """Test each news source individually"""
    print("=== DEBUGGING NEWS SOURCES ===")
    
    news_service = NewsService()
    test_symbol = "AAPL"
    
    print(f"Testing news sources for {test_symbol}")
    print("-" * 50)
    
    # Test Yahoo Finance
    print("\n1. Testing Yahoo Finance...")
    try:
        yahoo_articles = news_service._get_yahoo_news(test_symbol, 7)
        print(f"   Yahoo Finance: {len(yahoo_articles)} articles")
        if yahoo_articles:
            for i, article in enumerate(yahoo_articles[:2]):
                print(f"   {i+1}. {article.title}")
                print(f"      Publisher: {article.publisher}")
                print(f"      Date: {article.published_date}")
                print(f"      Summary: {article.summary[:100]}...")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test Google News
    print("\n2. Testing Google News...")
    try:
        google_articles = news_service._get_google_news(test_symbol, 7)
        print(f"   Google News: {len(google_articles)} articles")
        if google_articles:
            for i, article in enumerate(google_articles[:2]):
                print(f"   {i+1}. {article.title}")
                print(f"      Publisher: {article.publisher}")
                print(f"      Date: {article.published_date}")
                print(f"      Summary: {article.summary[:100]}...")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test Finviz
    print("\n3. Testing Finviz...")
    try:
        finviz_articles = news_service._get_finviz_news(test_symbol, 7)
        print(f"   Finviz: {len(finviz_articles)} articles")
        if finviz_articles:
            for i, article in enumerate(finviz_articles[:2]):
                print(f"   {i+1}. {article.title}")
                print(f"      Publisher: {article.publisher}")
                print(f"      Date: {article.published_date}")
                print(f"      Summary: {article.summary[:100]}...")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test MarketWatch
    print("\n4. Testing MarketWatch...")
    try:
        mw_articles = news_service._get_marketwatch_news(test_symbol, 7)
        print(f"   MarketWatch: {len(mw_articles)} articles")
        if mw_articles:
            for i, article in enumerate(mw_articles[:2]):
                print(f"   {i+1}. {article.title}")
                print(f"      Publisher: {article.publisher}")
                print(f"      Date: {article.published_date}")
                print(f"      Summary: {article.summary[:100]}...")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test Investing.com
    print("\n5. Testing Investing.com...")
    try:
        inv_articles = news_service._get_investing_news(test_symbol, 7)
        print(f"   Investing.com: {len(inv_articles)} articles")
        if inv_articles:
            for i, article in enumerate(inv_articles[:2]):
                print(f"   {i+1}. {article.title}")
                print(f"      Publisher: {article.publisher}")
                print(f"      Date: {article.published_date}")
                print(f"      Summary: {article.summary[:100]}...")
    except Exception as e:
        print(f"   ERROR: {e}")

def test_aggregated_news():
    """Test the main aggregated news function"""
    print("\n\n=== TESTING AGGREGATED NEWS ===")
    
    news_service = NewsService()
    test_symbol = "AAPL"
    
    try:
        print(f"Fetching aggregated news for {test_symbol}...")
        articles = news_service.get_stock_news(test_symbol, days_back=7, max_articles=10)
        print(f"Total articles found: {len(articles)}")
        
        if articles:
            print("\nArticle details:")
            for i, article in enumerate(articles):
                print(f"{i+1}. {article.title}")
                print(f"   Publisher: {article.publisher}")
                print(f"   Date: {article.published_date}")
                print(f"   Link: {article.link}")
                print(f"   Summary: {article.summary[:100]}...")
                print()
        else:
            print("No articles found!")
            
    except Exception as e:
        print(f"ERROR in aggregated news: {e}")
        import traceback
        traceback.print_exc()

def test_news_analysis():
    """Test the news analysis functionality"""
    print("\n\n=== TESTING NEWS ANALYSIS ===")
    
    try:
        analyzer = AdvancedNewsAnalyzer()
        test_symbol = "AAPL"
        
        print(f"Running comprehensive analysis for {test_symbol}...")
        result = analyzer.analyze_stock_news_comprehensive(test_symbol)
        
        print(f"Analysis Results:")
        print(f"  Trading Signal: {result.trading_signal}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Market Impact: {result.market_impact}")
        print(f"  Sentiment: {result.overall_sentiment}")
        print(f"  Sentiment Score: {result.sentiment_score}")
        print(f"  Articles Analyzed: {result.articles_analyzed}")
        print(f"  Target Action: {result.target_action}")
        
        if result.risk_factors:
            print(f"  Risk Factors: {result.risk_factors}")
        
        if result.catalysts:
            print(f"  Catalysts: {result.catalysts}")
            
    except Exception as e:
        print(f"ERROR in news analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_individual_news_sources()
    test_aggregated_news()
    test_news_analysis()
    print("\n=== DEBUG COMPLETE ===")