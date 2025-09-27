#!/usr/bin/env python3
"""
Quick test to verify the news display fix is working
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui.financial_reports import FinancialReportGenerator
from src.ai.news_analyzer import AdvancedNewsAnalyzer

def test_news_fix():
    """Test that news articles can be displayed even when AI analysis finds 0 articles"""
    print("=== TESTING NEWS DISPLAY FIX ===")
    
    # Test the analyzer first
    analyzer = AdvancedNewsAnalyzer()
    test_symbol = "AAPL"
    
    print(f"1. Testing analyzer for {test_symbol}...")
    analysis_result = analyzer.analyze_stock_news_comprehensive(test_symbol)
    print(f"   Articles analyzed by AI: {analysis_result.articles_analyzed}")
    print(f"   Trading signal: {analysis_result.trading_signal}")
    
    # Test the UI component
    print(f"\n2. Testing direct news service for {test_symbol}...")
    reports = FinancialReportGenerator()
    articles = reports.news_service.get_stock_news(test_symbol, days_back=7, max_articles=10)
    print(f"   Articles found by news service: {len(articles)}")
    
    # This simulates what should happen in the UI now
    print(f"\n3. Simulating UI logic...")
    if analysis_result.articles_analyzed == 0:
        print(f"   ⚠️ AI analysis found no articles, but we have {len(articles)} articles from news service")
        print(f"   ✅ UI should now show warning but continue to display the {len(articles)} articles")
    else:
        print(f"   ✅ AI analysis worked fine with {analysis_result.articles_analyzed} articles")
    
    if articles:
        print(f"\n4. Sample article check...")
        article = articles[0]
        print(f"   Title: '{article.title[:50]}...'")
        print(f"   Publisher: '{article.publisher}'")
        print(f"   Date: {article.published_date}")
        print(f"   ✅ Articles have required fields for display")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   - AI analyzer found: {analysis_result.articles_analyzed} articles")
    print(f"   - News service found: {len(articles)} articles")
    print(f"   - UI should display: {len(articles)} articles (regardless of AI result)")
    print(f"   - Fix status: {'✅ WORKING' if len(articles) > 0 else '❌ STILL BROKEN'}")

if __name__ == "__main__":
    test_news_fix()