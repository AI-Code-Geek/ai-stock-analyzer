#!/usr/bin/env python3
"""
Test script to verify news analysis UI integration is working properly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui.financial_reports import FinancialReportGenerator
from src.data.news_service import NewsService
from src.ai.news_analyzer import AdvancedNewsAnalyzer

def test_ui_integration():
    """Test the UI integration for news analysis"""
    print("=== TESTING UI INTEGRATION ===")
    
    try:
        # Initialize components
        print("1. Initializing components...")
        news_service = NewsService()
        news_analyzer = AdvancedNewsAnalyzer()
        financial_reports = FinancialReportGenerator()
        print("   ✅ All components initialized successfully")
        
        # Test symbol
        test_symbol = "AAPL"
        print(f"\n2. Testing news fetching for {test_symbol}...")
        
        # Test basic news fetching
        articles = news_service.get_stock_news(test_symbol, days_back=7, max_articles=5)
        print(f"   ✅ Fetched {len(articles)} articles")
        
        if articles:
            print("   Sample articles:")
            for i, article in enumerate(articles[:2], 1):
                print(f"     {i}. '{article.title[:50]}...' ({article.publisher})")
        
        # Test news analysis
        print(f"\n3. Testing comprehensive analysis...")
        analysis_result = news_analyzer.analyze_stock_news_comprehensive(test_symbol)
        print(f"   ✅ Analysis completed")
        print(f"   📊 Trading Signal: {analysis_result.trading_signal.value}")
        print(f"   📈 Confidence: {analysis_result.confidence:.1%}")
        print(f"   🎯 Target Action: {analysis_result.target_action}")
        
        # Test news summary functionality
        print(f"\n4. Testing news summary...")
        news_summary = news_service.get_news_summary(test_symbol)
        print(f"   ✅ News summary generated")
        print(f"   📰 Articles: {news_summary['news_count']}")
        print(f"   😊 Sentiment: {news_summary['sentiment_analysis']['overall_sentiment']}")
        
        # Test data structure for UI
        print(f"\n5. Testing UI data structure...")
        formatted_news = news_service.format_news_for_ai(articles, test_symbol)
        print(f"   ✅ Formatted news for AI ({len(formatted_news)} characters)")
        
        print("\n" + "=" * 50)
        print("🎉 UI INTEGRATION TEST SUCCESSFUL!")
        print("✅ News fetching working")
        print("✅ Analysis engine functional") 
        print("✅ Trading signals generating")
        print("✅ Data properly formatted for UI")
        print("🚀 Ready for Streamlit display!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during UI integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_quality():
    """Test the quality of news data being returned"""
    print("\n" + "=" * 50)
    print("=== TESTING DATA QUALITY ===")
    
    try:
        news_service = NewsService()
        test_symbol = "TSLA"  # Use different symbol
        
        print(f"Testing data quality for {test_symbol}...")
        
        articles = news_service.get_stock_news(test_symbol, days_back=3, max_articles=5)
        
        quality_checks = {
            "articles_found": len(articles) > 0,
            "titles_not_empty": all(article.title and article.title.strip() for article in articles),
            "summaries_not_empty": all(article.summary and article.summary.strip() for article in articles),
            "publishers_identified": all(article.publisher and article.publisher.strip() for article in articles),
            "links_provided": all(article.link for article in articles),
            "dates_valid": all(article.published_date for article in articles)
        }
        
        print(f"\nQuality Checks for {len(articles)} articles:")
        for check, passed in quality_checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check.replace('_', ' ').title()}: {passed}")
        
        if articles:
            print(f"\nSample Article Quality:")
            article = articles[0]
            print(f"   Title: '{article.title}'")
            print(f"   Publisher: '{article.publisher}'")
            print(f"   Summary: '{article.summary[:100]}...'")
            print(f"   Date: {article.published_date}")
            print(f"   Link: {article.link}")
        
        all_passed = all(quality_checks.values())
        print(f"\n🎯 Overall Quality: {'✅ EXCELLENT' if all_passed else '⚠️ NEEDS IMPROVEMENT'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error during data quality test: {e}")
        return False

if __name__ == "__main__":
    ui_success = test_ui_integration()
    quality_success = test_data_quality()
    
    if ui_success and quality_success:
        print("\n🏆 ALL TESTS PASSED - NEWS SYSTEM READY!")
    else:
        print("\n💥 Some tests failed - check output above")
        sys.exit(1)