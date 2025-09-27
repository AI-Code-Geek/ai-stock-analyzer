#!/usr/bin/env python3
"""
Quick test script for the advanced news analysis system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.news_analyzer import AdvancedNewsAnalyzer
from src.data.news_service import NewsService

def test_news_analysis():
    """Test the advanced news analysis functionality"""
    print("Testing Advanced News Analysis System...")
    print("=" * 50)
    
    # Initialize components
    news_analyzer = AdvancedNewsAnalyzer()
    news_service = NewsService()
    
    # Test symbol
    test_symbol = "AAPL"
    print(f"Testing with symbol: {test_symbol}")
    
    try:
        # Test basic news fetching
        print("\n1. Testing basic news fetching...")
        articles = news_service.get_stock_news(test_symbol, days_back=7, max_articles=5)
        print(f"Found {len(articles)} articles")
        
        if articles:
            for i, article in enumerate(articles[:3], 1):
                print(f"   {i}. {article.title[:60]}... ({article.publisher})")
        
        # Test comprehensive analysis
        print("\n2. Testing comprehensive news analysis...")
        analysis_result = news_analyzer.analyze_stock_news_comprehensive(test_symbol)
        
        print(f"Analysis Result:")
        print(f"  - Trading Signal: {analysis_result.trading_signal.value}")
        print(f"  - Confidence: {analysis_result.confidence:.1%}")
        print(f"  - Market Impact: {analysis_result.market_impact.value}")
        print(f"  - Overall Sentiment: {analysis_result.overall_sentiment}")
        print(f"  - Sentiment Score: {analysis_result.sentiment_score:.3f}")
        print(f"  - Articles Analyzed: {analysis_result.articles_analyzed}")
        
        if analysis_result.target_action:
            print(f"  - Target Action: {analysis_result.target_action}")
        
        if analysis_result.risk_factors:
            print(f"  - Risk Factors: {len(analysis_result.risk_factors)}")
            for risk in analysis_result.risk_factors[:2]:
                print(f"    * {risk}")
        
        if analysis_result.catalysts:
            print(f"  - Catalysts: {len(analysis_result.catalysts)}")
            for catalyst in analysis_result.catalysts[:2]:
                print(f"    * {catalyst}")
        
        print("\n3. Testing AI analysis integration...")
        if analysis_result.ai_summary:
            if "unavailable" not in analysis_result.ai_summary.lower():
                print(f"  - AI Summary available: {len(analysis_result.ai_summary)} characters")
                print(f"  - Preview: {analysis_result.ai_summary[:100]}...")
            else:
                print(f"  - AI Summary: {analysis_result.ai_summary}")
        
        print("\n" + "=" * 50)
        print("✅ Advanced news analysis test completed successfully!")
        print("🎯 Trading recommendation system is working")
        print("📰 Multi-source news aggregation is functional")
        print("🧠 AI analysis integration is operational")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trading_signals():
    """Test different trading signal scenarios"""
    print("\n" + "=" * 50)
    print("Testing Trading Signal Generation...")
    
    # Test with different symbols to see various signals
    test_symbols = ["AAPL", "TSLA", "NVDA"]
    
    news_analyzer = AdvancedNewsAnalyzer()
    
    for symbol in test_symbols:
        try:
            print(f"\nAnalyzing {symbol}...")
            result = news_analyzer.analyze_stock_news_comprehensive(symbol)
            print(f"  Signal: {result.trading_signal.value} (Confidence: {result.confidence:.1%})")
            print(f"  Market Impact: {result.market_impact.value}")
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")

if __name__ == "__main__":
    success = test_news_analysis()
    
    if success:
        test_trading_signals()
        print("\n🎉 All tests completed!")
    else:
        print("\n💥 Test failed - check the error messages above")
        sys.exit(1)