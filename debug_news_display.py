#!/usr/bin/env python3
"""
Debug script to test specifically what's happening with news display in UI
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.news_service import NewsService

def debug_news_display():
    """Debug news display issue"""
    print("=== DEBUGGING NEWS DISPLAY ISSUE ===")
    
    news_service = NewsService()
    test_symbol = "AAPL"
    
    print(f"Testing news fetching for UI display for {test_symbol}...")
    
    # Test the exact same call as the UI makes
    articles = news_service.get_stock_news(test_symbol, days_back=7, max_articles=10)
    
    print(f"Total articles returned: {len(articles)}")
    
    if not articles:
        print("❌ NO ARTICLES RETURNED - This is the problem!")
        return False
    
    print("\n=== DETAILED ARTICLE INSPECTION ===")
    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}:")
        print(f"  Title: '{article.title}'")
        print(f"  Title length: {len(article.title)}")
        print(f"  Title type: {type(article.title)}")
        print(f"  Publisher: '{article.publisher}'")
        print(f"  Publisher type: {type(article.publisher)}")
        print(f"  Summary: '{article.summary[:100]}...'")
        print(f"  Summary length: {len(article.summary)}")
        print(f"  Published date: {article.published_date}")
        print(f"  Published date type: {type(article.published_date)}")
        print(f"  Link: '{article.link}'")
        print(f"  Sentiment: {article.sentiment}")
        
        # Test the exact publisher parsing logic from UI
        source_key = article.publisher.split(' - ')[0] if ' - ' in article.publisher else article.publisher
        print(f"  Source key for UI: '{source_key}'")
        
        # Test the date formatting that the UI uses
        try:
            formatted_date = article.published_date.strftime('%Y-%m-%d %H:%M')
            print(f"  Formatted date: '{formatted_date}'")
        except Exception as e:
            print(f"  ❌ Error formatting date: {e}")
        
        # Test expander title construction
        try:
            source_colors = {
                'Yahoo Finance': '🟠',
                'Google News': '🔵', 
                'Finviz': '🟢',
                'MarketWatch': '🟡',
                'Investing.com': '🟣'
            }
            source_emoji = source_colors.get(source_key, '⚪')
            expander_title = f"{source_emoji} {i}. {article.title} ({article.published_date.strftime('%Y-%m-%d %H:%M')})"
            print(f"  Expander title: '{expander_title}'")
            print(f"  Expander title length: {len(expander_title)}")
        except Exception as e:
            print(f"  ❌ Error creating expander title: {e}")
    
    print("\n=== SOURCE DISTRIBUTION ===")
    source_counts = {}
    for article in articles:
        source = article.publisher.split(' - ')[0] if ' - ' in article.publisher else article.publisher
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in source_counts.items():
        print(f"  {source}: {count} articles")
    
    print(f"\n✅ Articles look good for display. If UI still shows no content, the issue is in Streamlit rendering.")
    return True

if __name__ == "__main__":
    debug_news_display()