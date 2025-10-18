#!/usr/bin/env python3
"""
Test script to debug RSS feed and API issues
"""

import requests
import feedparser
import json

def test_jsonplaceholder():
    """Test JSONPlaceholder API"""
    print("Testing JSONPlaceholder API...")
    try:
        response = requests.get('https://jsonplaceholder.typicode.com/posts?_limit=3', timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Got {len(data)} posts")
            if data:
                print(f"First post title: {data[0].get('title', 'No title')}")
                return True
    except Exception as e:
        print(f"Error: {e}")
    return False

def test_rss_feeds():
    """Test RSS feeds"""
    feeds = [
        'https://techcrunch.com/feed/',
        'https://hnrss.org/frontpage',
        'http://export.arxiv.org/rss/cs.AI'
    ]

    for url in feeds:
        print(f"\nTesting RSS: {url}")
        try:
            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                print(f"Feed title: {feed.feed.get('title', 'No title')}")
                print(f"Entries: {len(feed.entries)}")

                if feed.entries:
                    entry = feed.entries[0]
                    print(f"First title: {entry.get('title', 'No title')}")
                    print(f"Has content: {bool(getattr(entry, 'content', None) or getattr(entry, 'summary', None) or getattr(entry, 'description', None))}")
                    return True
        except Exception as e:
            print(f"Error: {e}")
    return False

if __name__ == "__main__":
    print("=== Testing Data Sources ===")

    # Test API first
    api_ok = test_jsonplaceholder()

    # Test RSS feeds
    rss_ok = test_rss_feeds()

    print("\n=== Results ===")
    print(f"API working: {api_ok}")
    print(f"RSS working: {rss_ok}")

    if api_ok:
        print("✅ System should work with JSONPlaceholder API")
    if rss_ok:
        print("✅ System should work with RSS feeds")