#!/usr/bin/env python3
"""
Sentiment Analysis Chatbot - Main Entry Point

This main.py file provides a unified entry point for the Sentiment Analysis system.
It handles NLTK data downloads, imports, initialization, testing, and UI launch.

Usage:
    python main.py                    # Launch interactive UI
    python main.py --test            # Run tests only
    python main.py --skip-ui         # Initialize only (no UI)
    python main.py --port 8502       # Launch on custom port
"""

import sys
import os
import argparse
import time
from typing import Optional

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

def download_nltk_data():
    """Download and verify NLTK data"""
    try:
        import nltk

        # Check if punkt tokenizer data exists
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ NLTK punkt tokenizer data already available")
        except LookupError:
            print("📥 Downloading NLTK punkt tokenizer data...")
            nltk.download('punkt', quiet=True)

        # Check if stopwords corpus exists
        try:
            nltk.data.find('corpora/stopwords')
            print("✓ NLTK stopwords corpus already available")
        except LookupError:
            print("📥 Downloading NLTK stopwords corpus...")
            nltk.download('stopwords', quiet=True)

        # Check if wordnet corpus exists
        try:
            nltk.data.find('corpora/wordnet')
            print("✓ NLTK wordnet corpus already available")
        except LookupError:
            print("📥 Downloading NLTK wordnet corpus...")
            nltk.download('wordnet', quiet=True)

        print("✅ NLTK data ready")
        return True

    except Exception as e:
        print(f"❌ NLTK data download failed: {e}")
        return False

def test_imports():
    """Test all required imports"""
    try:
        print("🔍 Testing imports...")

        # Core imports
        import streamlit as st
        print("✓ Streamlit imported")

        # ML and NLP libraries
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        print("✓ NLTK imports successful")

        # Local imports
        from sentiment_analyzer import SentimentAnalyzer, SentimentAwareResponseGenerator
        print("✓ Sentiment analyzer imports successful")

        # Knowledge updater imports
        from knowledge_updater.core.config import get_config
        from knowledge_updater.core.logging import get_logger
        print("✓ Knowledge updater imports successful")

        print("✅ All imports verified")
        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def initialize_sentiment_analyzer():
    """Initialize the sentiment analysis system"""
    try:
        print("🚀 Initializing Sentiment Analysis System...")

        from sentiment_analyzer import SentimentAnalyzer

        # Initialize sentiment analyzer
        start_time = time.time()
        analyzer = SentimentAnalyzer()
        init_time = time.time() - start_time

        print(f"✅ Sentiment analyzer initialized in {init_time:.3f}s")
        print(f"   - Positive words: {len(analyzer.positive_words)}")
        print(f"   - Negative words: {len(analyzer.negative_words)}")
        print(f"   - Emotion categories: {len(analyzer.emotion_keywords)}")

        return analyzer

    except Exception as e:
        print(f"❌ Sentiment analyzer initialization failed: {e}")
        return None

def run_tests(analyzer):
    """Run comprehensive tests"""
    try:
        print("\\n🧪 Running comprehensive tests...")

        test_texts = [
            "I love this new AI assistant!",
            "This is terrible and frustrating.",
            "This seems okay, nothing special.",
            "I'm absolutely thrilled with the results!",
            "This is quite disappointing and annoying."
        ]

        all_passed = True

        for i, text in enumerate(test_texts, 1):
            try:
                start_time = time.time()
                result = analyzer.analyze_sentiment(text)
                processing_time = time.time() - start_time

                sentiment = result.get('overall_sentiment', 'unknown')
                confidence = result.get('confidence', 0)

                print(f"  Test {i}: ✅ PASSED")
                print(f"    Text: {text}")
                print(f"    Sentiment: {sentiment}")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Processing time: {processing_time:.3f}s")

                if processing_time > 5.0:
                    print("    ⚠️  Warning: Response time > 5s (not real-time)")
                    all_passed = False

            except Exception as e:
                print(f"  Test {i}: ❌ FAILED - {e}")
                all_passed = False

        if all_passed:
            print("\\n✅ All tests passed! Sentiment analysis is ready for use.")
        else:
            print("\\n⚠️  Some tests failed. System may have issues.")

        return all_passed

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def launch_ui(analyzer, port: int = 8501):
    """Launch the Streamlit UI"""
    try:
        print(f"\\n🖥️  Launching Sentiment Analysis UI...")
        print(f"   Run this command in terminal: streamlit run sentiment_analyzer.py --server.port={port}")
        print(f"   Then access at: http://localhost:{port}")
        print("   Press Ctrl+C to stop")

        # Import and run the UI directly
        from sentiment_analyzer import create_sentiment_analysis_ui

        print("\\n🚀 Starting sentiment analysis interface...")
        create_sentiment_analysis_ui()

    except KeyboardInterrupt:
        print("\\n👋 Sentiment Analysis UI stopped by user")
    except Exception as e:
        print(f"❌ UI launch failed: {e}")
        print(f"   Try running manually: streamlit run sentiment_analyzer.py --server.port={port}")

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis System - Real-time Emotion Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Launch interactive UI
  python main.py --test            # Run tests only
  python main.py --skip-ui         # Initialize only (no UI)
  python main.py --port 8502       # Launch on custom port
  python main.py --help            # Show this help message
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests only (do not launch UI)'
    )

    parser.add_argument(
        '--skip-ui',
        action='store_true',
        help='Initialize analyzer only (do not launch UI or run tests)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for Streamlit UI (default: 8501)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    print("😊 SENTIMENT ANALYSIS SYSTEM")
    print("=" * 50)
    print("Real-time Emotion Detection and Sentiment Analysis")
    print()

    # Step 1: Download NLTK data
    if not download_nltk_data():
        print("❌ Cannot continue without NLTK data")
        sys.exit(1)

    # Step 2: Test imports
    if not test_imports():
        print("❌ Cannot continue with import issues")
        sys.exit(1)

    # Step 3: Initialize sentiment analyzer
    analyzer = initialize_sentiment_analyzer()
    if not analyzer:
        print("❌ Cannot continue without analyzer initialization")
        sys.exit(1)

    # Step 4: Handle different modes
    if args.skip_ui:
        print("\\n✅ Sentiment analyzer initialized successfully!")
        print("   Ready for programmatic use.")
        return analyzer

    elif args.test:
        print()
        run_tests(analyzer)
        return analyzer

    else:
        # Launch UI
        launch_ui(analyzer, args.port)
        return analyzer

if __name__ == "__main__":
    try:
        analyzer = main()
    except KeyboardInterrupt:
        print("\\n👋 Sentiment Analysis System stopped")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)