#!/usr/bin/env python3
"""
Domain Expert Chatbot - Main Entry Point

This main.py file provides a unified entry point for the Domain Expert Chatbot.
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
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        print("✓ NLTK imports successful")

        # Local imports
        from domain_expert_chatbot import DomainExpertChatbot, ArXivDatasetLoader
        print("✓ Domain expert chatbot imports successful")

        # Knowledge updater imports
        from knowledge_updater.core.config import get_config
        from knowledge_updater.core.logging import get_logger
        from knowledge_updater.vector_db.manager import VectorDBManager
        from knowledge_updater.embeddings.generator import EmbeddingGenerator
        print("✓ Knowledge updater imports successful")

        print("✅ All imports verified")
        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def initialize_chatbot():
    """Initialize the domain expert chatbot"""
    try:
        print("🚀 Initializing Domain Expert Chatbot...")

        from domain_expert_chatbot import DomainExpertChatbot

        # Initialize chatbot
        start_time = time.time()
        chatbot = DomainExpertChatbot()
        init_time = time.time() - start_time

        print(f"✅ Chatbot initialized in {init_time:.3f}")
        print(f"   - Papers loaded: {len(chatbot.dataset_loader.processed_data)}")
        print(f"   - Domain: {chatbot.domain}")
        print(f"   - Confidence threshold: {chatbot.confidence_threshold}")

        return chatbot

    except Exception as e:
        print(f"❌ Chatbot initialization failed: {e}")
        return None

def run_tests(chatbot):
    """Run comprehensive tests"""
    try:
        print("\\n🧪 Running comprehensive tests...")

        test_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What are the latest trends in AI?"
        ]

        all_passed = True

        for i, query in enumerate(test_queries, 1):
            try:
                start_time = time.time()
                result = chatbot.chat(query)
                processing_time = time.time() - start_time

                confidence = result.get('confidence', 0)
                papers_found = result.get('papers_found', 0)

                print(f"  Test {i}: ✅ PASSED")
                print(f"    Query: {query}")
                print(f"    Processing time: {processing_time:.3f}")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Papers found: {papers_found}")

                if processing_time > 5.0:
                    print("    ⚠️  Warning: Response time > 5s (not real-time)")
                    all_passed = False

            except Exception as e:
                print(f"  Test {i}: ❌ FAILED - {e}")
                all_passed = False

        if all_passed:
            print("\\n✅ All tests passed! Chatbot is ready for use.")
        else:
            print("\\n⚠️  Some tests failed. Chatbot may have issues.")

        return all_passed

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def launch_ui(chatbot, port: int = 8501):
    """Launch the Streamlit UI"""
    try:
        print(f"\\n🖥️  Launching Domain Expert Chatbot UI...")
        print(f"   Run this command in terminal: streamlit run domain_expert_chatbot.py --server.port={port}")
        print(f"   Then access at: http://localhost:{port}")
        print("   Press Ctrl+C to stop")

        # Import and run the UI directly
        from domain_expert_chatbot import create_domain_expert_ui

        print("\\n🚀 Starting chatbot interface...")
        create_domain_expert_ui()

    except KeyboardInterrupt:
        print("\\n👋 Chatbot UI stopped by user")
    except Exception as e:
        print(f"❌ UI launch failed: {e}")
        print(f"   Try running manually: streamlit run domain_expert_chatbot.py --server.port={port}")

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Domain Expert Chatbot - Computer Science Q&A System",
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
        help='Initialize chatbot only (do not launch UI or run tests)'
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

    print("🤖 DOMAIN EXPERT CHATBOT")
    print("=" * 50)
    print("Computer Science Q&A System with arXiv Research Integration")
    print()

    # Step 1: Download NLTK data
    if not download_nltk_data():
        print("❌ Cannot continue without NLTK data")
        sys.exit(1)

    # Step 2: Test imports
    if not test_imports():
        print("❌ Cannot continue with import issues")
        sys.exit(1)

    # Step 3: Initialize chatbot
    chatbot = initialize_chatbot()
    if not chatbot:
        print("❌ Cannot continue without chatbot initialization")
        sys.exit(1)

    # Step 4: Handle different modes
    if args.skip_ui:
        print("\\n✅ Chatbot initialized successfully!")
        print("   Ready for programmatic use.")
        return chatbot

    elif args.test:
        print()
        run_tests(chatbot)
        return chatbot

    else:
        # Launch UI
        launch_ui(chatbot, args.port)
        return chatbot

if __name__ == "__main__":
    try:
        chatbot = main()
    except KeyboardInterrupt:
        print("\\n👋 Domain Expert Chatbot stopped")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)