#!/usr/bin/env python3
"""
Medical QA Chatbot - Main Entry Point

This main.py file provides a unified entry point for the Medical QA Chatbot.
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
        from medical_qa_chatbot import MedicalQABot, MedQuADDatasetLoader, MedicalEntityRecognizer
        print("✓ Medical QA chatbot imports successful")

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

def initialize_medical_bot():
    """Initialize the medical QA chatbot"""
    try:
        print("🚀 Initializing Medical QA Chatbot...")

        from medical_qa_chatbot import MedicalQABot

        # Initialize chatbot
        start_time = time.time()
        bot = MedicalQABot()
        init_time = time.time() - start_time

        print(f"✅ Medical QA bot initialized in {init_time:.3f}s")
        print(f"   - Q&A pairs loaded: {len(bot.dataset_loader.processed_data)}")
        print(f"   - Confidence threshold: {bot.confidence_threshold}")
        print(f"   - Medical disclaimer: {'Enabled' if bot.enable_medical_disclaimer else 'Disabled'}")

        return bot

    except Exception as e:
        print(f"❌ Medical bot initialization failed: {e}")
        return None

def run_tests(bot):
    """Run comprehensive tests"""
    try:
        print("\\n🧪 Running comprehensive tests...")

        test_questions = [
            "What are symptoms of diabetes?",
            "How to manage high blood pressure?",
            "What are side effects of aspirin?",
            "What is hypertension?",
            "How to treat a cold?"
        ]

        all_passed = True

        for i, question in enumerate(test_questions, 1):
            try:
                start_time = time.time()
                result = bot.ask_question(question)
                processing_time = time.time() - start_time

                confidence = result.get('confidence', 0)
                entities_found = result.get('entities_found', {})
                search_method = result.get('search_method', 'unknown')

                print(f"  Test {i}: ✅ PASSED")
                print(f"    Question: {question}")
                print(f"    Search method: {search_method}")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Entities detected: {list(entities_found.keys())}")
                print(f"    Processing time: {processing_time:.3f}s")

                if processing_time > 5.0:
                    print("    ⚠️  Warning: Response time > 5s (not real-time)")
                    all_passed = False

            except Exception as e:
                print(f"  Test {i}: ❌ FAILED - {e}")
                all_passed = False

        if all_passed:
            print("\\n✅ All tests passed! Medical QA bot is ready for use.")
        else:
            print("\\n⚠️  Some tests failed. Bot may have issues.")

        return all_passed

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def launch_ui(bot, port: int = 8501):
    """Launch the Streamlit UI"""
    try:
        print(f"\\n🖥️  Launching Medical QA Chatbot UI...")
        print(f"   Run this command in terminal: streamlit run medical_qa_chatbot.py --server.port={port}")
        print(f"   Then access at: http://localhost:{port}")
        print("   Press Ctrl+C to stop")

        # Import and run the UI directly
        from medical_qa_chatbot import create_medical_qa_ui

        print("\\n🚀 Starting medical QA interface...")
        create_medical_qa_ui()

    except KeyboardInterrupt:
        print("\\n👋 Medical QA Chatbot UI stopped by user")
    except Exception as e:
        print(f"❌ UI launch failed: {e}")
        print(f"   Try running manually: streamlit run medical_qa_chatbot.py --server.port={port}")

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Medical QA Chatbot - Health Information System",
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
        help='Initialize bot only (do not launch UI or run tests)'
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

    print("🩺 MEDICAL QA CHATBOT")
    print("=" * 50)
    print("Health Information System with MedQuAD Dataset Integration")
    print()

    # Step 1: Download NLTK data
    if not download_nltk_data():
        print("❌ Cannot continue without NLTK data")
        sys.exit(1)

    # Step 2: Test imports
    if not test_imports():
        print("❌ Cannot continue with import issues")
        sys.exit(1)

    # Step 3: Initialize medical bot
    bot = initialize_medical_bot()
    if not bot:
        print("❌ Cannot continue without bot initialization")
        sys.exit(1)

    # Step 4: Handle different modes
    if args.skip_ui:
        print("\\n✅ Medical QA bot initialized successfully!")
        print("   Ready for programmatic use.")
        return bot

    elif args.test:
        print()
        run_tests(bot)
        return bot

    else:
        # Launch UI
        launch_ui(bot, args.port)
        return bot

if __name__ == "__main__":
    try:
        bot = main()
    except KeyboardInterrupt:
        print("\\n👋 Medical QA Chatbot stopped")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)