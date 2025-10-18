#!/usr/bin/env python3
"""
Comprehensive Test Script for All Internship Projects
Tests each project individually in isolated environments
"""

import sys
import os
import importlib
import traceback
from typing import Dict, Any

def test_project_isolation(project_name: str, module_path: str, test_func) -> Dict[str, Any]:
    """Test a project in isolation"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING PROJECT: {project_name}")
    print(f"{'='*60}")

    start_time = __import__('time').time()
    result = {
        'project': project_name,
        'success': False,
        'error': None,
        'execution_time': 0,
        'details': {}
    }

    try:
        # Clear module cache to ensure clean import
        modules_to_clear = [m for m in sys.modules.keys() if project_name.lower() in m.lower()]
        for module in modules_to_clear:
            del sys.modules[module]

        # Add current directory to path
        if '.' not in sys.path:
            sys.path.insert(0, '.')

        # Execute test function
        test_result = test_func()

        result['success'] = True
        result['details'] = test_result
        result['execution_time'] = __import__('time').time() - start_time

        print(f"✅ {project_name} test completed successfully in {result['execution_time']:.2f}s")

    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        result['execution_time'] = __import__('time').time() - start_time

        print(f"❌ {project_name} test failed in {result['execution_time']:.2f}s")
        print(f"   Error: {e}")

    return result

def test_knowledge_updater():
    """Test knowledge_updater core system"""
    print("🔧 Testing knowledge_updater core system...")

    import knowledge_updater
    from knowledge_updater.core.config import get_config
    from knowledge_updater.core.logging import get_logger

    config = get_config()
    logger = get_logger('test')

    return {
        'module_version': getattr(knowledge_updater, '__version__', 'unknown'),
        'config_type': type(config).__name__,
        'logger_type': type(logger).__name__
    }

def test_domain_expert_chatbot():
    """Test domain expert chatbot"""
    print("🧠 Testing domain expert chatbot...")

    from internship_projects.domain_expert_chatbot.domain_expert_chatbot import DomainExpertChatbot

    chatbot = DomainExpertChatbot()
    stats = chatbot.get_dataset_stats()

    return {
        'chatbot_class': 'DomainExpertChatbot',
        'dataset_size': stats.get('total_papers', 0),
        'categories_count': len(stats.get('categories', {})),
        'avg_abstract_length': stats.get('avg_abstract_length', 0)
    }

def test_multimodal_chatbot():
    """Test multimodal chatbot"""
    print("🖼️ Testing multimodal chatbot...")

    from internship_projects.multimodal_chatbot.multimodal_chatbot import MultimodalChatbot

    # Note: Requires API key for full functionality
    chatbot = MultimodalChatbot()
    stats = chatbot.get_stats()

    return {
        'chatbot_class': 'MultimodalChatbot',
        'models_available': stats.get('models_available', []),
        'knowledge_base_enabled': stats.get('knowledge_base_enabled', False),
        'max_history_length': stats.get('max_history_length', 0)
    }

def test_medical_qa_chatbot():
    """Test medical Q&A chatbot"""
    print("🏥 Testing medical Q&A chatbot...")

    from internship_projects.medical_qa_chatbot.medical_qa_chatbot import MedicalQABot

    chatbot = MedicalQABot()
    stats = chatbot.get_dataset_stats()

    return {
        'chatbot_class': 'MedicalQABot',
        'dataset_size': stats.get('total_qa_pairs', 0),
        'focus_areas': stats.get('unique_focus_areas', 0),
        'avg_question_length': stats.get('avg_question_length', 0)
    }

def test_multilingual_chatbot():
    """Test multilingual chatbot"""
    print("🌐 Testing multilingual chatbot...")

    from internship_projects.multilingual_support.multilingual_chatbot import MultilingualChatbot

    chatbot = MultilingualChatbot()
    stats = chatbot.get_conversation_stats()
    supported_langs = chatbot.get_supported_languages()

    return {
        'chatbot_class': 'MultilingualChatbot',
        'supported_languages_count': len(supported_langs),
        'primary_language': chatbot.primary_language,
        'auto_translate': chatbot.auto_translate
    }

def test_sentiment_analyzer():
    """Test sentiment analyzer"""
    print("😊 Testing sentiment analyzer...")

    from internship_projects.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_sentiment("I am very happy with this excellent service!")

    return {
        'analyzer_class': 'SentimentAnalyzer',
        'sentiment': result.get('overall_sentiment', 'unknown'),
        'confidence': result.get('confidence', 0),
        'dominant_emotion': result.get('dominant_emotion', 'neutral'),
        'processing_time': result.get('processing_time', 0)
    }

def main():
    """Run all project tests"""
    print("🚀 Starting Comprehensive Project Testing Suite")
    print("=" * 60)

    # Define test projects
    test_projects = [
        ("Knowledge Updater", "knowledge_updater", test_knowledge_updater),
        ("Domain Expert Chatbot", "domain_expert_chatbot", test_domain_expert_chatbot),
        ("Multimodal Chatbot", "multimodal_chatbot", test_multimodal_chatbot),
        ("Medical Q&A Chatbot", "medical_qa_chatbot", test_medical_qa_chatbot),
        ("Multilingual Support", "multilingual_support", test_multilingual_chatbot),
        ("Sentiment Analysis", "sentiment_analysis", test_sentiment_analyzer),
    ]

    results = []

    for project_name, module_name, test_func in test_projects:
        result = test_project_isolation(project_name, module_name, test_func)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")

    successful = 0
    failed = 0

    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{result['project']:25} {status} ({result['execution_time']:.2f}s)")

        if not result['success']:
            failed += 1
            print(f"   Error: {result['error']}")
        else:
            successful += 1

    print(f"\n🎯 FINAL RESULTS: {successful} passed, {failed} failed")

    if failed == 0:
        print("🎉 ALL PROJECTS TESTED SUCCESSFULLY!")
        return 0
    else:
        print("⚠️ SOME PROJECTS FAILED - CHECK ERRORS ABOVE")
        return 1

if __name__ == "__main__":
    sys.exit(main())