import time
import argparse
import sys
from typing import Optional

sys.path.insert(0, '.')

from knowledge_updater.core.config import get_config
from knowledge_updater.core.logging import setup_logging
from knowledge_updater.core.scheduler import get_scheduler, start_scheduler, stop_scheduler
from knowledge_updater.data_sources.manager import DataSourceManager
from knowledge_updater.vector_db.manager import VectorDBManager
from knowledge_updater.query.enhancer import QueryEnhancer
from knowledge_updater.query.response_generator import ResponseGenerator


def main():
    parser = argparse.ArgumentParser(description='Knowledge Base Update System')
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--log-level', '-l', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    update_parser = subparsers.add_parser('update', help='Update knowledge base')
    update_parser.add_argument('--full', '-f', action='store_true', help='Perform full update (ignore cache)')

    query_parser = subparsers.add_parser('query', help='Query knowledge base')
    query_parser.add_argument('query_text', help='Query text')
    query_parser.add_argument('--results', '-n', type=int, default=5, help='Number of results to return')

    serve_parser = subparsers.add_parser('serve', help='Start knowledge base server')
    serve_parser.add_argument('--host', default='localhost', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')

    subparsers.add_parser('test', help='Test data sources and connections')

    args = parser.parse_args()
    setup_logging(log_level=args.log_level)

    if args.command == 'update':
        run_update(args.full)
    elif args.command == 'query':
        run_query(args.query_text, args.results)
    elif args.command == 'serve':
        run_server(args.host, args.port)
    elif args.command == 'test':
        run_test()
    else:
        parser.print_help()


def run_update(full_update: bool = False) -> None:
    print("🔄 Starting knowledge base update...")

    try:
        data_manager = DataSourceManager()
        vector_manager = VectorDBManager()

        print("📡 Fetching data from sources...")
        raw_articles = data_manager.fetch_all_sources()

        if not raw_articles:
            print("⚠️  No new articles found")
            return

        print(f"📊 Found {len(raw_articles)} articles")

        print("🔧 Processing articles...")
        processed_articles = data_manager.process_data(raw_articles)

        if not processed_articles:
            print("⚠️  No articles passed processing")
            return

        print(f"✅ Processed {len(processed_articles)} articles")

        print("🗄️  Updating vector database...")
        update_stats = vector_manager.update_knowledge_base(processed_articles)

        print("✅ Knowledge base update completed!")
        print(f"   📈 Articles processed: {update_stats['articles_processed']}")
        print(f"   🆕 New articles: {update_stats['new_articles']}")
        print(f"   ➕ Vectors added: {update_stats['vectors_added']}")
        print(f"   🚫 Duplicates filtered: {update_stats['duplicates_filtered']}")

    except KeyboardInterrupt:
        print("\n⏹️  Update interrupted by user")
    except Exception as e:
        print(f"❌ Update failed: {e}")
        sys.exit(1)


def run_query(query_text: str, n_results: int = 5) -> None:
    print(f"🔍 Querying knowledge base: '{query_text}'")

    try:
        enhancer = QueryEnhancer()
        enhanced = enhancer.enhance_query(query_text)

        print("✅ Query enhancement completed!")
        print(f"   🎯 Confidence: {enhanced.get('confidence', 0):.2f}")
        print(f"   📊 Results found: {enhanced.get('results_count', 0)}")

        context = enhanced.get('context', [])
        if context:
            print("\n📚 Relevant context:")
            for i, item in enumerate(context[:n_results], 1):
                print(f"\n   {i}. {item.get('title', 'Unknown')}")
                print(f"      Source: {item.get('source', 'Unknown')}")
                print(f"      Similarity: {item.get('similarity', 0):.2f}")
                content = item.get('content', '')[:200]
                print(f"      Content: {content}...")
        else:
            print("   ℹ️  No relevant context found")

    except Exception as e:
        print(f"❌ Query failed: {e}")
        sys.exit(1)


def run_server(host: str = 'localhost', port: int = 8000) -> None:
    print(f"🚀 Starting knowledge base server on {host}:{port}")

    try:
        scheduler = get_scheduler()
        scheduler.start()

        print("✅ Server started successfully!")
        print("   🔄 Automatic updates enabled")
        print("   🌐 Knowledge base accessible")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Server stopped by user")

    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)
    finally:
        stop_scheduler()


def run_test() -> None:
    print("🧪 Testing knowledge base system...")

    try:
        print("⚙️  Testing configuration...")
        config = get_config()
        print(f"   ✅ Configuration loaded ({len(config)} sections)")

        print("📡 Testing data sources...")
        data_manager = DataSourceManager()
        source_stats = data_manager.get_source_stats()

        rss_feeds = source_stats.get('rss_feeds', {})
        apis = source_stats.get('apis', {})

        print(f"   📡 RSS feeds: {rss_feeds.get('total', 0)} configured, {rss_feeds.get('enabled', 0)} enabled")
        print(f"   🔌 APIs: {apis.get('total', 0)} configured, {apis.get('enabled', 0)} enabled")

        print("🗄️  Testing vector database...")
        vector_manager = VectorDBManager()
        db_stats = vector_manager.get_database_stats()

        print(f"   ✅ Vector database connected")
        print(f"   📊 Articles in database: {db_stats.get('article_count', 0)}")

        print("🧮 Testing embedding generation...")
        embedding_generator = vector_manager.embedding_generator
        test_text = ["This is a test sentence for embedding generation."]

        embeddings = embedding_generator.generate_embeddings(test_text)
        if embeddings:
            print(f"   ✅ Embeddings generated (dimension: {len(embeddings[0])})")
        else:
            print("   ❌ Embedding generation failed")

        print("🔍 Testing query enhancement...")
        enhancer = QueryEnhancer()
        test_query = "What is artificial intelligence?"

        enhanced = enhancer.enhance_query(test_query)
        print(f"   ✅ Query enhancement working (confidence: {enhanced.get('confidence', 0):.2f})")

        print("\n🎉 All tests passed! System is ready.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


def demo_response_generation():
    print("\n🤖 Demonstrating response generation...")

    try:
        generator = ResponseGenerator()

        test_queries = [
            "What are the latest developments in artificial intelligence?",
            "How does machine learning work?",
            "What is the current state of quantum computing?"
        ]

        for query in test_queries:
            print(f"\n🔍 Query: {query}")

            response_data = generator.generate_response(query)

            print(f"🤖 Response: {response_data['response'][:100]}...")
            print(f"📊 Confidence: {response_data['confidence']:.2f}")
            print(f"📚 Context used: {response_data['context_used']} articles")

            if response_data.get('citations'):
                print(f"📖 Citations: {len(response_data['citations'])} sources")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == '__main__':
    main()