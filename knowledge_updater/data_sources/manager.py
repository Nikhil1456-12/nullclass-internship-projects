from typing import Dict, List, Any, Optional
import time
from datetime import datetime, timezone

from ..core.config import get_config
from ..core.logging import get_logger, LogContext
from .rss_handler import RSSHandler
from .api_handler import APIHandler
from .data_processor import DataProcessor

logger = get_logger(__name__)


class DataSourceManager:
    def __init__(self):
        self.config = get_config()
        self.rss_handler = RSSHandler()
        self.api_handler = APIHandler()
        self.data_processor = DataProcessor()

    def fetch_all_sources(self) -> List[Dict[str, Any]]:
        start_time = time.time()
        all_articles = []

        try:
            logger.info("Starting data source fetch")

            with LogContext(logger, source_type="RSS"):
                rss_articles = self.rss_handler.fetch_feeds()
                all_articles.extend(rss_articles)
                logger.info("RSS feeds processed", count=len(rss_articles))

            with LogContext(logger, source_type="API"):
                api_articles = self.api_handler.fetch_apis()
                all_articles.extend(api_articles)
                logger.info("APIs processed", count=len(api_articles))

            fetch_time = time.time() - start_time
            logger.info(
                "Data source fetch completed",
                total_articles=len(all_articles),
                fetch_time=f"{fetch_time:.2f}s"
            )

            return all_articles

        except Exception as e:
            fetch_time = time.time() - start_time
            logger.error(
                "Data source fetch failed",
                fetch_time=f"{fetch_time:.2f}s",
                error=str(e)
            )
            raise

    def process_data(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not articles:
            logger.info("No articles to process")
            return []

        start_time = time.time()

        try:
            logger.info("Starting data processing", article_count=len(articles))

            processed_articles = self.data_processor.process_articles(articles)

            for article in processed_articles:
                article['processing_metadata'] = {
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'processor_version': '1.0.0',
                    'pipeline_stage': 'data_ingestion'
                }

            process_time = time.time() - start_time
            logger.info(
                "Data processing completed",
                input_count=len(articles),
                output_count=len(processed_articles),
                process_time=f"{process_time:.2f}s"
            )

            return processed_articles

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Data processing failed",
                process_time=f"{process_time:.2f}s",
                error=str(e)
            )
            raise

    def fetch_and_process(self) -> List[Dict[str, Any]]:
        raw_articles = self.fetch_all_sources()
        processed_articles = self.process_data(raw_articles)
        return processed_articles

    def get_source_stats(self) -> Dict[str, Any]:
        config = get_config()
        rss_feeds = config.get('data_sources', {}).get('rss_feeds', [])
        apis = config.get('data_sources', {}).get('apis', [])

        return {
            'rss_feeds': {
                'total': len(rss_feeds),
                'enabled': sum(1 for feed in rss_feeds if feed.get('enabled', True))
            },
            'apis': {
                'total': len(apis),
                'enabled': sum(1 for api in apis if api.get('enabled', True))
            },
            'processor_cache': self.data_processor.get_cache_stats()
        }

    def reset_caches(self) -> None:
        self.data_processor.reset_deduplication_cache()
        logger.info("All caches reset")

    def test_sources(self) -> Dict[str, Any]:
        results = {
            'rss_feeds': {},
            'apis': {},
            'summary': {
                'total_sources': 0,
                'working_sources': 0,
                'failed_sources': 0
            }
        }

        config = get_config()
        rss_feeds = config.get('data_sources', {}).get('rss_feeds', [])
        apis = config.get('data_sources', {}).get('apis', [])

        for feed in rss_feeds:
            if feed.get('enabled', True):
                results['summary']['total_sources'] += 1
                try:
                    articles = self.rss_handler._fetch_single_feed(feed)
                    results['rss_feeds'][feed.get('name', 'Unknown')] = {
                        'status': 'success',
                        'articles_count': len(articles)
                    }
                    results['summary']['working_sources'] += 1
                except Exception as e:
                    results['rss_feeds'][feed.get('name', 'Unknown')] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    results['summary']['failed_sources'] += 1

        for api in apis:
            if api.get('enabled', True):
                results['summary']['total_sources'] += 1
                try:
                    articles = self.api_handler._fetch_single_api(api)
                    results['apis'][api.get('name', 'Unknown')] = {
                        'status': 'success',
                        'articles_count': len(articles)
                    }
                    results['summary']['working_sources'] += 1
                except Exception as e:
                    results['apis'][api.get('name', 'Unknown')] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    results['summary']['failed_sources'] += 1

        return results