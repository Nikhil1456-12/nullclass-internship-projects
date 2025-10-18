"""
API handler for data ingestion from various web APIs
"""

import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import requests
import json

from ..core.config import get_config
from ..core.logging import get_logger, LogContext
from ..utils.rate_limiter import rate_limit

logger = get_logger(__name__)


class APIHandler:
    """Handles API-based data ingestion"""

    def __init__(self):
        """Initialize API handler"""
        self.config = get_config()
        self.api_config = self.config.get('data_sources', {}).get('apis', [])
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Knowledge-Updater-Bot/1.0',
            'Accept': 'application/json'
        })

    def fetch_apis(self) -> List[Dict[str, Any]]:
        """
        Fetch data from all configured APIs

        Returns:
            List of articles from all APIs
        """
        all_articles = []

        for api_config in self.api_config:
            if not api_config.get('enabled', True):
                logger.debug("Skipping disabled API", api=api_config.get('name'))
                continue

            try:
                with LogContext(logger, api_name=api_config.get('name')):
                    articles = self._fetch_single_api(api_config)
                    if articles:
                        all_articles.extend(articles)
                        logger.info(
                            "API processed successfully",
                            api_name=api_config.get('name'),
                            articles_fetched=len(articles)
                        )

            except Exception as e:
                logger.error(
                    "Failed to process API",
                    api_name=api_config.get('name'),
                    error=str(e)
                )

        return all_articles

    @rate_limit(requests_per_minute=60)
    def _fetch_single_api(self, api_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch data from a single API

        Args:
            api_config: Configuration for the API

        Returns:
            List of articles from the API
        """
        base_url = api_config.get('base_url')
        if not base_url:
            raise ValueError("API base URL not configured")

        api_key = self._get_api_key(api_config)
        endpoints = api_config.get('endpoints', [])

        all_articles = []

        for endpoint in endpoints:
            try:
                articles = self._fetch_endpoint(base_url, endpoint, api_key, api_config)
                if articles:
                    all_articles.extend(articles)
            except Exception as e:
                logger.error(
                    "Failed to fetch endpoint",
                    endpoint=endpoint.get('path'),
                    error=str(e)
                )

        return all_articles

    def _get_api_key(self, api_config: Dict[str, Any]) -> Optional[str]:
        """Get API key from configuration or environment"""
        api_key_env = api_config.get('api_key_env')
        if api_key_env:
            import os
            return os.getenv(api_key_env)

        api_key = api_config.get('api_key')
        if api_key:
            return api_key

        return None

    def _fetch_endpoint(
        self,
        base_url: str,
        endpoint: Dict[str, Any],
        api_key: Optional[str],
        api_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fetch data from a specific API endpoint"""
        path = endpoint.get('path')
        if not path:
            return []

        url = f"{base_url.rstrip('/')}{path}"

        # Prepare request parameters
        params = endpoint.get('params', {}).copy()
        headers = endpoint.get('headers', {}).copy()

        # Add API key if required
        if api_key:
            api_key_param = endpoint.get('api_key_param', 'api_key')
            if endpoint.get('api_key_in_header'):
                headers[api_key_param] = api_key
            else:
                params[api_key_param] = api_key

        try:
            logger.debug("Fetching API endpoint", url=url, params=params)

            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Parse API response based on known formats
            articles = self._parse_api_response(data, api_config, endpoint)

            logger.debug(
                "API endpoint processed",
                url=url,
                articles_parsed=len(articles)
            )

            return articles

        except requests.exceptions.RequestException as e:
            logger.error("API request failed", url=url, error=str(e))
            raise
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON response", url=url, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error fetching endpoint", url=url, error=str(e))
            raise

    def _parse_api_response(
        self,
        data: Dict[str, Any],
        api_config: Dict[str, Any],
        endpoint: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse API response into standardized article format"""
        articles = []

        # Handle different API response formats
        api_name = api_config.get('name', 'Unknown API')

        if api_name == "NewsAPI":
            articles = self._parse_newsapi_response(data, api_config)
        elif api_name == "Wikipedia Recent Changes":
            articles = self._parse_wikipedia_response(data, api_config)
        elif api_name == "JSONPlaceholder Demo":
            articles = self._parse_jsonplaceholder_response(data, api_config)
        else:
            # Generic parsing for unknown APIs
            articles = self._parse_generic_response(data, api_config)

        return articles

    def _parse_newsapi_response(self, data: Dict[str, Any], api_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse NewsAPI response format"""
        articles = []

        for item in data.get('articles', []):
            try:
                # Extract content
                title = item.get('title', '').strip()
                description = item.get('description', '').strip()
                content = item.get('content', '').strip()

                # Combine description and content
                full_content = f"{description} {content}".strip()
                if not full_content:
                    continue

                # Parse publication date
                published_str = item.get('publishedAt', '')
                try:
                    published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    published = datetime.now(timezone.utc)

                # Generate unique ID
                content_hash = hashlib.md5((title + full_content).encode('utf-8')).hexdigest()
                article_id = f"newsapi_{content_hash}"

                # Extract metadata
                metadata = {
                    'source': item.get('source', {}).get('name', 'NewsAPI'),
                    'url': item.get('url', ''),
                    'author': item.get('author', ''),
                    'image_url': item.get('urlToImage', ''),
                    'api_source': 'NewsAPI'
                }

                article = {
                    'id': article_id,
                    'title': title,
                    'content': full_content,
                    'published_at': published,
                    'metadata': metadata,
                    'content_hash': content_hash,
                    'word_count': len(full_content.split())
                }

                if self._validate_article(article):
                    articles.append(article)

            except Exception as e:
                logger.warning("Failed to parse NewsAPI article", error=str(e))

        return articles

    def _parse_wikipedia_response(self, data: Dict[str, Any], api_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Wikipedia API response format"""
        articles = []

        for item in data.get('query', {}).get('recentchanges', []):
            try:
                title = item.get('title', '').strip()
                if not title:
                    continue

                # Get page content (this would need additional API call in real implementation)
                content = f"Wikipedia page: {title}"
                if item.get('comment'):
                    content += f" - {item.get('comment')}"

                # Parse timestamp
                timestamp_str = item.get('timestamp', '')
                try:
                    # Wikipedia uses ISO format
                    published = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    published = datetime.now(timezone.utc)

                # Generate unique ID
                content_hash = hashlib.md5((title + content).encode('utf-8')).hexdigest()
                article_id = f"wikipedia_{content_hash}"

                # Extract metadata
                metadata = {
                    'source': 'Wikipedia',
                    'page_id': item.get('pageid', ''),
                    'rev_id': item.get('revid', ''),
                    'user': item.get('user', ''),
                    'type': item.get('type', ''),
                    'api_source': 'Wikipedia'
                }

                article = {
                    'id': article_id,
                    'title': title,
                    'content': content,
                    'published_at': published,
                    'metadata': metadata,
                    'content_hash': content_hash,
                    'word_count': len(content.split())
                }

                if self._validate_article(article):
                    articles.append(article)

            except Exception as e:
                logger.warning("Failed to parse Wikipedia article", error=str(e))

        return articles

    def _parse_jsonplaceholder_response(self, data: Dict[str, Any], api_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse JSONPlaceholder API response format"""
        articles = []

        for item in data:
            try:
                title = item.get('title', '').strip()
                body = item.get('body', '').strip()

                if not title or not body:
                    continue

                content = f"{title}\n\n{body}"

                # Parse user ID as a simple timestamp approximation
                user_id = item.get('userId', 1)
                # Create a fake but consistent timestamp based on ID
                import time
                fake_timestamp = datetime.now(timezone.utc).timestamp() - (user_id * 3600)

                # Generate unique ID
                content_hash = hashlib.md5((title + content).encode('utf-8')).hexdigest()
                article_id = f"jsonplaceholder_{content_hash}"

                # Extract metadata
                metadata = {
                    'source': 'JSONPlaceholder Demo',
                    'post_id': item.get('id', ''),
                    'user_id': user_id,
                    'api_source': 'JSONPlaceholder'
                }

                article = {
                    'id': article_id,
                    'title': title,
                    'content': content,
                    'published_at': datetime.fromtimestamp(fake_timestamp, timezone.utc),
                    'metadata': metadata,
                    'content_hash': content_hash,
                    'word_count': len(content.split())
                }

                if self._validate_article(article):
                    articles.append(article)

            except Exception as e:
                logger.warning("Failed to parse JSONPlaceholder item", error=str(e))

        return articles

    def _parse_generic_response(self, data: Dict[str, Any], api_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse generic API response format"""
        articles = []

        # Try to find articles in common response structures
        items = []

        # Check common response structures
        if 'articles' in data:
            items = data['articles']
        elif 'items' in data:
            items = data['items']
        elif 'results' in data:
            items = data['results']
        elif 'data' in data:
            items = data['data']
        elif isinstance(data, list):
            items = data
        else:
            logger.warning("Unknown API response structure", api=api_config.get('name'))
            return []

        for item in items:
            try:
                # Try to extract common fields
                title = ""
                content = ""

                # Try different field names for title
                for field in ['title', 'name', 'headline', 'subject']:
                    if field in item and item[field]:
                        title = str(item[field]).strip()
                        break

                # Try different field names for content
                for field in ['content', 'description', 'summary', 'body', 'text']:
                    if field in item and item[field]:
                        content = str(item[field]).strip()
                        break

                if not title or not content:
                    continue

                # Parse date if available
                published = datetime.now(timezone.utc)
                for field in ['published_at', 'created_at', 'timestamp', 'date']:
                    if field in item and item[field]:
                        try:
                            if isinstance(item[field], str):
                                published = datetime.fromisoformat(item[field].replace('Z', '+00:00'))
                            elif isinstance(item[field], int):
                                published = datetime.fromtimestamp(item[field], timezone.utc)
                            break
                        except (ValueError, TypeError):
                            continue

                # Generate unique ID
                content_hash = hashlib.md5((title + content).encode('utf-8')).hexdigest()
                article_id = f"{api_config.get('name', 'generic')}_{content_hash}"

                # Extract metadata
                metadata = {
                    'source': api_config.get('name', 'Generic API'),
                    'api_source': api_config.get('name', 'Generic API')
                }

                # Add any additional fields as metadata
                for key, value in item.items():
                    if key not in ['title', 'content', 'description', 'summary', 'body', 'text']:
                        metadata[key] = value

                article = {
                    'id': article_id,
                    'title': title,
                    'content': content,
                    'published_at': published,
                    'metadata': metadata,
                    'content_hash': content_hash,
                    'word_count': len(content.split())
                }

                if self._validate_article(article):
                    articles.append(article)

            except Exception as e:
                logger.warning("Failed to parse generic API item", error=str(e))

        return articles

    def _validate_article(self, article: Dict[str, Any]) -> bool:
        """Validate article quality and relevance"""
        config = get_config()
        quality_config = config.get('data_quality', {})

        # Check content length
        min_length = quality_config.get('min_content_length', 100)
        max_length = quality_config.get('max_content_length', 10000)

        content = article.get('content', '')
        word_count = article.get('word_count', 0)

        if len(content) < min_length or len(content) > max_length:
            return False

        # Check required fields
        required_fields = quality_config.get('required_fields', ['title', 'content'])
        for field in required_fields:
            if not article.get(field):
                return False

        # Check for spam/advertisement content
        spam_keywords = quality_config.get('content_filters', [])
        content_lower = content.lower()

        for keyword in spam_keywords:
            if keyword.lower() in content_lower:
                return False

        return True