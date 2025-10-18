"""
RSS feed handler for data ingestion
"""

import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import feedparser
import requests
from bs4 import BeautifulSoup
import re

from ..core.config import get_config
from ..core.logging import get_logger, LogContext
from ..utils.rate_limiter import rate_limit

logger = get_logger(__name__)


class RSSHandler:
    """Handles RSS feed data ingestion"""

    def __init__(self):
        """Initialize RSS handler"""
        self.config = get_config()
        self.rss_config = self.config.get('data_sources', {}).get('rss_feeds', [])
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Knowledge-Updater-Bot/1.0'
        })

    def fetch_feeds(self) -> List[Dict[str, Any]]:
        """
        Fetch data from all configured RSS feeds

        Returns:
            List of articles from all feeds
        """
        all_articles = []

        for feed_config in self.rss_config:
            if not feed_config.get('enabled', True):
                logger.debug("Skipping disabled RSS feed", feed=feed_config.get('name'))
                continue

            try:
                with LogContext(logger, feed_name=feed_config.get('name')):
                    articles = self._fetch_single_feed(feed_config)
                    if articles:
                        all_articles.extend(articles)
                        logger.info(
                            "RSS feed processed successfully",
                            feed_name=feed_config.get('name'),
                            articles_fetched=len(articles)
                        )

            except Exception as e:
                logger.error(
                    "Failed to process RSS feed",
                    feed_name=feed_config.get('name'),
                    error=str(e)
                )

        return all_articles

    @rate_limit(requests_per_minute=30)
    def _fetch_single_feed(self, feed_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch articles from a single RSS feed

        Args:
            feed_config: Configuration for the RSS feed

        Returns:
            List of articles from the feed
        """
        feed_url = feed_config.get('url')
        if not feed_url:
            raise ValueError("RSS feed URL not configured")

        try:
            # Parse RSS feed
            logger.debug("Fetching RSS feed", url=feed_url)
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning("RSS feed parsing had issues", url=feed_url, error=feed.bozo_exception)
                if hasattr(feed, 'bozo_exception') and 'not well-formed' in str(feed.bozo_exception):
                    # Try to handle malformed feeds
                    return self._handle_malformed_feed(feed_url)

            articles = []
            max_articles = feed_config.get('max_articles', 50)

            for entry in feed.entries[:max_articles]:
                try:
                    article = self._parse_article(entry, feed_config)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(
                        "Failed to parse article",
                        title=getattr(entry, 'title', 'Unknown'),
                        error=str(e)
                    )

            return articles

        except Exception as e:
            logger.error("Failed to fetch RSS feed", url=feed_url, error=str(e))
            raise

    def _parse_article(self, entry, feed_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a single RSS entry into a standardized article format

        Args:
            entry: RSS feed entry
            feed_config: Feed configuration

        Returns:
            Parsed article or None if invalid
        """
        # Extract basic fields
        title = getattr(entry, 'title', '')
        if isinstance(title, str):
            title = title.strip()
        else:
            title = str(title).strip() if title else ''

        if not title:
            return None

        # Get content
        content = self._extract_content(entry)

        # Get publication date
        published = self._parse_date(entry)

        # Generate unique ID
        content_hash = hashlib.md5((title + content[:500]).encode('utf-8')).hexdigest()
        article_id = f"{feed_config.get('name', 'unknown')}_{content_hash}"

        # Extract metadata
        metadata = {
            'source': feed_config.get('name', 'RSS Feed'),
            'url': str(getattr(entry, 'link', '')),
            'feed_url': feed_config.get('url'),
            'author': str(getattr(entry, 'author', '')),
            'tags': getattr(entry, 'tags', []),
            'categories': getattr(entry, 'categories', [])
        }

        article = {
            'id': article_id,
            'title': title,
            'content': content,
            'published_at': published,
            'metadata': metadata,
            'content_hash': content_hash,
            'word_count': len(content.split()) if content else 0
        }

        # Validate article quality
        if self._validate_article(article):
            return article

        return None

    def _extract_content(self, entry) -> str:
        """Extract content from RSS entry"""
        content = ""

        # Try different content fields in order of preference
        content_fields = [
            'content', 'summary', 'description', 'content_encoded',
            'summary_detail', 'description_detail'
        ]

        for field in content_fields:
            if hasattr(entry, field):
                field_value = getattr(entry, field)
                if isinstance(field_value, str):
                    content = field_value.strip()
                elif isinstance(field_value, list) and field_value:
                    content = str(field_value[0]).strip()
                elif hasattr(field_value, 'value'):
                    content = str(field_value.value).strip()
                elif field_value:
                    content = str(field_value).strip()

                if content:
                    break

        # Clean HTML content
        if content:
            content = self._clean_html(content)

        return content

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract text"""
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[\r\n\t]', ' ', text)

            return text.strip()

        except Exception:
            # Fallback to regex cleaning if BeautifulSoup fails
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse publication date from RSS entry"""
        date_fields = ['published', 'pubDate', 'updated', 'created', 'modified']

        for field in date_fields:
            if hasattr(entry, field):
                date_value = getattr(entry, field)
                if date_value:
                    try:
                        # feedparser usually converts to time.struct_time
                        if hasattr(date_value, 'tm_year'):
                            return datetime(*date_value[:6], tzinfo=timezone.utc)
                        # Try parsing as string
                        return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        continue

        # Default to current time if no date found
        return datetime.now(timezone.utc)

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
            logger.debug(
                "Article failed content length validation",
                title=article.get('title', 'Unknown'),
                content_length=len(content),
                word_count=word_count
            )
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
                logger.debug(
                    "Article filtered due to spam content",
                    title=article.get('title', 'Unknown'),
                    keyword=keyword
                )
                return False

        return True

    def _handle_malformed_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """
        Handle malformed RSS feeds by attempting direct parsing

        Args:
            feed_url: URL of the malformed feed

        Returns:
            List of articles if parsing succeeds, empty list otherwise
        """
        try:
            logger.debug("Attempting to handle malformed RSS feed", url=feed_url)

            response = requests.get(feed_url, timeout=30)
            response.raise_for_status()

            # Try to extract basic information from the response
            soup = BeautifulSoup(response.content, 'xml')

            # Look for common RSS elements
            items = soup.find_all(['item', 'entry'])

            articles = []
            for item in items[:10]:  # Limit to first 10 items
                title_elem = item.find(['title', 'dc:title'])
                desc_elem = item.find(['description', 'content', 'summary'])

                if title_elem:
                    title = title_elem.get_text().strip()
                    content = desc_elem.get_text().strip() if desc_elem else ""

                    if title and content:
                        article = {
                            'id': f"malformed_{hashlib.md5((title + content).encode()).hexdigest()}",
                            'title': title,
                            'content': content,
                            'published_at': datetime.now(timezone.utc),
                            'metadata': {
                                'source': 'Malformed RSS Feed',
                                'url': feed_url,
                                'note': 'Parsed from malformed feed'
                            },
                            'content_hash': hashlib.md5((title + content).encode()).hexdigest(),
                            'word_count': len(content.split())
                        }

                        if self._validate_article(article):
                            articles.append(article)

            return articles

        except Exception as e:
            logger.error("Failed to handle malformed feed", url=feed_url, error=str(e))
            return []