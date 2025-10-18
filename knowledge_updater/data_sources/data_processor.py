"""
Data processor for cleaning and validating ingested content
"""

import re
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Set, Optional
import json

from ..core.config import get_config
from ..core.logging import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """Processes and validates ingested data"""

    def __init__(self):
        """Initialize data processor"""
        self.config = get_config()
        self.seen_hashes: Set[str] = set()

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of articles

        Args:
            articles: Raw articles from data sources

        Returns:
            Processed and validated articles
        """
        processed_articles = []

        for article in articles:
            try:
                processed = self._process_single_article(article)
                if processed:
                    processed_articles.append(processed)
            except Exception as e:
                logger.warning(
                    "Failed to process article",
                    article_id=article.get('id', 'unknown'),
                    error=str(e)
                )

        # Remove duplicates
        unique_articles = self._deduplicate_articles(processed_articles)

        logger.info(
            "Article processing completed",
            input_count=len(articles),
            processed_count=len(processed_articles),
            unique_count=len(unique_articles)
        )

        return unique_articles

    def _process_single_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single article

        Args:
            article: Raw article data

        Returns:
            Processed article or None if invalid
        """
        # Clean content
        cleaned_content = self._clean_content(article.get('content', ''))
        if not cleaned_content:
            return None

        # Update word count after cleaning
        word_count = len(cleaned_content.split())

        # Validate content length after cleaning
        if not self._validate_content_length(cleaned_content, word_count):
            return None

        # Extract entities and keywords
        entities = self._extract_entities(cleaned_content)
        keywords = self._extract_keywords(cleaned_content)

        # Generate summary if not present
        summary = self._generate_summary(cleaned_content, article.get('title', ''))

        # Update article with processed data
        processed_article = article.copy()
        processed_article.update({
            'content': cleaned_content,
            'word_count': word_count,
            'entities': entities,
            'keywords': keywords,
            'summary': summary,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'content_hash': hashlib.md5(cleaned_content.encode('utf-8')).hexdigest()
        })

        return processed_article

    def _clean_content(self, content: str) -> str:
        """Clean and normalize article content"""
        if not content:
            return ""

        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove HTML entities
        content = re.sub(r'&[a-zA-Z]+;', ' ', content)

        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)

        # Remove email addresses
        content = re.sub(r'\S+@\S+', '', content)

        # Remove phone numbers (basic pattern)
        content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', content)

        # Remove extra punctuation
        content = re.sub(r'[^\w\s\-.,!?:;()]+', ' ', content)

        # Fix common OCR errors or typos (basic)
        content = re.sub(r'\bteh\b', 'the', content, flags=re.IGNORECASE)
        content = re.sub(r'\bfo\b', 'of', content, flags=re.IGNORECASE)

        # Remove standalone numbers that are likely not useful
        words = content.split()
        cleaned_words = []
        for word in words:
            # Keep numbers if they're part of meaningful content
            if word.isdigit() and len(word) < 4:
                continue
            cleaned_words.append(word)

        return ' '.join(cleaned_words).strip()

    def _validate_content_length(self, content: str, word_count: int) -> bool:
        """Validate content length requirements"""
        config = get_config()
        quality_config = config.get('data_quality', {})

        min_length = quality_config.get('min_content_length', 100)
        max_length = quality_config.get('max_content_length', 10000)
        min_words = quality_config.get('min_word_count', 50)
        max_words = quality_config.get('max_word_count', 5000)

        return (
            len(content) >= min_length and
            len(content) <= max_length and
            word_count >= min_words and
            word_count <= max_words
        )

    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content (simplified)"""
        entities = []

        # Simple pattern for potential entities (capitalized words)
        words = content.split()
        for i, word in enumerate(words):
            if (word and word[0].isupper() and len(word) > 2 and
                not word.endswith('.') and not word.endswith(',')):

                # Check if it's likely a proper noun (not at start of sentence if short)
                if len(word) > 3 or (i > 0 and words[i-1].endswith('.')):
                    entities.append(word)

        # Remove duplicates and limit count
        return list(set(entities))[:20]

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content (simplified)"""
        # Remove stop words (basic list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }

        words = re.findall(r'\b\w+\b', content.lower())
        word_counts = {}

        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Return top keywords by frequency
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in keywords[:15]]

    def _generate_summary(self, content: str, title: str) -> str:
        """Generate a simple summary from content"""
        sentences = re.split(r'[.!?]+', content)

        # Filter out very short sentences
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not meaningful_sentences:
            return ""

        # Use first sentence as summary, or combine first two if short
        summary = meaningful_sentences[0]
        if len(summary) < 100 and len(meaningful_sentences) > 1:
            summary += ". " + meaningful_sentences[1]

        return summary[:200] + "..." if len(summary) > 200 else summary

    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on content hash"""
        unique_articles = []
        seen_hashes = set()

        for article in articles:
            content_hash = article.get('content_hash', '')

            if content_hash and content_hash not in seen_hashes:
                unique_articles.append(article)
                seen_hashes.add(content_hash)
            elif content_hash in seen_hashes:
                logger.debug(
                    "Duplicate article filtered",
                    title=article.get('title', 'Unknown'),
                    content_hash=content_hash
                )

        return unique_articles

    def reset_deduplication_cache(self) -> None:
        """Reset the deduplication cache"""
        self.seen_hashes.clear()
        logger.info("Deduplication cache reset")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the deduplication cache"""
        return {
            'cached_hashes': len(self.seen_hashes),
            'cache_memory_mb': len(self.seen_hashes) * 32 / (1024 * 1024)  # Approximate
        }