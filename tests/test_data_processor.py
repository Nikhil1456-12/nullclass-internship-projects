"""
Unit tests for data processing functionality
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from knowledge_updater.data_sources.data_processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor"""

    def setup_method(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()

        self.sample_articles = [
            {
                'id': 'test_1',
                'title': 'Test Article 1',
                'content': 'This is a test article with some content. It contains information about testing and quality assurance.',
                'published_at': datetime.now(timezone.utc),
                'metadata': {'source': 'Test Source'},
                'content_hash': 'hash1',
                'word_count': 15
            },
            {
                'id': 'test_2',
                'title': 'Test Article 2',
                'content': 'Another test article with different content. This one discusses various topics and includes technical details.',
                'published_at': datetime.now(timezone.utc),
                'metadata': {'source': 'Another Source'},
                'content_hash': 'hash2',
                'word_count': 16
            }
        ]

    def test_process_single_article(self):
        """Test processing a single article"""
        article = self.sample_articles[0].copy()
        result = self.processor._process_single_article(article)

        assert result is not None
        assert 'content' in result
        assert 'word_count' in result
        assert 'entities' in result
        assert 'keywords' in result
        assert 'summary' in result
        assert 'processed_at' in result

        # Check that content is cleaned
        assert 'http' not in result['content']  # URLs should be removed

    def test_process_article_with_empty_content(self):
        """Test processing article with empty content"""
        article = {
            'id': 'empty_test',
            'title': 'Empty Content',
            'content': '',
            'published_at': datetime.now(timezone.utc),
            'metadata': {},
            'content_hash': 'empty_hash',
            'word_count': 0
        }

        result = self.processor._process_single_article(article)
        assert result is None  # Should return None for empty content

    def test_process_article_with_short_content(self):
        """Test processing article with very short content"""
        article = {
            'id': 'short_test',
            'title': 'Short',
            'content': 'Short content',
            'published_at': datetime.now(timezone.utc),
            'metadata': {},
            'content_hash': 'short_hash',
            'word_count': 2
        }

        result = self.processor._process_single_article(article)
        assert result is None  # Should return None for too short content

    def test_process_articles_batch(self):
        """Test processing multiple articles"""
        articles = self.sample_articles.copy()
        results = self.processor.process_articles(articles)

        assert len(results) == 2
        for result in results:
            assert 'entities' in result
            assert 'keywords' in result
            assert 'summary' in result

    def test_deduplication(self):
        """Test article deduplication"""
        # Create duplicate articles
        articles = self.sample_articles.copy()
        duplicate_article = self.sample_articles[0].copy()
        duplicate_article['id'] = 'test_1_duplicate'
        articles.append(duplicate_article)

        results = self.processor.process_articles(articles)

        # Should have only 2 unique articles
        assert len(results) == 2

        # Check that duplicates are filtered
        ids = [article['id'] for article in results]
        assert 'test_1' in ids
        assert 'test_1_duplicate' not in ids

    def test_content_cleaning(self):
        """Test content cleaning functionality"""
        dirty_content = '''
        This is content with http://example.com URLs and test@example.com emails.
        It also has 123-456-7890 phone numbers and EXTRA   SPACES.
        Plus some HTML-like content <script>alert("test")</script>.
        '''

        cleaned = self.processor._clean_content(dirty_content)

        assert 'http' not in cleaned
        assert '@' not in cleaned
        assert '123-456-7890' not in cleaned
        assert '<script>' not in cleaned
        assert '  ' not in cleaned  # Multiple spaces should be normalized

    def test_entity_extraction(self):
        """Test entity extraction"""
        content = "Apple Inc. released the iPhone 14 in September 2023. Tim Cook announced the new features."
        entities = self.processor._extract_entities(content)

        assert 'Apple' in entities or 'Inc.' in entities
        assert 'iPhone' in entities
        assert 'Tim' in entities or 'Cook' in entities

    def test_keyword_extraction(self):
        """Test keyword extraction"""
        content = "Machine learning and artificial intelligence are important technologies. Python programming is widely used in data science."
        keywords = self.processor._extract_keywords(content)

        # Should extract meaningful keywords
        assert any(kw in ['machine', 'learning', 'artificial', 'intelligence', 'python', 'programming', 'data', 'science'] for kw in keywords)

        # Should not include stop words
        assert 'and' not in keywords
        assert 'are' not in keywords
        assert 'in' not in keywords

    def test_summary_generation(self):
        """Test summary generation"""
        content = "This is the first sentence. This is the second sentence with more details. This is the third sentence that concludes the content."
        title = "Test Article"

        summary = self.processor._generate_summary(content, title)

        assert len(summary) > 0
        assert 'first sentence' in summary.lower()
        assert 'second sentence' in summary.lower()
        assert len(summary) <= 200  # Should be truncated if too long

    def test_cache_reset(self):
        """Test deduplication cache reset"""
        # Process some articles to populate cache
        self.processor.process_articles(self.sample_articles)

        # Check cache stats
        stats = self.processor.get_cache_stats()
        assert stats['cached_hashes'] > 0

        # Reset cache
        self.processor.reset_deduplication_cache()

        # Check cache is empty
        stats = self.processor.get_cache_stats()
        assert stats['cached_hashes'] == 0

    def test_content_length_validation(self):
        """Test content length validation"""
        # Test valid content
        valid_content = "This is a valid content with sufficient length to pass validation and provide meaningful information."
        assert self.processor._validate_content_length(valid_content, 15)

        # Test too short content
        short_content = "Short"
        assert not self.processor._validate_content_length(short_content, 1)

        # Test too long content (this would be tested with actual config limits)
        # For this test, we'll assume the limits are reasonable

    def test_article_validation(self):
        """Test article validation"""
        # Valid article
        valid_article = {
            'id': 'valid_test',
            'title': 'Valid Article',
            'content': 'This is valid content with sufficient length and quality information for processing.',
            'published_at': datetime.now(timezone.utc),
            'metadata': {'source': 'Test'},
            'content_hash': 'valid_hash',
            'word_count': 15
        }

        assert self.processor._validate_article(valid_article)

        # Invalid article - too short
        invalid_article = {
            'id': 'invalid_test',
            'title': 'Short',
            'content': 'Short',
            'published_at': datetime.now(timezone.utc),
            'metadata': {'source': 'Test'},
            'content_hash': 'invalid_hash',
            'word_count': 1
        }

        assert not self.processor._validate_article(invalid_article)

    def test_spam_filtering(self):
        """Test spam content filtering"""
        # Article with spam content
        spam_article = {
            'id': 'spam_test',
            'title': 'Click Here for Amazing Results',
            'content': 'This is spam content with advertisement and clickbait material that should be filtered.',
            'published_at': datetime.now(timezone.utc),
            'metadata': {'source': 'Spam Source'},
            'content_hash': 'spam_hash',
            'word_count': 15
        }

        # This would normally be filtered by the spam keywords in config
        # For this test, we'll check the validation logic
        assert not self.processor._validate_article(spam_article)

    def test_process_articles_empty_list(self):
        """Test processing empty article list"""
        results = self.processor.process_articles([])
        assert results == []

    def test_process_articles_with_none_values(self):
        """Test processing articles with None values"""
        articles_with_none = [
            {
                'id': 'none_test',
                'title': None,
                'content': 'Valid content',
                'published_at': datetime.now(timezone.utc),
                'metadata': {},
                'content_hash': 'none_hash',
                'word_count': 2
            }
        ]

        results = self.processor.process_articles(articles_with_none)
        # Should filter out articles with None titles
        assert len(results) == 0