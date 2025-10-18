"""
Vector database manager for coordinating vector operations
"""

import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

from ..core.config import get_config
from ..core.logging import get_logger, LogContext
from ..embeddings.generator import EmbeddingGenerator
from .faiss_client import FAISSClient

logger = get_logger(__name__)


class VectorDBManager:
    """Manages vector database operations"""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize vector database manager

        Args:
            provider: Vector database provider ('chromadb', 'faiss')
        """
        self.config = get_config()
        self.provider_name = provider or self.config.get('vector_db', {}).get('provider', 'chromadb')

        # Initialize vector database client
        if self.provider_name == 'chromadb':
            self.client = ChromaDBClient()
        elif self.provider_name == 'faiss':
            self.client = FAISSClient()
        else:
            raise ValueError(f"Unsupported vector database provider: {self.provider_name}")

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()

        # Get configuration
        self.collection_name = self.config.get('vector_db', {}).get('collection_name', 'knowledge_base')
        self.persist_directory = self.config.get('vector_db', {}).get('persist_directory', './data/chroma')

        logger.info(
            "Vector database manager initialized",
            provider=self.provider_name,
            collection=self.collection_name
        )

    def update_knowledge_base(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update knowledge base with new articles

        Args:
            articles: List of processed articles

        Returns:
            Update statistics
        """
        if not articles:
            logger.info("No articles to update")
            return {'articles_processed': 0, 'vectors_added': 0}

        start_time = time.time()

        try:
            with LogContext(logger, article_count=len(articles)):
                logger.info("Starting knowledge base update")

                # Filter out articles that are already in the database
                new_articles = self._filter_existing_articles(articles)

                if not new_articles:
                    logger.info("All articles already exist in database")
                    return {
                        'articles_processed': len(articles),
                        'vectors_added': 0,
                        'duplicates_filtered': len(articles)
                    }

                # Generate embeddings for new articles
                texts = [article.get('content', '') for article in new_articles]
                embeddings = self.embedding_generator.generate_embeddings(texts)

                if not embeddings:
                    logger.warning("No embeddings generated")
                    return {'articles_processed': len(articles), 'vectors_added': 0}

                # Prepare data for vector database
                vectors_data = self._prepare_vectors_data(new_articles, embeddings)

                # Add vectors to database
                vectors_added = self.client.add_vectors(vectors_data)

                update_time = time.time() - start_time

                logger.info(
                    "Knowledge base update completed",
                    articles_processed=len(articles),
                    new_articles=len(new_articles),
                    vectors_added=vectors_added,
                    update_time=f"{update_time:.3f}s"
                )

                return {
                    'articles_processed': len(articles),
                    'new_articles': len(new_articles),
                    'vectors_added': vectors_added,
                    'duplicates_filtered': len(articles) - len(new_articles),
                    'update_time': update_time
                }

        except Exception as e:
            update_time = time.time() - start_time
            logger.error(
                "Knowledge base update failed",
                update_time=f"{update_time:.3f}s",
                error=str(e)
            )
            raise

    def _filter_existing_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out articles that already exist in the database"""
        if not articles:
            return []

        # Get existing article IDs
        existing_ids = set(self.client.get_existing_ids())

        # Filter new articles
        new_articles = []
        for article in articles:
            article_id = article.get('id', '')
            if article_id not in existing_ids:
                new_articles.append(article)
            else:
                logger.debug("Article already exists", article_id=article_id)

        return new_articles

    def _prepare_vectors_data(
        self,
        articles: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """Prepare article data for vector database storage"""
        vectors_data = []

        for article, embedding in zip(articles, embeddings):
            if not embedding:
                continue

            # Create metadata
            metadata = {
                'title': article.get('title', ''),
                'content': article.get('content', ''),
                'published_at': article.get('published_at', ''),
                'source': article.get('metadata', {}).get('source', ''),
                'url': article.get('metadata', {}).get('url', ''),
                'author': article.get('metadata', {}).get('author', ''),
                'word_count': article.get('word_count', 0),
                'entities': article.get('entities', []),
                'keywords': article.get('keywords', []),
                'summary': article.get('summary', ''),
                'content_hash': article.get('content_hash', ''),
                'added_at': datetime.now(timezone.utc).isoformat(),
                'processing_metadata': article.get('processing_metadata', {})
            }

            # Add any additional metadata from the article
            for key, value in article.get('metadata', {}).items():
                if key not in metadata:
                    metadata[f"article_{key}"] = value

            vector_data = {
                'id': article.get('id', ''),
                'vector': embedding,
                'metadata': metadata
            }

            vectors_data.append(vector_data)

        return vectors_data

    def query_knowledge_base(
        self,
        query: str,
        n_results: int = 5,
        similarity_threshold: float = 0.7,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information

        Args:
            query: Search query
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score
            **kwargs: Additional query parameters

        Returns:
            List of relevant articles with similarity scores
        """
        if not query.strip():
            return []

        start_time = time.time()

        try:
            logger.debug("Querying knowledge base", query=query[:50], n_results=n_results)

            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embeddings([query])

            if not query_embedding or not query_embedding[0]:
                logger.warning("Failed to generate query embedding")
                return []

            # Query vector database
            results = self.client.query_vectors(
                query_embedding[0],
                n_results=n_results,
                **kwargs
            )

            # Filter by similarity threshold
            filtered_results = []
            for result in results:
                similarity = result.get('similarity', 0)
                if similarity >= similarity_threshold:
                    filtered_results.append(result)
                else:
                    logger.debug(
                        "Result filtered by similarity threshold",
                        similarity=similarity,
                        threshold=similarity_threshold
                    )

            query_time = time.time() - start_time

            logger.info(
                "Knowledge base query completed",
                query_length=len(query),
                results_found=len(results),
                results_returned=len(filtered_results),
                query_time=f"{query_time:.3f}s"
            )

            return filtered_results

        except Exception as e:
            query_time = time.time() - start_time
            logger.error(
                "Knowledge base query failed",
                query_time=f"{query_time:.3f}s",
                error=str(e)
            )
            raise

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return stats
        except Exception as e:
            logger.error("Failed to get database stats", error=str(e))
            return {}

    def delete_articles(self, article_ids: List[str]) -> int:
        """
        Delete articles from the knowledge base

        Args:
            article_ids: List of article IDs to delete

        Returns:
            Number of articles deleted
        """
        try:
            deleted_count = self.client.delete_vectors(article_ids)
            logger.info("Articles deleted", count=deleted_count, ids=article_ids)
            return deleted_count
        except Exception as e:
            logger.error("Failed to delete articles", error=str(e))
            raise

    def clear_collection(self) -> bool:
        """
        Clear all data from the collection

        Returns:
            True if successful
        """
        try:
            success = self.client.clear_collection(self.collection_name)
            if success:
                logger.info("Collection cleared", collection=self.collection_name)
            return success
        except Exception as e:
            logger.error("Failed to clear collection", error=str(e))
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector database"""
        try:
            # Check if collection exists and is accessible
            stats = self.get_database_stats()

            # Test query
            test_query = "test query"
            results = self.query_knowledge_base(test_query, n_results=1)

            return {
                'status': 'healthy',
                'provider': self.provider_name,
                'collection_exists': True,
                'article_count': stats.get('article_count', 0),
                'query_successful': True
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'provider': self.provider_name,
                'error': str(e),
                'collection_exists': False,
                'query_successful': False
            }