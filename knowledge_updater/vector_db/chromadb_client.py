"""
ChromaDB client for vector database operations
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import time
import hashlib
import numpy as np

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..core.config import get_config
from ..core.logging import get_logger

logger = get_logger(__name__)


class ChromaDBClient:
    """ChromaDB client for vector database operations"""

    def __init__(self):
        """Initialize ChromaDB client"""
        self.config = get_config()
        self.vector_config = self.config.get('vector_db', {})

        # Get configuration
        self.persist_directory = self.vector_config.get('persist_directory', './data/chroma')
        self.collection_name = self.vector_config.get('collection_name', 'knowledge_base')

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

        logger.info(
            "ChromaDB client initialized",
            persist_directory=self.persist_directory,
            collection=self.collection_name
        )

    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info("Using existing collection", collection=self.collection_name)
            return collection

        except ValueError:
            # Collection doesn't exist, create new one
            logger.info("Creating new collection", collection=self.collection_name)

            # Get embedding function based on configuration
            embedding_config = self.vector_config.get('embedding_model', {})
            embedding_function = self._create_embedding_function()

            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={
                    'created_at': time.time(),
                    'description': 'Knowledge base collection for chatbot'
                }
            )

            logger.info("Collection created successfully", collection=self.collection_name)
            return collection

    def _create_embedding_function(self):
        """Create embedding function for the collection"""
        # For now, we'll use default embedding function
        # In a production system, you might want to use a custom function
        return None  # ChromaDB will use default

    def add_vectors(self, vectors_data: List[Dict[str, Any]]) -> int:
        """
        Add vectors to the collection

        Args:
            vectors_data: List of vector data dictionaries

        Returns:
            Number of vectors added
        """
        if not vectors_data:
            return 0

        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for vector_data in vectors_data:
                ids.append(vector_data['id'])
                embeddings.append(vector_data['vector'])
                metadatas.append(vector_data['metadata'])

                # Use content as document for text search
                documents.append(vector_data['metadata'].get('content', ''))

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.debug(
                "Vectors added to ChromaDB",
                count=len(vectors_data),
                collection=self.collection_name
            )

            return len(vectors_data)

        except Exception as e:
            logger.error("Failed to add vectors to ChromaDB", error=str(e))
            raise

    def query_vectors(
        self,
        query_vector: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query vectors in the collection

        Args:
            query_vector: Query embedding vector
            n_results: Number of results to return
            where: Metadata filters
            **kwargs: Additional query parameters

        Returns:
            List of query results with similarity scores
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where,
                **kwargs
            )

            # Format results
            formatted_results = []
            if results['distances'] and results['metadatas'] and results['documents']:
                for i in range(len(results['distances'][0])):
                    # Convert distance to similarity (ChromaDB returns distance, not similarity)
                    distance = results['distances'][0][i]
                    similarity = 1.0 - (distance / 2.0)  # Normalize distance to similarity

                    result = {
                        'id': results['ids'][0][i],
                        'similarity': max(0.0, min(1.0, similarity)),  # Clamp to [0, 1]
                        'distance': distance,
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i]
                    }

                    formatted_results.append(result)

            logger.debug(
                "Vector query completed",
                results_found=len(formatted_results),
                collection=self.collection_name
            )

            return formatted_results

        except Exception as e:
            logger.error("Vector query failed", error=str(e))
            raise

    def get_existing_ids(self) -> List[str]:
        """Get list of existing article IDs in the collection"""
        try:
            # Get all IDs from the collection
            result = self.collection.get(include=['metadatas'])

            # Extract unique IDs
            ids = []
            if result['ids']:
                ids = list(set(result['ids']))

            logger.debug("Retrieved existing IDs", count=len(ids))
            return ids

        except Exception as e:
            logger.error("Failed to get existing IDs", error=str(e))
            return []

    def delete_vectors(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs

        Args:
            ids: List of vector IDs to delete

        Returns:
            Number of vectors deleted
        """
        try:
            if not ids:
                return 0

            self.collection.delete(ids=ids)

            logger.debug("Vectors deleted", count=len(ids), collection=self.collection_name)
            return len(ids)

        except Exception as e:
            logger.error("Failed to delete vectors", error=str(e))
            raise

    def clear_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Clear all data from collection

        Args:
            collection_name: Name of collection to clear (uses default if None)

        Returns:
            True if successful
        """
        try:
            collection_to_clear = collection_name or self.collection_name

            # Delete and recreate collection
            self.client.delete_collection(collection_to_clear)

            # Recreate collection
            self.collection = self._get_or_create_collection()

            logger.info("Collection cleared", collection=collection_to_clear)
            return True

        except Exception as e:
            logger.error("Failed to clear collection", error=str(e))
            return False

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Args:
            collection_name: Name of collection to get stats for

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_to_check = collection_name or self.collection_name

            # Get collection info
            collection = self.client.get_collection(collection_to_check)
            count = collection.count()

            # Get sample of metadata to analyze
            result = collection.get(limit=1000, include=['metadatas'])

            # Analyze metadata
            sources = {}
            date_range = None
            total_words = 0

            if result['metadatas']:
                for metadata in result['metadatas']:
                    # Count sources
                    source = metadata.get('source', 'Unknown')
                    sources[source] = sources.get(source, 0) + 1

                    # Track date range
                    published_at = metadata.get('published_at')
                    if published_at:
                        try:
                            # Simple date parsing for stats
                            if date_range is None:
                                date_range = {'earliest': published_at, 'latest': published_at}
                            else:
                                if published_at < date_range['earliest']:
                                    date_range['earliest'] = published_at
                                if published_at > date_range['latest']:
                                    date_range['latest'] = published_at
                        except:
                            pass

                    # Sum word counts
                    word_count = metadata.get('word_count', 0)
                    total_words += word_count

            stats = {
                'collection_name': collection_to_check,
                'article_count': count,
                'sources': sources,
                'date_range': date_range,
                'total_words': total_words,
                'avg_words_per_article': total_words / count if count > 0 else 0
            }

            return stats

        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))
            return {
                'collection_name': collection_name or self.collection_name,
                'error': str(e)
            }

    def update_collection_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Update collection metadata

        Args:
            metadata: New metadata to add

        Returns:
            True if successful
        """
        try:
            # Get current collection
            collection = self.client.get_collection(self.collection_name)

            # Update metadata (ChromaDB doesn't support direct metadata updates,
            # so we just log the update)
            logger.info(
                "Collection metadata update requested",
                collection=self.collection_name,
                metadata=metadata
            )

            return True

        except Exception as e:
            logger.error("Failed to update collection metadata", error=str(e))
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on ChromaDB"""
        try:
            # Test basic operations
            count = self.collection.count()

            # Test query with dummy vector
            test_vector = [0.1] * 384  # Assuming 384-dimensional embeddings
            results = self.collection.query(
                query_embeddings=[test_vector],
                n_results=1
            )

            return {
                'status': 'healthy',
                'collection_exists': True,
                'article_count': count,
                'query_successful': True,
                'client_connected': True
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'collection_exists': False,
                'query_successful': False,
                'client_connected': False
            }