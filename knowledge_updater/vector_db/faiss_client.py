"""
FAISS client for vector database operations
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
import time
import hashlib
import numpy as np

import faiss

from ..core.config import get_config
from ..core.logging import get_logger

logger = get_logger(__name__)


class FAISSClient:
    """FAISS client for vector database operations"""

    def __init__(self):
        """Initialize FAISS client"""
        self.config = get_config()
        self.vector_config = self.config.get('vector_db', {})

        # Get configuration
        self.persist_directory = self.vector_config.get('persist_directory', './data/faiss')
        self.collection_name = self.vector_config.get('collection_name', 'knowledge_base')
        self.index_file = os.path.join(self.persist_directory, f'{self.collection_name}.faiss')
        self.metadata_file = os.path.join(self.persist_directory, f'{self.collection_name}_metadata.pkl')

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize or load FAISS index
        self.index = None
        self.metadata_db = {}  # Maps vector IDs to metadata
        self.id_to_idx = {}     # Maps vector IDs to FAISS indices
        self.idx_to_id = {}     # Maps FAISS indices to vector IDs

        self._load_or_create_index()

        logger.info(
            "FAISS client initialized",
            persist_directory=self.persist_directory,
            collection=self.collection_name,
            vectors_count=len(self.metadata_db)
        )

    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                # Load existing index
                self.index = faiss.read_index(self.index_file)

                # Load metadata
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata_db = data.get('metadata_db', {})
                    self.id_to_idx = data.get('id_to_idx', {})
                    self.idx_to_id = data.get('idx_to_id', {})

                logger.info("Loaded existing FAISS index", vectors_count=len(self.metadata_db))
                return
            except Exception as e:
                logger.warning("Failed to load existing index, creating new one", error=str(e))

        # Create new index (assuming 384-dimensional embeddings)
        embedding_dim = 384  # This should match your embedding model dimensions
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity

        logger.info("Created new FAISS index", embedding_dim=embedding_dim)

    def _save_index(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)

            # Save metadata
            data = {
                'metadata_db': self.metadata_db,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id
            }

            with open(self.metadata_file, 'wb') as f:
                pickle.dump(data, f)

            logger.debug("FAISS index saved to disk")
        except Exception as e:
            logger.error("Failed to save FAISS index", error=str(e))

    def add_vectors(self, vectors_data: List[Dict[str, Any]]) -> int:
        """
        Add vectors to the index

        Args:
            vectors_data: List of vector data dictionaries

        Returns:
            Number of vectors added
        """
        if not vectors_data:
            return 0

        try:
            # Prepare data for FAISS
            vectors = []
            new_metadata = {}
            new_id_to_idx = {}

            for vector_data in vectors_data:
                vector_id = vector_data['id']
                vector = np.array(vector_data['vector'], dtype=np.float32)
                metadata = vector_data['metadata'].copy()

                # Store vector in metadata for FAISS (since FAISS doesn't store original vectors)
                metadata['_vector'] = vector.tolist()

                vectors.append(vector)
                new_metadata[vector_id] = metadata
                new_id_to_idx[vector_id] = len(self.metadata_db) + len(vectors) - 1

            if not vectors:
                return 0

            # Add to FAISS index
            vectors_array = np.array(vectors, dtype=np.float32)

            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors_array)

            start_idx = len(self.metadata_db)
            self.index.add(vectors_array)

            # Update metadata and mappings
            self.metadata_db.update(new_metadata)
            self.id_to_idx.update(new_id_to_idx)

            # Update reverse mapping
            for i, vector_id in enumerate([v['id'] for v in vectors_data]):
                self.idx_to_id[start_idx + i] = vector_id

            # Save to disk
            self._save_index()

            logger.debug(
                "Vectors added to FAISS",
                count=len(vectors_data),
                collection=self.collection_name,
                total_vectors=len(self.metadata_db)
            )

            return len(vectors_data)

        except Exception as e:
            logger.error("Failed to add vectors to FAISS", error=str(e))
            raise

    def query_vectors(
        self,
        query_vector: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query vectors in the index

        Args:
            query_vector: Query embedding vector
            n_results: Number of results to return
            where: Metadata filters (not implemented for FAISS)
            **kwargs: Additional query parameters

        Returns:
            List of query results with similarity scores
        """
        try:
            if len(self.metadata_db) == 0:
                return []

            # Prepare query vector
            query_array = np.array([query_vector], dtype=np.float32)
            faiss.normalize_L2(query_array)

            # Search the index
            distances, indices = self.index.search(query_array, min(n_results, len(self.metadata_db)))

            # Format results
            formatted_results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # No more results
                    break

                # Convert distance to similarity (FAISS returns inner product for normalized vectors)
                similarity = (distance + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]

                vector_id = self.idx_to_id.get(idx, '')
                metadata = self.metadata_db.get(vector_id, {})

                result = {
                    'id': vector_id,
                    'similarity': max(0.0, min(1.0, similarity)),
                    'distance': distance,
                    'metadata': metadata,
                    'document': metadata.get('content', '')
                }

                formatted_results.append(result)

            logger.debug(
                "FAISS query completed",
                results_found=len(formatted_results),
                collection=self.collection_name
            )

            return formatted_results

        except Exception as e:
            logger.error("FAISS query failed", error=str(e))
            raise

    def get_existing_ids(self) -> List[str]:
        """Get list of existing article IDs in the index"""
        return list(self.metadata_db.keys())

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

            deleted_count = 0

            # Remove from metadata and mappings
            for vector_id in ids:
                if vector_id in self.metadata_db:
                    del self.metadata_db[vector_id]
                    del self.id_to_idx[vector_id]
                    deleted_count += 1

            # Rebuild index (FAISS doesn't support deletion, so we need to recreate)
            if deleted_count > 0:
                self._rebuild_index()

            logger.debug("Vectors deleted from FAISS", count=deleted_count, collection=self.collection_name)
            return deleted_count

        except Exception as e:
            logger.error("Failed to delete vectors from FAISS", error=str(e))
            raise

    def _rebuild_index(self):
        """Rebuild FAISS index after deletions"""
        try:
            # Get remaining vectors and metadata
            remaining_ids = list(self.metadata_db.keys())

            if not remaining_ids:
                # No vectors left, create empty index
                embedding_dim = 384
                self.index = faiss.IndexFlatIP(embedding_dim)
                self.id_to_idx = {}
                self.idx_to_id = {}
            else:
                # Rebuild index with remaining vectors
                vectors = []
                new_id_to_idx = {}
                new_idx_to_id = {}

                for i, vector_id in enumerate(remaining_ids):
                    metadata = self.metadata_db[vector_id]
                    vector = metadata.get('_vector', [])  # Store vector in metadata for rebuilding

                    if vector:
                        vectors.append(vector)
                        new_id_to_idx[vector_id] = i
                        new_idx_to_id[i] = vector_id

                if vectors:
                    vectors_array = np.array(vectors, dtype=np.float32)
                    faiss.normalize_L2(vectors_array)

                    embedding_dim = len(vectors[0])
                    self.index = faiss.IndexFlatIP(embedding_dim)
                    self.index.add(vectors_array)

                    self.id_to_idx = new_id_to_idx
                    self.idx_to_id = new_idx_to_id

            # Save to disk
            self._save_index()

        except Exception as e:
            logger.error("Failed to rebuild FAISS index", error=str(e))

    def clear_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Clear all data from collection

        Args:
            collection_name: Name of collection to clear (uses default if None)

        Returns:
            True if successful
        """
        try:
            # Clear all data
            self.index = faiss.IndexFlatIP(384)  # Reset to empty index
            self.metadata_db = {}
            self.id_to_idx = {}
            self.idx_to_id = {}

            # Save empty index
            self._save_index()

            logger.info("FAISS collection cleared", collection=collection_name or self.collection_name)
            return True

        except Exception as e:
            logger.error("Failed to clear FAISS collection", error=str(e))
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
            # Analyze metadata
            sources = {}
            date_range = None
            total_words = 0

            for metadata in self.metadata_db.values():
                # Count sources
                source = metadata.get('source', 'Unknown')
                sources[source] = sources.get(source, 0) + 1

                # Track date range
                published_at = metadata.get('published_at')
                if published_at:
                    try:
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
                'collection_name': collection_name or self.collection_name,
                'article_count': len(self.metadata_db),
                'sources': sources,
                'date_range': date_range,
                'total_words': total_words,
                'avg_words_per_article': total_words / len(self.metadata_db) if self.metadata_db else 0,
                'index_type': 'FAISS',
                'embedding_dim': self.index.d if hasattr(self.index, 'd') else 0
            }

            return stats

        except Exception as e:
            logger.error("Failed to get FAISS collection stats", error=str(e))
            return {
                'collection_name': collection_name or self.collection_name,
                'error': str(e)
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on FAISS"""
        try:
            # Test basic operations
            count = len(self.metadata_db)

            # Test query with dummy vector
            if count > 0:
                test_vector = [0.1] * 384
                results = self.query_vectors(test_vector, n_results=1)
                query_successful = True
            else:
                query_successful = True  # Empty index is still healthy

            return {
                'status': 'healthy',
                'collection_exists': True,
                'article_count': count,
                'query_successful': query_successful,
                'client_connected': True,
                'index_type': 'FAISS'
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'collection_exists': False,
                'query_successful': False,
                'client_connected': False,
                'index_type': 'FAISS'
            }