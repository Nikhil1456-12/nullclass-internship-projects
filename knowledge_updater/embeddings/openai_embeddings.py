"""
OpenAI embeddings backend for text vectorization
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
import time
import backoff

# Lazy import to avoid compatibility issues
def _import_openai():
    try:
        import openai
        from openai import AsyncOpenAI, OpenAI
        return openai, AsyncOpenAI, OpenAI
    except ImportError as e:
        raise ImportError(f"OpenAI package not available: {e}. Please install with: pip install openai")

from ..core.config import get_config
from ..core.logging import get_logger
from ..utils.rate_limiter import rate_limit

logger = get_logger(__name__)


class OpenAIEmbeddings:
    """OpenAI embeddings backend"""

    def __init__(self):
        """Initialize OpenAI embeddings client"""
        self.config = get_config()
        self.openai_config = self.config.get('vector_db', {}).get('openai_embeddings', {})

        # Lazy import OpenAI modules
        self.openai, self.AsyncOpenAI, self.OpenAI = _import_openai()

        # Get API key
        api_key = os.getenv(self.openai_config.get('api_key_env', 'OPENAI_API_KEY'))
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize clients
        self.client = self.OpenAI(api_key=api_key)
        self.async_client = self.AsyncOpenAI(api_key=api_key)

        # Get model configuration
        self.model = self.openai_config.get('model', 'text-embedding-ada-002')
        self.batch_size = self.openai_config.get('batch_size', 100)
        self.max_retries = 3

        logger.info(
            "OpenAI embeddings initialized",
            model=self.model,
            batch_size=self.batch_size
        )

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size

        # Process texts in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                batch_embeddings = self._generate_batch_embeddings(batch)
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    "OpenAI embedding batch processed",
                    batch_index=i//batch_size,
                    batch_size=len(batch),
                    embeddings_generated=len(batch_embeddings)
                )

            except Exception as e:
                logger.error(
                    "OpenAI embedding batch failed",
                    batch_index=i//batch_size,
                    error=str(e)
                )
                # Return empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings

    async def generate_embeddings_async(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings asynchronously using OpenAI API

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size

        # Process texts in batches asynchronously
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                batch_embeddings = await self._generate_batch_embeddings_async(batch)
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    "OpenAI async embedding batch processed",
                    batch_index=i//batch_size,
                    batch_size=len(batch),
                    embeddings_generated=len(batch_embeddings)
                )

            except Exception as e:
                logger.error(
                    "OpenAI async embedding batch failed",
                    batch_index=i//batch_size,
                    error=str(e)
                )
                # Return empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Use generic exception to avoid import issues
        max_tries=3,
        max_time=30
    )
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            embeddings = [data.embedding for data in response.data]
            return embeddings

        except Exception as e:
            logger.error("OpenAI embedding API call failed", error=str(e))
            raise

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Use generic exception to avoid import issues
        max_tries=3,
        max_time=30
    )
    async def _generate_batch_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts asynchronously"""
        try:
            response = await self.async_client.embeddings.create(
                input=texts,
                model=self.model
            )

            embeddings = [data.embedding for data in response.data]
            return embeddings

        except Exception as e:
            logger.error("OpenAI async embedding API call failed", error=str(e))
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of OpenAI embeddings"""
        # OpenAI ada-002 has 1536 dimensions
        if 'ada-002' in self.model:
            return 1536
        elif 'text-embedding-3-small' in self.model:
            return 1536
        elif 'text-embedding-3-large' in self.model:
            return 3072
        else:
            # Default fallback
            return 1536

    def get_config(self) -> Dict[str, Any]:
        """Get OpenAI embeddings configuration"""
        return {
            'model': self.model,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'api_key_env': self.openai_config.get('api_key_env', 'OPENAI_API_KEY')
        }

    def health_check(self) -> Dict[str, Any]:
        """Check if OpenAI API is accessible"""
        try:
            # Test with a simple embedding
            test_text = ["Test sentence for health check"]
            embeddings = self.generate_embeddings(test_text)

            return {
                'status': 'healthy',
                'api_accessible': True,
                'model': self.model,
                'embedding_dimension': len(embeddings[0]) if embeddings else 0
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'api_accessible': False,
                'error': str(e),
                'model': self.model
            }