"""
Main embedding generator for converting text to vectors
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
import time
from datetime import datetime, timezone

from ..core.config import get_config
from ..core.logging import get_logger, LogContext
from .openai_embeddings import OpenAIEmbeddings
from .sentence_bert import SentenceBERTEmbeddings

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Main embedding generator that supports multiple backends"""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize embedding generator

        Args:
            provider: Embedding provider ('openai', 'sentence_bert', or None for auto)
        """
        self.config = get_config()
        self.provider_name = provider or self._get_default_provider()

        # Initialize the appropriate embedding backend
        if self.provider_name == 'openai':
            self.backend = OpenAIEmbeddings()
        elif self.provider_name == 'sentence_bert':
            self.backend = SentenceBERTEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider_name}")

        logger.info(
            "Embedding generator initialized",
            provider=self.provider_name,
            backend_type=type(self.backend).__name__
        )

    def _get_default_provider(self) -> str:
        """Get default embedding provider from config"""
        vector_config = self.config.get('vector_db', {})
        return vector_config.get('embedding_provider', 'sentence_bert')

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings
            batch_size: Batch size for processing (uses config default if None)
            **kwargs: Additional arguments for the backend

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start_time = time.time()

        try:
            with LogContext(logger, provider=self.provider_name, text_count=len(texts)):
                # Get batch size from config or parameter
                if batch_size is None:
                    vector_config = self.config.get('vector_db', {})
                    batch_size = vector_config.get('embedding_batch_size', 32)

                logger.debug(
                    "Generating embeddings",
                    text_count=len(texts),
                    batch_size=batch_size
                )

                # Generate embeddings using backend
                embeddings = self.backend.generate_embeddings(texts, batch_size, **kwargs)

                # Validate embeddings
                validated_embeddings = self._validate_embeddings(embeddings, texts)

                generation_time = time.time() - start_time
                logger.info(
                    "Embeddings generated successfully",
                    text_count=len(texts),
                    embedding_count=len(validated_embeddings),
                    generation_time=f"{generation_time:.3f}s",
                    avg_time_per_text=f"{generation_time/len(texts):.3f}s"
                )

                return validated_embeddings

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(
                "Embedding generation failed",
                generation_time=f"{generation_time:.3f}s",
                error=str(e)
            )
            raise

    async def generate_embeddings_async(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings asynchronously

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            **kwargs: Additional arguments for the backend

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start_time = time.time()

        try:
            with LogContext(logger, provider=self.provider_name, text_count=len(texts)):
                # Get batch size from config or parameter
                if batch_size is None:
                    vector_config = self.config.get('vector_db', {})
                    batch_size = vector_config.get('embedding_batch_size', 32)

                logger.debug(
                    "Generating embeddings asynchronously",
                    text_count=len(texts),
                    batch_size=batch_size
                )

                # Generate embeddings using backend
                embeddings = await self.backend.generate_embeddings_async(texts, batch_size, **kwargs)

                # Validate embeddings
                validated_embeddings = self._validate_embeddings(embeddings, texts)

                generation_time = time.time() - start_time
                logger.info(
                    "Async embeddings generated successfully",
                    text_count=len(texts),
                    embedding_count=len(validated_embeddings),
                    generation_time=f"{generation_time:.3f}s"
                )

                return validated_embeddings

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(
                "Async embedding generation failed",
                generation_time=f"{generation_time:.3f}s",
                error=str(e)
            )
            raise

    def _validate_embeddings(self, embeddings: List[List[float]], texts: List[str]) -> List[List[float]]:
        """
        Validate generated embeddings

        Args:
            embeddings: Generated embedding vectors
            texts: Original text strings

        Returns:
            Validated embeddings

        Raises:
            ValueError: If embeddings are invalid
        """
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        # Check for empty or invalid embeddings
        valid_embeddings = []
        for i, (embedding, text) in enumerate(zip(embeddings, texts)):
            if not embedding:
                logger.warning("Empty embedding generated", text_index=i, text_preview=text[:50])
                continue

            if not all(isinstance(x, (int, float)) for x in embedding):
                logger.warning("Invalid embedding values", text_index=i)
                continue

            # Check embedding dimensionality
            embedding_array = np.array(embedding)
            if embedding_array.ndim != 1:
                logger.warning("Invalid embedding dimensions", text_index=i, shape=embedding_array.shape)
                continue

            valid_embeddings.append(embedding)

        if len(valid_embeddings) != len(texts):
            logger.warning(
                "Some embeddings were invalid",
                total_texts=len(texts),
                valid_embeddings=len(valid_embeddings)
            )

        return valid_embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of generated embeddings"""
        return self.backend.get_embedding_dimension()

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current embedding provider"""
        return {
            'provider': self.provider_name,
            'backend': type(self.backend).__name__,
            'dimension': self.get_embedding_dimension(),
            'config': self.backend.get_config()
        }

    def switch_provider(self, provider: str) -> None:
        """
        Switch to a different embedding provider

        Args:
            provider: New provider name ('openai' or 'sentence_bert')
        """
        if provider == self.provider_name:
            return

        logger.info("Switching embedding provider", from_provider=self.provider_name, to_provider=provider)

        # Create new backend
        if provider == 'openai':
            self.backend = OpenAIEmbeddings()
        elif provider == 'sentence_bert':
            self.backend = SentenceBERTEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        self.provider_name = provider

        logger.info("Embedding provider switched", new_provider=provider)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the embedding generator

        Returns:
            Health check results
        """
        try:
            # Test with a simple text
            test_text = ["This is a test sentence for embedding generation."]
            embeddings = self.generate_embeddings(test_text)

            return {
                'status': 'healthy',
                'provider': self.provider_name,
                'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                'test_successful': len(embeddings) == 1,
                'response_time_ms': 0  # Would need to measure this properly
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'provider': self.provider_name,
                'error': str(e),
                'test_successful': False
            }