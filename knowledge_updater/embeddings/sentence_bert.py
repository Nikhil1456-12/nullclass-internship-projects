"""
Sentence-BERT embeddings backend for text vectorization
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import time

from sentence_transformers import SentenceTransformer
import torch

from ..core.config import get_config
from ..core.logging import get_logger

logger = get_logger(__name__)


class SentenceBERTEmbeddings:
    """Sentence-BERT embeddings backend"""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Sentence-BERT embeddings

        Args:
            model_name: Name of the Sentence-BERT model to use
        """
        self.config = get_config()
        self.vector_config = self.config.get('vector_db', {})

        # Get model name from config or parameter
        self.model_name = model_name or self.vector_config.get(
            'embedding_model',
            'sentence-transformers/all-MiniLM-L6-v2'
        )

        # Initialize model
        self.model = None
        self.device = self._get_device()
        self.batch_size = self.vector_config.get('embedding_batch_size', 32)
        self.max_length = self.vector_config.get('embedding_max_length', 512)

        self._load_model()

        logger.info(
            "Sentence-BERT embeddings initialized",
            model=self.model_name,
            device=str(self.device),
            batch_size=self.batch_size
        )

    def _get_device(self) -> str:
        """Get the appropriate device (CPU/GPU)"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon
        else:
            return 'cpu'

    def _load_model(self) -> None:
        """Load the Sentence-BERT model"""
        try:
            logger.info("Loading Sentence-BERT model", model=self.model_name)

            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Set max sequence length if specified
            if self.max_length and hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_length

            logger.info(
                "Sentence-BERT model loaded successfully",
                model=self.model_name,
                device=str(self.device)
            )

        except Exception as e:
            logger.error("Failed to load Sentence-BERT model", error=str(e))
            raise

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using Sentence-BERT

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.model is None:
            raise RuntimeError("Sentence-BERT model not loaded")

        batch_size = batch_size or self.batch_size

        try:
            # Generate embeddings in batches
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=len(batch)
                )

                # Convert to list of lists
                for embedding in batch_embeddings:
                    all_embeddings.append(embedding.tolist())

                logger.debug(
                    "Sentence-BERT batch processed",
                    batch_index=i//batch_size,
                    batch_size=len(batch),
                    embeddings_generated=len(batch_embeddings)
                )

            return all_embeddings

        except Exception as e:
            logger.error("Sentence-BERT embedding generation failed", error=str(e))
            raise

    async def generate_embeddings_async(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings asynchronously using Sentence-BERT

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            List of embedding vectors
        """
        # Sentence-BERT doesn't have native async support, so we run it in a thread pool
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            None,
            self.generate_embeddings,
            texts,
            batch_size,
            **kwargs
        )

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of Sentence-BERT embeddings"""
        if self.model is None:
            raise RuntimeError("Sentence-BERT model not loaded")

        return self.model.get_sentence_embedding_dimension()

    def get_config(self) -> Dict[str, Any]:
        """Get Sentence-BERT configuration"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'embedding_dimension': self.get_embedding_dimension()
        }

    def health_check(self) -> Dict[str, Any]:
        """Check if Sentence-BERT model is working"""
        try:
            # Test with a simple embedding
            test_text = ["Test sentence for health check"]
            embeddings = self.generate_embeddings(test_text)

            return {
                'status': 'healthy',
                'model_loaded': True,
                'model_name': self.model_name,
                'device': str(self.device),
                'embedding_dimension': len(embeddings[0]) if embeddings else 0
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_loaded': False,
                'error': str(e),
                'model_name': self.model_name,
                'device': str(self.device)
            }

    def reload_model(self, model_name: Optional[str] = None) -> None:
        """
        Reload the model (useful for switching models)

        Args:
            model_name: New model name (optional)
        """
        if model_name:
            self.model_name = model_name

        logger.info("Reloading Sentence-BERT model", model=self.model_name)

        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._load_model()

        logger.info("Sentence-BERT model reloaded", model=self.model_name)

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the loaded model"""
        if self.model is None:
            return {'loaded': False}

        return {
            'loaded': True,
            'model_name': self.model_name,
            'device': str(self.device),
            'embedding_dimension': self.get_embedding_dimension(),
            'max_seq_length': getattr(self.model, 'max_seq_length', None),
            'model_card': getattr(self.model, '_model_card_text', None)[:500] + "..." if hasattr(self.model, '_model_card_text') else None
        }