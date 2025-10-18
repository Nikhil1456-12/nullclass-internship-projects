"""
Vector database modules for the Knowledge Base Update System
"""

from .manager import VectorDBManager
from .faiss_client import FAISSClient

__all__ = [
    "VectorDBManager",
    "FAISSClient"
]