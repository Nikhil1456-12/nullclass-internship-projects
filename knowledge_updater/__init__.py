"""
Knowledge Base Update System for Chatbot

A comprehensive system for dynamically expanding chatbot knowledge through
automated, periodic updates to a vector database.
"""

__version__ = "1.0.0"
__author__ = "Knowledge Updater System"

from .core.config import ConfigManager
from .core.scheduler import KnowledgeUpdateScheduler
from .core.logging import setup_logging

__all__ = [
    "ConfigManager",
    "KnowledgeUpdateScheduler",
    "setup_logging"
]