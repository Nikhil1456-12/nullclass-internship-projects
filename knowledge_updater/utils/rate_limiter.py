"""
Rate limiting utilities for API requests
"""

import time
import asyncio
from functools import wraps
from typing import Dict, Any, Callable
import threading
from datetime import datetime, timedelta

from ..core.config import get_config
from ..core.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Thread-safe rate limiter"""

    def __init__(self):
        """Initialize rate limiter"""
        self.config = get_config()
        self.requests: Dict[str, list] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str, requests_per_minute: int) -> bool:
        """
        Check if request is allowed under rate limit

        Args:
            key: Unique identifier for the rate limit (e.g., API name)
            requests_per_minute: Maximum requests per minute

        Returns:
            True if request is allowed, False otherwise
        """
        now = datetime.now()

        with self._lock:
            # Clean old requests (older than 1 minute)
            cutoff_time = now - timedelta(minutes=1)
            if key in self.requests:
                self.requests[key] = [
                    req_time for req_time in self.requests[key]
                    if req_time > cutoff_time
                ]
            else:
                self.requests[key] = []

            # Check if under limit
            if len(self.requests[key]) < requests_per_minute:
                self.requests[key].append(now)
                return True

            return False

    def get_wait_time(self, key: str, requests_per_minute: int) -> float:
        """
        Get time to wait before next request

        Args:
            key: Rate limit key
            requests_per_minute: Maximum requests per minute

        Returns:
            Seconds to wait (0 if no wait needed)
        """
        now = datetime.now()

        with self._lock:
            if key in self.requests:
                # Find oldest request in current window
                cutoff_time = now - timedelta(minutes=1)
                recent_requests = [
                    req_time for req_time in self.requests[key]
                    if req_time > cutoff_time
                ]

                if len(recent_requests) >= requests_per_minute and recent_requests:
                    oldest_request = min(recent_requests)
                    wait_time = (oldest_request + timedelta(minutes=1)) - now
                    return max(0, wait_time.total_seconds())

            return 0.0


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limit(requests_per_minute: int = None):
    """
    Decorator to rate limit function calls

    Args:
        requests_per_minute: Maximum requests per minute. If None, uses config default.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get rate limit from function or config
            limit = requests_per_minute
            if limit is None:
                config = get_config()
                limit = config.get('rate_limiting', {}).get('default_requests_per_minute', 60)

            # Generate key based on function name
            key = f"{func.__module__}.{func.__name__}"

            # Check if request is allowed
            if not _rate_limiter.is_allowed(key, limit):
                wait_time = _rate_limiter.get_wait_time(key, limit)
                if wait_time > 0:
                    logger.debug(
                        "Rate limit exceeded, waiting",
                        function=func.__name__,
                        wait_time=f"{wait_time:.2f}s"
                    )
                    time.sleep(wait_time)

                    # Check again after waiting
                    if not _rate_limiter.is_allowed(key, limit):
                        logger.warning(
                            "Rate limit still exceeded after waiting",
                            function=func.__name__,
                            limit=limit
                        )
                        raise Exception(f"Rate limit exceeded for {func.__name__}")

            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Function failed despite rate limiting",
                    function=func.__name__,
                    error=str(e)
                )
                raise

        return wrapper
    return decorator


async def async_rate_limit(requests_per_minute: int = None):
    """
    Async version of rate_limit decorator

    Args:
        requests_per_minute: Maximum requests per minute. If None, uses config default.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get rate limit from function or config
            limit = requests_per_minute
            if limit is None:
                config = get_config()
                limit = config.get('rate_limiting', {}).get('default_requests_per_minute', 60)

            # Generate key based on function name
            key = f"{func.__module__}.{func.__name__}"

            # Check if request is allowed
            if not _rate_limiter.is_allowed(key, limit):
                wait_time = _rate_limiter.get_wait_time(key, limit)
                if wait_time > 0:
                    logger.debug(
                        "Rate limit exceeded, waiting",
                        function=func.__name__,
                        wait_time=f"{wait_time:.2f}s"
                    )
                    await asyncio.sleep(wait_time)

                    # Check again after waiting
                    if not _rate_limiter.is_allowed(key, limit):
                        logger.warning(
                            "Rate limit still exceeded after waiting",
                            function=func.__name__,
                            limit=limit
                        )
                        raise Exception(f"Rate limit exceeded for {func.__name__}")

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Async function failed despite rate limiting",
                    function=func.__name__,
                    error=str(e)
                )
                raise

        return wrapper
    return decorator


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    return _rate_limiter


def reset_rate_limits() -> None:
    """Reset all rate limiting counters"""
    global _rate_limiter
    _rate_limiter = RateLimiter()
    logger.info("Rate limits reset")


def get_rate_limit_status() -> Dict[str, Any]:
    """Get current rate limit status"""
    status = {}

    with _rate_limiter._lock:
        for key, requests in _rate_limiter.requests.items():
            status[key] = {
                'request_count': len(requests),
                'oldest_request': min(requests).isoformat() if requests else None,
                'newest_request': max(requests).isoformat() if requests else None
            }

    return status