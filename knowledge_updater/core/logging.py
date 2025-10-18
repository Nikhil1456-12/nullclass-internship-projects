import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import structlog
from .config import get_config


def setup_logging(
    config_path: Optional[str] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    config = get_config()
    log_config = config.get('logging', {})

    level_name = log_level or log_config.get('level', 'INFO')
    file_path = log_file or log_config.get('file_path', './logs/knowledge_updater.log')

    log_dir = Path(file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level_name.upper(), logging.INFO)

    file_formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    max_bytes = log_config.get('max_file_size_mb', 100) * 1024 * 1024
    backup_count = log_config.get('backup_count', 5)

    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(numeric_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()
    logger.info("Logging system initialized", level=level_name, file_path=file_path)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


class LogContext:
    def __init__(self, logger: structlog.stdlib.BoundLogger, **context):
        self.logger = logger
        self.context = context
        self.bound_logger = None

    def __enter__(self) -> structlog.stdlib.BoundLogger:
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def log_performance(logger: structlog.stdlib.BoundLogger, operation: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                logger.info(
                    "Operation completed",
                    operation=operation,
                    execution_time=f"{execution_time:.3f}s",
                    success=True
                )
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Operation failed",
                    operation=operation,
                    execution_time=f"{execution_time:.3f}s",
                    error=str(e),
                    success=False
                )
                raise

        return wrapper
    return decorator