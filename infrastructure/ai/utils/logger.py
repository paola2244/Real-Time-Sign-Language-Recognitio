"""
Logger configuration module for the sign language recognition system.

This module sets up centralized logging with both file and console output,
configurable log levels, and consistent formatting across the application.
"""

import logging
import os
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with both file and console handlers.

    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, logging.DEBUG)
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler - UTF-8 encoding para evitar problemas de charmap
    log_file = logs_dir / f"sign_language_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Console handler - UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    # Configurar el stream para UTF-8 en Windows
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


class LoggerFactory:
    """Factory class for creating configured loggers."""

    _loggers = {}

    @staticmethod
    def get_logger(name: str, log_level=logging.INFO) -> logging.Logger:
        """
        Get or create a logger with caching.

        Args:
            name: Logger name
            log_level: Logging level

        Returns:
            logging.Logger: Logger instance
        """
        if name not in LoggerFactory._loggers:
            LoggerFactory._loggers[name] = setup_logger(name, log_level)
        return LoggerFactory._loggers[name]

    @staticmethod
    def set_log_level(level):
        """Set log level for all registered loggers."""
        for logger in LoggerFactory._loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
