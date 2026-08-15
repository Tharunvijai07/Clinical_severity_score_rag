"""
utils/logger.py
───────────────
Provides a pre-configured loguru logger for every module.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
"""
import sys
from loguru import logger as _root_logger

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _root_logger.remove()  # remove default handler
    _root_logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    _CONFIGURED = True


def get_logger(name: str):
    """Return a named loguru logger (bound with the caller's module name)."""
    _configure()
    return _root_logger.bind(name=name)
