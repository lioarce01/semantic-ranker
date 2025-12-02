"""
Common utilities for CLI scripts.

Provides shared functionality like path setup and logging configuration.
"""

import sys
import logging
from pathlib import Path


def setup_project_path():
    """Add parent directory to sys.path for imports

    This ensures semantic_ranker package can be imported
    regardless of where the CLI script is run from.
    """
    current_dir = Path(__file__).parent.parent
    parent_dir = current_dir.parent

    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


def setup_logging(level=logging.INFO, name: str = None) -> logging.Logger:
    """Configure logging with consistent format

    Args:
        level: Logging level (default: INFO)
        name: Logger name (default: __name__)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name or __name__)
