import logging
import os
from logging.handlers import RotatingFileHandler

# Create a logger instance
logger = logging.getLogger("iaat_whisper")

def setup_logger():
    """Setup the logger for the application."""
    # Prevent re-configuration
    if logger.hasHandlers():
        return

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = RotatingFileHandler(
        os.path.join(log_dir, "usage.log"), maxBytes=10 * 1024 * 1024, backupCount=5
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)