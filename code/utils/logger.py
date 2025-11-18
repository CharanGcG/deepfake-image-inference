"""
utils package for deepfake detection project.

Contains:
- checkpoint.py: save/load model checkpoints
- metrics.py: compute evaluation metrics
- scheduler.py: learning rate scheduler with warmup
- seed.py: reproducibility utilities
- logger.py: session logging utilities

Each module follows best practices, with docstrings, exception handling,
and edge case safety.
"""

# utils/logger.py
import logging
import os
from datetime import datetime


def get_logger(log_dir: str, name: str = "train") -> logging.Logger:
    """Create a logger that writes to file and console.

    Args:
        log_dir (str): Directory to store logs.
        name (str): Logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger
    except Exception as e:
        raise RuntimeError(f"Failed to create logger: {e}")
