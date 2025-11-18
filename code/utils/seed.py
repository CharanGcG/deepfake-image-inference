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

# utils/seed.py
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all random generators for reproducibility."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        raise RuntimeError(f"Failed to set seed: {e}")

