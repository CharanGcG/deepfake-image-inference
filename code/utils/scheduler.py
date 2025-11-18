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

# utils/scheduler.py
import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """Cosine annealing scheduler with warmup."""

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = float(self.last_epoch + 1) / float(max(1, self.warmup_epochs))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
            return [base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

