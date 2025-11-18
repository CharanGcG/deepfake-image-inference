# code/transforms.py
"""
Transforms module exposing train and val transforms.

Uses torchvision.transforms and provides simple interfaces.
"""

from torchvision import transforms
from typing import Callable


def get_train_transform(img_size: int = 256) -> Callable:
    """Return a torchvision transform pipeline for training images.

    Args:
        img_size: expected image size (images are already 256x256 in your dataset)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform(img_size: int = 256) -> Callable:
    """Return transform pipeline for validation/test images."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
