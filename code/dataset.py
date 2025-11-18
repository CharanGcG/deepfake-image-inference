from typing import Optional, Callable, Tuple, Dict, Any
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform: Optional[Callable] = None, img_size: int = 224, logger=None):
        """Create dataset from CSV.

        Args:
            csv_path: Path to CSV file (must contain 'path' and 'label')
            root_dir: Base directory for image paths in CSV
            transform: Optional callable applied to PIL.Image; must return torch.Tensor
            img_size: Fallback image size when creating synthetic zero images
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain 'path' and 'label' columns")

        self.root_dir = root_dir
        self.img_size = img_size

        # Default transform includes resize + ImageNet normalization if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.transform = transform

        # Reset index for safe integer indexing
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        normalized_path = os.path.normpath(rel_path)
        full_path = os.path.join(self.root_dir, normalized_path)
        if not os.path.isfile(full_path):
            print(f"Warning: image not found: {full_path}")
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        try:
            img = Image.open(full_path).convert("RGB")
            return img
        except Exception as e:
            print(f"Warning: failed to open image {full_path}: {e}")
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_path = row["path"]
        label = int(row["label"] if not pd.isna(row["label"]) else 0)

        img = self._load_image(rel_path)

        try:
            img = self.transform(img)
        except Exception as e:
            print(f"Warning: transform failed for {rel_path}: {e}")
            img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)

        meta = {"path": rel_path, "index": int(idx)}
        return img, label, meta