# utils/checkpoint.py
import os
import torch
from typing import Any, Dict

def save_checkpoint(state: Dict[str, Any], is_best: bool, output_dir: str, filename: str = "last.pth") -> str:
    try:
        os.makedirs(output_dir, exist_ok=True)
        last_path = os.path.join(output_dir, filename)
        torch.save(state, last_path)
        if is_best:
            best_path = os.path.join(output_dir, "best.pth")
            torch.save(state, best_path)
        return last_path
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint: {e}")

def load_checkpoint(filepath: str, map_location: str = "cpu") -> Dict[str, Any]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    try:
        ckpt = torch.load(filepath, map_location=map_location)
        # Ensure backward compatibility if scaler is missing
        if "scaler_state" not in ckpt:
            ckpt["scaler_state"] = None
        return ckpt
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
