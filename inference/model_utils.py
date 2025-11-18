# inference/model_utils.py
import torch
from pathlib import Path
from typing import Tuple
import logging

def load_checkpoint_weights(weights_path: str):
    """
    Load checkpoint and return a state_dict suitable for load_state_dict.
    Supports: plain state_dict, {'state_dict':...}, {'model_state':...}
    """
    cp = torch.load(weights_path, map_location="cpu")
    if isinstance(cp, dict):
        # common keys
        if "state_dict" in cp:
            state = cp["state_dict"]
        elif "model_state" in cp:
            state = cp["model_state"]
        else:
            # could already be state_dict
            state = cp
    else:
        state = cp

    # strip "module." prefix if present (saved with DataParallel)
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state

def load_model(model_constructor, weights_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_constructor()
    state = load_checkpoint_weights(weights_path)
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        # try the common key name 'model_state' inside a nested dict (already handled above)
        logging.warning(f"load_state_dict(strict=False) failed: {e}. Trying partial load.")
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, device
