# inference/model_utils.py
import os
import json
import torch
from .config import MODEL_WEIGHTS, MODEL_CONFIG_JSON, DEVICE

def _try_load_json_config(path):
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            cfg = json.load(fh)
        # training main.py expects top-level "model_config" in the passed config file.
        if isinstance(cfg, dict) and "model_config" in cfg:
            return cfg["model_config"]
        # if the file itself is the model_config, return as-is
        return cfg
    except Exception:
        # not a JSON file
        return None

def load_model(model_weights=MODEL_WEIGHTS, model_config_json=MODEL_CONFIG_JSON, device=DEVICE):
    """
    Loads the AASIST Model using the model_config (d_args) used during training.
    Raises sensible error if model_config cannot be found - you must provide the same config JSON used for training.
    Returns: model (in eval mode), raw_checkpoint (torch.load output)
    """
    # import the Model class from aasist.AASIST
    try:
        from aasist.models.AASIST import Model as AASISTModel
    except Exception as e:
        raise ImportError("Could not import Model from aasist.AASIST. Ensure aasist/AASIST.py exists and is on PYTHONPATH.") from e

    # load model_config
    model_config = _try_load_json_config(model_config_json)
    if model_config is None:
        raise RuntimeError(
            f"Could not parse model config JSON at {model_config_json}. "
            "For correct architecture you must provide the same training config JSON (the file passed to main.py --config)."
        )

    # instantiate model
    device_t = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = AASISTModel(model_config).to(device_t)

    # load checkpoint
    ckpt = torch.load(model_weights, map_location=device_t)
    # handle common checkpoint layouts
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
        sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
    else:
        sd = ckpt

    try:
        model.load_state_dict(sd)
    except RuntimeError:
        # fallback: non-strict load
        model.load_state_dict(sd, strict=False)

    model.eval()
    return model, ckpt
