# --- code/models/backbones.py ---

import torch.nn as nn
import timm
from code.models.cvt import cvt_13
from code.utils.logger import get_logger
from code.args import get_args


#args = get_args()
#logger = get_logger(args.run_dir, name="train")


def _strip_head(model: nn.Module) -> nn.Module:
    """
    Remove any existing classification head from the backbone.
    """
    if hasattr(model, "head"):
        model.head = nn.Identity()
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    return model


def create_backbone(name: str, pretrained: bool) -> nn.Module:
    """
    Create a CvT-13 backbone or timm model by name.

    Args:
        name: Backbone identifier ("cvt_13" or timm model name)
        pretrained: Whether to load pretrained weights

    Returns:
        nn.Module: Backbone model with no classification head
    """
    name = name.lower()

    # Microsoft CvT-13 or custom fallback
    if name == "cvt_13":
        model = cvt_13(pretrained=pretrained)
        return _strip_head(model)

    # Fall back to timm models
    try:
        model = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        return model
    except Exception as e:
        raise ValueError(f"Unknown backbone '{name}' or failed to create: {e}")