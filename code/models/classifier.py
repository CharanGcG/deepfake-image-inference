import torch
import torch.nn as nn
from code.models.backbones import create_backbone
from code.utils.logger import get_logger
from code.args import get_args

#args = get_args()
#logger = get_logger(args.run_dir, name="train")


class DeepfakeModel(nn.Module):
    """
    Deepfake detection model with a CvT-13 backbone (or timm fallback) and a classifier.
    """

    def __init__(
        self,
        backbone_name: str = "cvt_13",
        pretrained: bool = True,
        num_classes: int = 2
    ):
        super().__init__()

        # Build backbone
        self.backbone = create_backbone(backbone_name, pretrained=pretrained)

        # Determine feature dimension output by backbone
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            if out.dim() > 2:
                out = out.mean(dim=[2, 3])
            in_features = out.shape[1]

        # Classifier head
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.mean(dim=[2, 3])  # global avg pool
        logits = self.classifier(feats)

        # Ensure output is always 2D [B, num_classes] for GradCAM compatibility
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return backbone features only (for GradCAM visualization).
        """
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.mean(dim=[2, 3])
        return feats

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True