import os
import torch
import numpy as np
from torchvision.utils import save_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from code.models.classifier import DeepfakeModel


def run_gradcam(model: DeepfakeModel, dataset, device: str, run_dir: str, num_samples: int = 10):
    model.eval().to(device)

    # Detect appropriate GradCAM layer
    if hasattr(model.backbone, 'model'):  # MicrosoftCvTBackbone
        # Use last stage's patch_embed.proj (4D output required for GradCAM)
        target_layers = [model.backbone.model.stage2.patch_embed.proj]
    elif hasattr(model.backbone, 'embed') and hasattr(model.backbone.embed, 'proj'):
        # Custom CvT
        target_layers = [model.backbone.embed.proj]
    else:
        # Fallback
        target_layers = [model.backbone]

    # Ensure gradients flow for GradCAM
    for p in model.parameters():
        p.requires_grad = True

    cam = GradCAM(model=model, target_layers=target_layers)

    out_dir = os.path.join(run_dir, "cam")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(min(num_samples, len(dataset))):
        img, label, meta = dataset[i]
        input_tensor = img.unsqueeze(0).to(device)

        # Forward and backward hooks handled internally by GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(label)])
        grayscale_cam = grayscale_cam[0, :]

        # Convert tensor to numpy image for visualization
        rgb_img = img.cpu().numpy().transpose(1, 2, 0)
        # Undo ImageNet normalization if used
        rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb_img = np.clip(rgb_img, 0, 1)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        safe_path = meta['path'].replace('/', '_').replace('\\', '_')
        save_path = os.path.join(out_dir, f"cam_{i}_{safe_path}.png")

        save_image(torch.tensor(cam_image).permute(2, 0, 1).float() / 255.0, save_path)
        print(f"Saved GradCAM image: {save_path}")