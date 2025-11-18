# inference/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional
import io

class GradCAM:
    def __init__(self, model, target_module=None):
        """
        If target_module is None, auto-detect the last nn.Conv2d in model.backbone (if present).
        model: the full model (DeepfakeModel instance)
        target_module: module object or module name string (optional)
        """
        self.model = model
        self.activations = None
        self.gradients = None

        # Try user-specified module name
        if isinstance(target_module, str):
            module = dict(self.model.named_modules()).get(target_module, None)
            if module is None:
                raise ValueError(f"Module name {target_module} not found in model")
            self.target = module
        elif target_module is None:
            # auto-detect last Conv2d in backbone
            convs = [(n, m) for n, m in self.model.named_modules() if isinstance(m, torch.nn.Conv2d)]
            if len(convs) == 0:
                raise ValueError("No Conv2d modules found in the model for Grad-CAM")
            # pick last conv
            name, module = convs[-1]
            self.target = module
            self.target_name = name
        else:
            # assume module object passed
            self.target = target_module

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # register hooks
        self.target.register_forward_hook(forward_hook)
        self.target.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int]=None, device=None):
        """
        input_tensor: 1 x C x H x W (on device)
        returns heatmap np.uint8 (H x W)
        """
        if device is None:
            device = next(self.model.parameters()).device

        input_tensor = input_tensor.to(device)
        self.model.zero_grad()
        out = self.model(input_tensor)  # expects logits shape [1, num_classes]

        # Normalize typical output formats:
        logits = out
        if isinstance(logits, dict) and 'logits' in logits:
            logits = logits['logits']

        # choose class index
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            # single-logit binary: take that scalar
            score = logits.squeeze()
        else:
            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=1).item())
            score = logits[0, class_idx]

        # backward to populate gradients on target
        score.backward(retain_graph=True)

        grads = self.gradients[0]  # C x H x W
        acts = self.activations[0]  # C x H x W

        # global average pooling on grads -> weights
        weights = torch.mean(grads.view(grads.size(0), -1), dim=1)  # C
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)
        cam_np = cam.cpu().numpy()
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        cam_np = (cam_np * 255).astype('uint8')
        return cam_np

def apply_colormap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha=0.5):
    org = np.array(pil_img)[:, :, ::-1]  # RGB->BGR for cv2
    heatmap_resized = cv2.resize(heatmap, (org.shape[1], org.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, org, 1 - alpha, 0)
    return Image.fromarray(overlay[:, :, ::-1])
