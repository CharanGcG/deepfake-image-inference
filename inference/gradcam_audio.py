# inference/gradcam_audio.py
import torch
import numpy as np
import torch.nn as nn
from .config import TARGET_MODULE_NAME

def _find_target_module(model, name=None):
    """Find module by dotted name or fallback to last Conv1d/Conv2d."""
    if name:
        try:
            parts = name.split(".")
            m = model
            for p in parts:
                m = getattr(m, p)
            return m
        except Exception:
            pass

    conv_mod = None
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            conv_mod = m
    if conv_mod is None:
        raise RuntimeError("No Conv1d/Conv2d found to hook for Grad-CAM. Set TARGET_MODULE_NAME in config.")
    return conv_mod

class GradCAM:
    """
    Robust Grad-CAM for audio models.  
    Usage:
      gradcam = GradCAM(model, target_module_name)
      cam = gradcam.compute_cam(input_tensor, target_index=1)  # returns numpy array
    """
    def __init__(self, model, target_module_name=None):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.target = _find_target_module(model, target_module_name or TARGET_MODULE_NAME)

        self.activations = None   # activation tensor (saved by forward hook)
        self.gradients = None     # gradients of activation (saved by hook)

        # register forward hook that also attaches a hook to activation tensor to capture gradient
        def forward_hook(module, inp, out):
            # out is a tensor (or tuple); we expect tensor
            self.activations = out.detach()
            # try to register gradient hook on the activation tensor to capture gradients later
            try:
                # out may be a Tensor or tuple; handle tensor
                if isinstance(out, torch.Tensor):
                    # register hook on the forward output tensor so its grad is captured on backward
                    out.register_hook(self._save_grad)
            except Exception:
                # not fatal; we'll fallback to module-level backward hook if available
                pass

        # try module-level backward hook (full hook available in newer PyTorch)
        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; first element is gradient w.r.t. module output
            try:
                if isinstance(grad_out, tuple) and len(grad_out) > 0:
                    g = grad_out[0]
                else:
                    g = grad_out
                if isinstance(g, torch.Tensor):
                    self.gradients = g.detach()
            except Exception:
                pass

        # attach hooks
        self.fwd_handle = self.target.register_forward_hook(forward_hook)
        # try to attach module backward hook if available
        try:
            # use register_full_backward_hook when available (PyTorch >=1.8)
            self.bwd_handle = self.target.register_full_backward_hook(backward_hook)
        except Exception:
            # some versions only have register_backward_hook (deprecated), try that
            try:
                self.bwd_handle = self.target.register_backward_hook(backward_hook)
            except Exception:
                self.bwd_handle = None

    def _save_grad(self, grad):
        """Save gradients from activation tensor hook."""
        self.gradients = grad.detach()

    def generate_cam_from_acts(self):
        """
        Compute CAM from self.activations and self.gradients.
        Returns: cam numpy array with shape equal to spatial dims:
                 - if activations (B,C,H,W) -> cam shape (H,W)
                 - if activations (B,C,T)   -> cam shape (T,)
        """
        if self.activations is None:
            raise RuntimeError("No activations saved. Run forward pass before computing CAM.")
        if self.gradients is None:
            raise RuntimeError("No gradients saved. Run backward pass before computing CAM.")

        acts = self.activations          # tensor
        grads = self.gradients           # tensor

        # ensure grads/acts are on CPU for numpy conversion later (but do ops on tensors)
        # compute channel weights by global average pooling over spatial dims (2..)
        if grads.dim() < 3:
            # unexpected shape
            raise RuntimeError(f"Expected gradients with dim >=3 (B,C,...) but got {grads.shape}")

        spatial_dims = tuple(range(2, grads.dim()))  # dims to pool over
        weights = grads.mean(dim=spatial_dims, keepdim=True)  # shape (B,C,1,1) or (B,C,1)

        # weighted combination of activations
        # acts shape: (B,C,...) -> multiply by weights and sum over channel dim
        cam = (weights * acts).sum(dim=1, keepdim=False)  # (B, H, W) or (B, T)
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()  # (B, ...)
        # normalize each sample individually to [0,1]
        out_cams = []
        for i in range(cam_np.shape[0]):
            arr = cam_np[i]
            mn = arr.min()
            mx = arr.max()
            if mx - mn < 1e-9:
                norm = np.zeros_like(arr)
            else:
                norm = (arr - mn) / (mx - mn)
            out_cams.append(norm)
        out = np.stack(out_cams, axis=0)  # (B, ...)
        # return only first batch element by default (inference uses single sample)
        return out

    def compute_cam(self, input_tensor, target_index=1, return_numpy=True):
        """
        Full convenience method:
          - runs forward on model
          - computes target score (output[:, target_index]) and backward
          - collects activations+gradients via hooks and returns normalized CAM
        Args:
          input_tensor: torch.Tensor of shape expected by model (e.g., (1, T) or (1,1,T))
          target_index: class index to explain (default 1 == fake)
          return_numpy: if True returns numpy array, else torch tensor
        Returns:
          cam (numpy array or torch tensor) for first batch item with shape:
            - (H, W) for 2D activations
            - (T,)   for 1D activations
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # clear previous saved activations/gradients
        self.activations = None
        self.gradients = None

        # forward: model returns (last_hidden, output)
        out_tuple = self.model(input_tensor)
        if isinstance(out_tuple, (tuple, list)) and len(out_tuple) >= 2:
            _last_hidden, output = out_tuple[0], out_tuple[1]
        else:
            # fallback: if model returns a single tensor treat that as output
            if isinstance(out_tuple, torch.Tensor):
                output = out_tuple
            else:
                raise RuntimeError("Unexpected model.forward output. Expected (last_hidden, output) or tensor.")

        # pick target score (sum across batch -> scalar) then backward
        # ensure gradients exist
        target_score = output[:, target_index].sum()
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # now activations and gradients should be populated by hooks
        cams = self.generate_cam_from_acts()  # (B, H, W) or (B, T)
        cam0 = cams[0]

        if return_numpy:
            return cam0
        else:
            return torch.from_numpy(cam0)

    def close(self):
        """Remove hooks if needed."""
        try:
            if hasattr(self, "fwd_handle") and self.fwd_handle is not None:
                self.fwd_handle.remove()
        except Exception:
            pass
        try:
            if hasattr(self, "bwd_handle") and self.bwd_handle is not None:
                self.bwd_handle.remove()
        except Exception:
            pass

    # ensure hooks removed on garbage collection
    def __del__(self):
        self.close()
