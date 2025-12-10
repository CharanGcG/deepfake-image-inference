# inference/visualize.py
import numpy as np
import cv2
import os
from .config import OUTPUT_DIR

def _normalize(arr):
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def cam_to_heatmap(cam, target_shape):
    """
    cam: numpy array shape (T,) or (H, W)
    target_shape: (H, W) of desired output (log_mel shape)
    returns heatmap (H, W, 3) uint8
    """
    H, W = target_shape
    if cam.ndim == 1:
        # cam along time axis -> expand to freq axis
        cam_2d = cv2.resize(cam.reshape(1, -1), (W, H))
    elif cam.ndim == 2:
        cam_2d = cv2.resize(cam, (W, H))
    else:
        raise ValueError("Unsupported cam dims: " + str(cam.shape))

    cam_norm = _normalize(cam_2d)
    heatmap = cv2.applyColorMap((cam_norm * 255).astype("uint8"), cv2.COLORMAP_JET)
    return heatmap

def overlay_log_mel_with_cam(log_mel, cam, out_path=None, alpha=0.5):
    """
    log_mel: numpy (n_mels, time)
    cam: numpy (T,) or (n_mels, time)
    """
    lm = log_mel
    lm_norm = _normalize(lm)
    lm_img = (lm_norm * 255).astype("uint8")
    lm_rgb = cv2.cvtColor(lm_img, cv2.COLOR_GRAY2BGR)

    H, W = lm_img.shape
    heatmap = cam_to_heatmap(cam, (H, W))

    overlay = cv2.addWeighted(lm_rgb, 1 - alpha, heatmap, alpha, 0)

    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, f"gradcam_{np.random.randint(1e9)}.png")
    cv2.imwrite(out_path, overlay)
    return out_path
