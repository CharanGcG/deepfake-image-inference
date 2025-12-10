# inference/config.py
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
# Path to the training config JSON used to build the model architecture.
# IMPORTANT: set this to the same config JSON you used for training (the one you passed as --config to main.py).
# If you don't have it at root, point to its path here.
MODEL_CONFIG_JSON = os.path.join(ROOT, "aasist", "config", "AASIST-L.conf")  # try this by default (may not be JSON)

# Path to the checkpoint (.pth) to use for inference
MODEL_WEIGHTS = os.path.join(ROOT, "aasist", "models", "weights", "AASIST-L.pth")  # change to AASIST.pth if desired

SAMPLE_RATE = 16000
N_MELS = 128

# device: "cpu" or "cuda"
DEVICE = "cuda" if (os.getenv("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: set name of target module to hook for Grad-CAM.
# If None -> code will auto-find last Conv2d/Conv1d; you can set e.g. "encoder.2.conv2" after you inspect model.named_modules()
TARGET_MODULE_NAME = None
