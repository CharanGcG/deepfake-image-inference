# inference/app.py
"""
Unified Image + Audio inference FastAPI app
- Accepts POST /image-infer and POST /audio-infer (multipart field 'file')
- Serves generated Grad-CAM images from /image-outputs and /audio-outputs
- Mounts a simple frontend directory at /ui (if present)
- Adds CORS middleware so browser-based frontends can call the API
"""

import base64
import io
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import torch
import torch.nn.functional as F

# --- optional pydub for audio conversion ---
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

# ---- adjust root so imports find your project code if running as script ----
ROOT = Path(__file__).resolve().parents[1]  # project root (DEEPFAKE-PROJECT/)
sys.path.insert(0, str(ROOT))

# -------------------------
# Import model utilities
# -------------------------
# Image model imports (adapt if your paths differ)
from code.models.classifier import DeepfakeModel
from code.transforms import get_val_transform
from inference.model_utils import load_model as load_image_model
from inference.gradcam import GradCAM as ImageGradCAM, apply_colormap_on_image

# Audio model imports (adapt if your paths differ)
from inference import config as audio_config
from inference.audio_model_utils import load_model as load_audio_model
from inference.gradcam_audio import GradCAM as AudioGradCAM
from inference.preprocess import load_audio_as_tensor
from inference.visualize import overlay_log_mel_with_cam

# -------------------------
# Constants / output dirs
# -------------------------
IMAGE_OUTPUTS: Path = Path(__file__).resolve().parents[0] / "outputs"  # inference/outputs (image)
AUDIO_OUTPUTS: Path = Path(audio_config.OUTPUT_DIR)  # audio OUTPUT_DIR from audio config
IMAGE_OUTPUTS.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUTS.mkdir(parents=True, exist_ok=True)

# Static directories for UI (optional)
STATIC_DIR: Path = Path(__file__).resolve().parents[0] / "static"
BASE_FRONTEND: Path = STATIC_DIR
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI(title="Unified Image + Audio Inference")

# Development origins - adjust for production to the actual frontend origin(s)
origins = [
    "http://localhost:5173",  # Vite dev
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # for quick testing you may use ["*"], but lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve frontend (if present) at /ui
if BASE_FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(BASE_FRONTEND), html=True), name="ui")

# mount image outputs at /image-outputs and audio outputs at /audio-outputs
app.mount("/image-outputs", StaticFiles(directory=str(IMAGE_OUTPUTS)), name="image_outputs")
app.mount("/audio-outputs", StaticFiles(directory=str(AUDIO_OUTPUTS)), name="audio_outputs")

# mount static (UI assets) at /static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -------------------------
# Load models once at startup
# -------------------------
def create_image_model() -> DeepfakeModel:
    # match the init signature used in classifier.py
    return DeepfakeModel(backbone_name="cvt_13", pretrained=False, num_classes=2)


# image model: returns (model, device)
IMAGE_MODEL, IMAGE_DEVICE = load_image_model(create_image_model, str(ROOT / "models" / "best.pth"))
IMAGE_MODEL.to(IMAGE_DEVICE)
IMAGE_MODEL.eval()

# transform for images
IMAGE_TRANSFORM = get_val_transform(img_size=224)

# image GradCAM helper (auto-detect last conv)
IMAGE_GRADCAM = ImageGradCAM(IMAGE_MODEL)

# audio model: load_model() from audio_model_utils (assumed to return (MODEL, CKPT) or MODEL)
AUDIO_MODEL_OR_TUPLE = load_audio_model()
if isinstance(AUDIO_MODEL_OR_TUPLE, tuple):
    AUDIO_MODEL, _AUDIO_CKPT = AUDIO_MODEL_OR_TUPLE
else:
    AUDIO_MODEL = AUDIO_MODEL_OR_TUPLE

AUDIO_DEVICE = audio_config.DEVICE
AUDIO_MODEL.to(AUDIO_DEVICE)
AUDIO_MODEL.eval()

AUDIO_GRADCAM = AudioGradCAM(AUDIO_MODEL)


# -------------------------
# Utility helpers
# -------------------------
def _build_absolute_url(request: Request, path: str) -> str:
    """Return an absolute URL for a path mounted on this app (path must start with /)."""
    base = str(request.base_url).rstrip("/")
    return base + path


def _convert_to_flac_if_needed(in_path: str) -> str:
    """
    Converts audio to flac if it isn't already a flac.
    Returns path to flac file (could be same as input if already flac).
    """
    _, ext = os.path.splitext(in_path)
    ext = ext.lower()
    if ext == ".flac":
        return in_path

    if not PYDUB_AVAILABLE:
        raise RuntimeError(
            "pydub is required for audio conversion but is not installed. "
            "pip install pydub and ensure ffmpeg is present on the system."
        )

    audio = AudioSegment.from_file(in_path)
    fd, flac_path = tempfile.mkstemp(suffix=".flac")
    os.close(fd)
    audio.export(flac_path, format="flac")
    return flac_path


# -------------------------
# Endpoints
# -------------------------
@app.get("/", response_class=JSONResponse)
def root():
    info = {
        "status": "running",
        "image_device": str(IMAGE_DEVICE),
        "audio_device": str(AUDIO_DEVICE),
        "endpoints": {
            "image_infer": "/image-infer (POST multipart field 'file')",
            "audio_infer": "/audio-infer (POST multipart field 'file')",
            "ui": "/ui",
        },
    }
    return JSONResponse(info)


@app.post("/image-infer")
async def image_infer(request: Request, file: UploadFile = File(...)):
    """
    Accepts multipart upload (field name 'file').
    Returns: { label, score, gradcam_url (absolute), gradcam_b64 (data URL) }
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # preprocess
    tensor = IMAGE_TRANSFORM(pil).unsqueeze(0)  # 1,C,H,W

    # forward pass for classification
    IMAGE_MODEL.eval()
    with torch.no_grad():
        out = IMAGE_MODEL(tensor.to(IMAGE_DEVICE))
        logits = out
        if isinstance(logits, dict) and "logits" in logits:
            logits = logits["logits"]

        # binary vs multi-class handling
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            score = torch.sigmoid(logits.squeeze()).item()
            label = "real" if score >= 0.9 else "fake"
        else:
            probs = torch.softmax(logits, dim=1)
            score = float(probs[0, 1]) if probs.size(1) > 1 else float(probs[0, 0])
            label = "real" if score >= 0.9 else "fake"

    # Grad-CAM (image)
    cam_np = IMAGE_GRADCAM.generate(tensor, class_idx=None, device=IMAGE_DEVICE)
    cam_img = apply_colormap_on_image(pil, cam_np, alpha=0.5)

    fname = f"gradcam_{uuid.uuid4().hex[:8]}.png"
    outpath = IMAGE_OUTPUTS / fname
    cam_img.save(outpath)

    # also return data URL (so frontend can render without separate fetch)
    buf = io.BytesIO()
    cam_img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    gradcam_path = f"/image-outputs/{fname}"
    gradcam_url = _build_absolute_url(request, gradcam_path)

    return JSONResponse(
        {
            "label": label,
            "score": float(score),
            "gradcam_url": gradcam_url,
            "gradcam_b64": data_url,
        }
    )


@app.post("/audio-infer")
async def audio_infer(request: Request, file: UploadFile = File(...)):
    """
    Accepts multipart upload (field name 'file'). Converts to flac if needed, runs audio model.
    Returns: { label, score, gradcam_url (absolute) }
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # allowed extensions
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in [".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # save uploaded file to temp
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    flac_path = tmp_path
    converted_tmp: Optional[str] = None
    try:
        # convert if not flac
        if not tmp_path.lower().endswith(".flac"):
            try:
                flac_path = _convert_to_flac_if_needed(tmp_path)
                converted_tmp = flac_path if flac_path != tmp_path else None
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Conversion to flac failed: {exc}")

        # Preprocess and forward
        wav, log_mel = load_audio_as_tensor(flac_path)
        wav = wav.to(AUDIO_DEVICE)

        with torch.no_grad():
            _, out = AUDIO_MODEL(wav)

        # interpret output: second column is 'fake' logit -> softmax -> probs[:,1]
        probs = F.softmax(out, dim=1)
        fake_prob = float(probs[:, 1].cpu().item())

        # label logic (adjust threshold if desired)
        label = "fake" if fake_prob < 0.5 else "real"

        # Grad-CAM (audio)
        cam = AUDIO_GRADCAM.compute_cam(wav, target_index=1)

        # save overlay image in outputs
        base = os.path.splitext(os.path.basename(file.filename))[0]
        out_img_name = f"{base}_{int(torch.rand(1).item() * 1e9)}_gradcam.png"
        out_img_path = os.path.join(AUDIO_OUTPUTS, out_img_name)
        overlay_log_mel_with_cam(log_mel, cam, out_path=out_img_path)

        # absolute url
        gradcam_path = f"/audio-outputs/{out_img_name}"
        gradcam_url = _build_absolute_url(request, gradcam_path)

        return JSONResponse({"label": label, "score": float(fake_prob), "gradcam_url": gradcam_url})
    finally:
        # cleanup temp files
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            if converted_tmp and os.path.exists(converted_tmp) and converted_tmp != tmp_path:
                os.remove(converted_tmp)
        except Exception:
            pass
