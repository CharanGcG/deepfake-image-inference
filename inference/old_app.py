# Fix import path so Python can see "code" and "code.models"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # DEEPFAKE-PROJECT/
sys.path.insert(0, str(ROOT))


# inference/app.py
import io
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import base64

# import your model and transforms
from code.models.classifier import DeepfakeModel
from code.transforms import get_val_transform   # assume you copied code/transforms.py -> transforms.py
from inference.model_utils import load_model
from inference.gradcam import GradCAM, apply_colormap_on_image
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS = BASE_DIR / "models" / "best.pth"
OUTPUTS = Path(__file__).resolve().parents[0] / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI()
#app.mount("/static", StaticFiles(directory=str(OUTPUTS)), name="static")
BASE_FRONTEND = BASE_DIR / "inference" / "frontend"
# serve the frontend at /ui (and allow index.html)
app.mount("/ui", StaticFiles(directory=str(BASE_FRONTEND), html=True), name="ui")


# create model constructor callable
def create_model():
    # match the init signature used in classifier.py
    return DeepfakeModel(backbone_name="cvt_13", pretrained=False, num_classes=2)

# load model once
model, device = load_model(create_model, str(WEIGHTS))
transform = get_val_transform(img_size=224)  # your val transform uses ToTensor+Normalize

# set up GradCAM (auto-detect last Conv2d)
gradcam = GradCAM(model)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")
    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # preprocess
    tensor = transform(pil).unsqueeze(0)  # 1,C,H,W

    # forward pass for classification
    model.eval()
    with torch.no_grad():
        out = model(tensor.to(device))
        logits = out
        if isinstance(logits, dict) and 'logits' in logits:
            logits = logits['logits']

        if logits.dim() == 1 or (logits.dim()==2 and logits.size(1)==1):
            score = torch.sigmoid(logits.squeeze()).item()
            label = "real" if score >= 0.5 else "fake"
        else:
            probs = torch.softmax(logits, dim=1)
            score = float(probs[0,1]) if probs.size(1) > 1 else float(probs[0,0])
            label = "real" if score >= 0.5 else "fake"

    # Grad-CAM (requires grad)
    cam_np = gradcam.generate(tensor, class_idx=None, device=device)
    cam_img = apply_colormap_on_image(pil, cam_np, alpha=0.5)

    fname = f"gradcam_{uuid.uuid4().hex[:8]}.png"
    outpath = OUTPUTS / fname
    cam_img.save(outpath)

    buf = io.BytesIO()
    cam_img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    return JSONResponse({
        "label": label,
        "score": float(score),
        "gradcam_url": f"/static/{fname}",
        "gradcam_b64": data_url
    })


@app.get("/")
def root():
    return {"status": "running", "device": str(device)}
