# inference/predict.py
import argparse
import torch
import torch.nn.functional as F
from .preprocess import load_audio_as_tensor
from .audio_model_utils import load_model
from .gradcam_audio import AudioGradCAM
from .visualize import overlay_log_mel_with_cam
from .config import OUTPUT_DIR, DEVICE
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True, help="Path to audio file (flac/wav)")
    parser.add_argument("--out", "-o", default=OUTPUT_DIR)
    parser.add_argument("--target_idx", type=int, default=1, help="class index to explain (default 1 => fake)")
    args = parser.parse_args()

    model, ckpt = load_model()
    model.to(DEVICE)
    model.eval()

    wav, log_mel = load_audio_as_tensor(args.file)
    # model.forward expects input x as tensor (B, T), we already return (1, T)
    wav = wav.to(DEVICE)

    # forward pass to get score
    with torch.no_grad():
        last_hidden, out = model(wav)
    # out shape (1,2) -> softmax -> fake prob = probs[:,1]
    probs = F.softmax(out, dim=1)
    fake_prob = float(probs[:, 1].cpu().item())
    label = "real" if fake_prob > 0.5 else "fake"
    print(f"Label: {label}  Score(fake prob): {fake_prob:.6f}")

    # Grad-CAM
    agc = AudioGradCAM(model)
    cam = agc.compute_cam(wav, target_index=args.target_idx)

    base = os.path.basename(args.file)
    out_img = os.path.join(args.out, base + "_gradcam.png")
    os.makedirs(args.out, exist_ok=True)
    out_path = overlay_log_mel_with_cam(log_mel, cam, out_path=out_img)
    print("Saved gradcam image ->", out_path)

if __name__ == "__main__":
    main()
