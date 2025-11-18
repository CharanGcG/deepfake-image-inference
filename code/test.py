# test.py
import torch
from torch.utils.data import DataLoader, Subset
from code.args import get_args
from code.dataset import DeepfakeDataset
from code.transforms import get_val_transform
from code.models.classifier import DeepfakeModel
from code.engine.evaluator import evaluate
import torch.nn as nn
from code.utils.logger import get_logger
import os


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Logger
    logger = get_logger(args.run_dir if hasattr(args, "run_dir") else "outputs/test_run", name="test")
    logger.info("========== Deepfake Testing Session Started ==========")
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")

    # Dataset
    logger.info("Loading test dataset...")
    test_ds = DeepfakeDataset(args.test_csv, args.root_dir, transform=get_val_transform(img_size=args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f"Test dataset loaded with {len(test_ds)} samples")

    # Model
    logger.info(f"Building model with backbone={args.backbone}, pretrained=False")
    model = DeepfakeModel(args.backbone, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    logger.info("Model and criterion setup complete")

    # Load checkpoint
    checkpoint_path = args.checkpoint if args.checkpoint else f"{args.run_dir}/best.pth"
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check for correct key
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        logger.info("Loaded model state from checkpoint")
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded model state from checkpoint")
    else:
        logger.error("Checkpoint does not contain 'model_state' or 'model'")
        raise KeyError("Invalid checkpoint format")

    # Evaluation
    logger.info("Starting evaluation on test set...")
    metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test metrics: {metrics}")
    logger.info("========== Testing Finished ==========")
    # GradCAM visualization
    try:
        from code.cam import run_gradcam
        logger.info("Running GradCAM visualization on 100 test images...")

        gradcam_dir = os.path.join(args.run_dir if hasattr(args, "run_dir") else "outputs/test_run", "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)

        # Use first 100 images (or all if <100)
        num_per_class = 50
        total_real = len(test_ds) // 2  # assuming CSV is ordered: first half real, second half fake
        total_fake = len(test_ds) // 2

        # pick first 50 real, and first 50 fake from the second half
        real_indices = list(range(num_per_class))
        fake_indices = list(range(total_real, total_real + num_per_class))
        subset_indices = real_indices + fake_indices

        test_subset = Subset(test_ds, subset_indices)

        run_gradcam(model, test_subset, device, gradcam_dir, 100)
        logger.info(f"GradCAM visualizations saved to {gradcam_dir}")
    except Exception as e:
        logger.error(f"GradCAM failed: {e}")


if __name__ == "__main__":
    main()
