# code/args.py
"""
Centralized argparse configuration.
"""

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training")

    # Paths
    parser.add_argument("--train_csv", type=str, default="dataset/train.csv")
    parser.add_argument("--val_csv", type=str, default="dataset/valid.csv")
    parser.add_argument("--test_csv", type=str, default="dataset/test.csv")
    parser.add_argument("--root_dir", type=str, default="dataset/real_vs_fake/real-vs-fake")
    parser.add_argument("--run_dir", type=str, default="models/version_1")
    # Paths
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint for evaluation")
    parser.add_argument("--img_size", type=int, default=224, help="Resize images to this size")


    # Model
    parser.add_argument("--backbone", type=str, default="cvt_13")
    parser.add_argument("--pretrained", action="store_true")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_head", type=int, default=3)
    parser.add_argument("--epochs_finetune", type=int, default=5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    # Flags
    parser.add_argument("--fresh", action="store_true", help="Start training from scratch")
    parser.add_argument("--do_cam", action="store_true", help="Run GradCAM after training")

    return parser.parse_args()

