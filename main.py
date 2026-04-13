import argparse
from pathlib import Path

import torch

from utils.dataset import create_dataloaders
from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.gradcam import generate_gradcam
from utils.config import CHECKPOINT_PATH, CLASS_NAMES
from models.efficientnet_b0 import PneumoniaClassifier


def run_train():
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = PneumoniaClassifier(num_classes=len(class_names))
    model.to(device)

    # Train and save best model to logs/best_model.pth
    train_model(model, train_loader, val_loader, device)


def run_test():
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = PneumoniaClassifier(num_classes=len(class_names))
    model.to(device)

    # Load best weights
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: {CHECKPOINT_PATH} not found. Train first with --mode train.")
        return

    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("=== Test set performance ===")
    evaluate_model(model, test_loader, device)


def run_gradcam(image_path: str):
    # Just call the helper – it loads the trained model internally
    generate_gradcam(image_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test", "gradcam"],
        help="What to run: train | test | gradcam",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image for Grad-CAM (required if mode=gradcam)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_train()

    elif args.mode == "test":
        run_test()

    elif args.mode == "gradcam":
        if not args.image:
            print("Error: please provide an image path with --image")
            return
        run_gradcam(args.image)


if __name__ == "__main__":
    main()
