# utils/gradcam.py  (no external pytorch_grad_cam dependency)

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

from models.efficientnet_b0 import PneumoniaClassifier


# ---------- simple Grad-CAM implementation ----------

class SimpleGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # hooks to grab activations and gradients
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple; we need the gradient wrt the output
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_backward_hook(bwd_hook)

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None):
        """
        x: input tensor of shape [1, C, H, W]
        returns: cam numpy array in [0,1] of shape [H, W]
        """
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # scalar score for chosen class
        score = logits[:, class_idx]
        score.backward()

        # GAP over gradients => channel weights
        # gradients: [1, C, h, w]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # upsample CAM to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # normalize to [0,1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


# ---------- image preprocessing ----------

def get_transform():
    # close enough to EfficientNet / ImageNet preprocessing
    return T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),  # make sure we always have 3 channels
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image(image_path: str):
    pil_img = Image.open(image_path).convert("L")  # X-rays are grayscale
    pil_rgb = pil_img.convert("RGB")

    # for overlay (unnormalised)
    rgb_np = np.array(pil_rgb.resize((224, 224))).astype(np.float32) / 255.0

    transform = get_transform()
    tensor = transform(pil_img).unsqueeze(0)  # [1,3,224,224]

    return rgb_np, tensor


# ---------- load trained model ----------

def _load_trained_model(device: torch.device):
    """
    Loads the best trained EfficientNet model from disk.
    Tries logs/best_model.pth first, then models/best_model.pth.
    """
    model = PneumoniaClassifier(num_classes=2)
    model.to(device)

    # try both locations
    candidates = [
        os.path.join("logs", "best_model.pth"),
        os.path.join("models", "best_model.pth"),
    ]

    weights_path = None
    for p in candidates:
        if os.path.exists(p):
            weights_path = p
            break

    if weights_path is None:
        raise FileNotFoundError(
            "Could not find best_model.pth in 'logs' or 'models'. "
            "Run training first, or move the file into one of those folders."
        )

    state = torch.load(weights_path, map_location=device)

    # allow for different saving formats
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def generate_gradcam(
    image_path: str,
    output_dir: str = "logs/gradcam",
    class_idx: Optional[int] = None,
):
    """
    Generates a Grad-CAM heatmap for a single image and saves it as PNG.

    image_path : path to a test X-ray image
    output_dir : folder where the Grad-CAM PNG will be saved
    class_idx  : optional target class (0=Normal, 1=PNEUMONIA). If None, use prediction.
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load trained model
    model = _load_trained_model(device)

    # 2) load + preprocess image
    rgb_img, input_tensor = load_image(image_path)
    input_tensor = input_tensor.to(device)

    # 3) build Grad-CAM on last conv layer
    target_layer = model.backbone.conv_head
    cam_generator = SimpleGradCAM(model, target_layer)

    # 4) make CAM
    cam = cam_generator.generate(input_tensor, class_idx=class_idx)  # [H,W] in [0,1]

    # 5) overlay heatmap on original image
    heatmap = plt.get_cmap("jet")(cam)[:, :, :3]  # [H,W,3] in [0,1]
    overlay = 0.4 * heatmap + 0.6 * rgb_img
    overlay = np.clip(overlay, 0, 1)

    # 6) save image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_gradcam.png")

    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(overlay)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Grad-CAM saved to: {out_path}")
    return out_path
