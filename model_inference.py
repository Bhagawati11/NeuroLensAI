"""
Loads your trained PyTorch EfficientNet-B3 checkpoint and runs the full
calibrated + deferral-gated inference path, producing the exact JSON
contract your team agreed on day 1.

ASSUMES (matches your locked architecture — adjust if your actual training
code differs):
  - torchvision's efficientnet_b3 backbone, classifier head replaced with a
    single Linear(in_features, 4) for the 4 BRISC classes
  - checkpoint saved as a plain state_dict (torch.save(model.state_dict(), ...))
  - Temperature Scaling applied as a single learned scalar T dividing the
    logits before softmax (Guo et al. 2017)
  - MC Dropout: existing nn.Dropout layers in the backbone switched to
    train() mode at inference time for N stochastic forward passes, while
    BatchNorm stays in eval() mode (critical — BN in train mode would use
    batch statistics instead of running stats and corrupt single-image
    inference)

If your actual training notebook built the model differently (different
backbone wrapper, timm instead of torchvision, a custom head with more
layers, dropout added manually rather than EfficientNet-B3's built-in
dropout), swap build_model() below for your real model-construction code —
everything downstream (predict_single_image) only needs the model to accept
a [1, 3, 300, 300] tensor and return [1, 4] logits.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from config import (
    CLASS_NAMES, CHECKPOINT_PATH, TEMPERATURE, THETA_STAR,HF_MODEL_REPO,
    HF_MODEL_FILENAME,
    MC_DROPOUT_PASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)

from huggingface_hub import hf_hub_download

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(num_classes: int = 4) -> nn.Module:
    model = models.efficientnet_b3(weights=None)  # weights=None: we're loading your fine-tuned checkpoint, not ImageNet weights
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: str = CHECKPOINT_PATH) -> nn.Module:
    # Use the local checkpoint if it exists.
    # Otherwise, download it from the public Hugging Face repository.
    if os.path.isfile(checkpoint_path):
        resolved_checkpoint = checkpoint_path
    else:
        resolved_checkpoint = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILENAME,
            repo_type="model",
        )

    model = build_model(num_classes=len(CLASS_NAMES))

    state_dict = torch.load(
        resolved_checkpoint,
        map_location=_device,
        weights_only=True,
    )

    # Support either a raw state_dict or a wrapped state_dict.
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()
    return model


def _enable_mc_dropout(model: nn.Module):
    """Switches only Dropout layers to train() mode; everything else (BatchNorm
    included) stays in eval() mode. This is what makes repeated forward passes
    stochastic without corrupting BatchNorm statistics."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _mc_dropout_forward(model: nn.Module, input_tensor: torch.Tensor, n_passes: int = MC_DROPOUT_PASSES):
    model.eval()
    _enable_mc_dropout(model)

    probs_list = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(input_tensor)
            calibrated_logits = logits / TEMPERATURE  # Temperature Scaling applied before softmax
            probs = F.softmax(calibrated_logits, dim=1)
            probs_list.append(probs)

    model.eval()  # reset Dropout back to eval() for any later non-MC use
    return torch.stack(probs_list, dim=0)  # [n_passes, 1, num_classes]


def _shannon_entropy(mean_probs: torch.Tensor) -> float:
    eps = 1e-12
    entropy = -torch.sum(mean_probs * torch.log2(mean_probs + eps))
    return float(entropy.item())


def predict_single_image(model: nn.Module, image_path: str, scan_id: str | None = None) -> dict:
    scan_id = scan_id or image_path.split("/")[-1].rsplit(".", 1)[0]

    image = Image.open(image_path)
    input_tensor = _preprocess(image).unsqueeze(0).to(_device)

    all_probs = _mc_dropout_forward(model, input_tensor)       # [N, 1, 4]
    mean_probs = all_probs.mean(dim=0).squeeze(0)               # [4]

    predictive_entropy = _shannon_entropy(mean_probs)
    predicted_idx = int(torch.argmax(mean_probs).item())
    predicted_class = CLASS_NAMES[predicted_idx]
    calibrated_confidence = float(mean_probs[predicted_idx].item())

    if predictive_entropy <= THETA_STAR:
        spatial_features = None
        try:
            from gradcam import GradCAMPlusPlus
            from spatial_features import extract_spatial_features

            cam = GradCAMPlusPlus(model)
            heatmap, _ = cam.generate(input_tensor, class_index=predicted_idx)
            spatial_features = extract_spatial_features(heatmap, predicted_class=predicted_class)
        except Exception as e:
            # Grad-CAM/spatial-feature extraction failing should never take down
            # a classification result — log it and fall back to null, same as
            # when teammate 3's own pipeline hasn't run yet.
            print(f"[spatial_features] extraction failed for {scan_id}: {e}")

        return {
            "scan_id": scan_id,
            "status": "ACCEPTED",
            "predicted_class": predicted_class,
            "calibrated_confidence": round(calibrated_confidence, 4),
            "predictive_entropy": round(predictive_entropy, 4),
            "spatial_features": spatial_features,
        }
    else:
        top2_idx = torch.topk(mean_probs, k=2).indices.tolist()
        top_classes = [CLASS_NAMES[i] for i in top2_idx]
        return {
            "scan_id": scan_id,
            "status": "DEFERRED",
            "predictive_entropy": round(predictive_entropy, 4),
            "top_classes": top_classes,
            "message": f"Predictive entropy exceeded validation-tuned threshold theta*={THETA_STAR}; deferred for clinician review.",
        }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python model_inference.py <path_to_mri_image>")
        sys.exit(1)

    model = load_model()
    result = predict_single_image(model, sys.argv[1])
    print(json.dumps(result, indent=2))
