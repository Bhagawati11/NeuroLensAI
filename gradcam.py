"""
Grad-CAM++ for the torchvision EfficientNet-B3 model built in
model_inference.py. Standard Grad-CAM++ formulation (Chattopadhyay et al.
2018) via forward/backward hooks on the final conv feature layer.

This exists so the Streamlit UI and spatial_features pipeline are
self-contained here, rather than depending on copying your training
notebook's Grad-CAM cell. If your notebook's Grad-CAM++ implementation is
already verified against the BRISC segmentation masks (Pointing Game etc.
as you've already done), that one is the source of truth for anything
going in the paper — use THIS one for the live UI/demo path, and ideally
spot-check that the two agree on a few scans before trusting this one
for anything beyond the demo.

Target layer: model.features[-1] — the final conv block output before
pooling, standard choice for EfficientNet-B3 Grad-CAM.
"""

import torch
import torch.nn.functional as F
import numpy as np


class GradCAMPlusPlus:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.features[-1]
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int = None):
        """
        input_tensor: [1, 3, H, W], already preprocessed (same as model_inference._preprocess)
        class_index: which class to explain; defaults to the model's own top prediction

        Returns (heatmap as 2D numpy array resized to input H/W, used class_index)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_index]
        score.backward(retain_graph=True)

        activations = self.activations[0]   # [C, h, w]
        gradients = self.gradients[0]        # [C, h, w]

        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3

        sum_activations = activations.sum(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        eps = 1e-8
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.full_like(alpha_denom, eps))
        alphas = grads_power_2 / alpha_denom

        weights = (alphas * F.relu(gradients)).sum(dim=(1, 2))  # [C]

        cam = (weights[:, None, None] * activations).sum(dim=0)  # [h, w]
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = np.array(
            torch.nn.functional.interpolate(
                torch.tensor(cam)[None, None, :, :], size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]
        )

        return cam_resized, class_index


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """image_rgb: HxWx3 uint8. heatmap: HxW float 0-1. Returns HxWx3 uint8 overlay."""
    import cv2

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (image_rgb * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
    return overlay
