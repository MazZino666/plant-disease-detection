"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explainability.
Highlights the leaf regions and lesions that trigger the model's prediction.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    Hooks into activations and gradients of the specified convolutional layer.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_gradcam_target_layer"):
                self.target_layer = model.get_gradcam_target_layer()
            else:
                # Fallback: search for the last Conv2d layer
                last_conv = None
                for module in model.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                if last_conv is None:
                    raise ValueError("No Conv2d layer found in the model for Grad-CAM.")
                self.target_layer = last_conv
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple containing the gradient w.r.t the layer's output
            self.gradients = grad_out[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for an input tensor (1, C, H, W).
        Returns:
            heatmap: 2D numpy array [H, W] normalized in [0, 1]
            predicted_class: int
            confidence: float (softmax probability)
        """
        self.model.zero_grad()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Forward pass
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = torch.argmax(probabilities, dim=1).item()

        confidence = probabilities[0, target_class].item()

        # Backward pass for the target class score
        score = logits[0, target_class]
        score.backward()

        # Channel-wise pooling of gradients
        # gradients shape: (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)
        cam = F.relu(cam)

        # Upsample to input image size
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class, confidence

    def remove_hooks(self):
        """Remove hooks to prevent memory leaks."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def apply_gradcam_overlay(
    image: np.ndarray or Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.
    Args:
        image: np.ndarray (H, W, 3) in uint8 [0, 255] or PIL Image
        heatmap: 2D np.ndarray (H, W) in [0, 1]
        alpha: float blending weight (0.0 to 1.0)
        colormap_name: matplotlib colormap ('jet', 'turbo', 'inferno', etc.)
    Returns:
        overlaid_img: np.ndarray (H, W, 3) in uint8 [0, 255]
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    # Ensure heatmap matches image spatial dimensions
    if heatmap.shape[:2] != image.shape[:2]:
        pil_heat = Image.fromarray((heatmap * 255).astype(np.uint8))
        pil_heat = pil_heat.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
        heatmap = np.array(pil_heat, dtype=np.float32) / 255.0

    cmap = plt.get_cmap(colormap_name)
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Strip alpha -> RGB in [0, 1]
    colored_heatmap = (colored_heatmap * 255).astype(np.float32)

    img_float = image.astype(np.float32)
    overlay = (1.0 - alpha) * img_float + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay
