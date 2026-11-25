"""
Explainable AI (XAI) Service for Vision Transformer (ViT-B/16).
Generates genuine Attention Rollout heatmaps from transformer self-attention layers.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import cv2
import numpy as np
import torch
import torch.nn as nn

from models.vit import build_vit_model, VisionTransformerClassifier, NUM_CLASSES
from services.preprocessing import (
    validate_and_decode_image,
    preprocess_for_classification
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "vit_model.pth"


class AttentionRolloutService:
    """
    Computes genuine Attention Rollout maps (Abnar & Zuidema, 2020) for ViT-B/16.
    Propagates multi-head self-attention matrices recursively through transformer blocks
    to highlight regions contributing most heavily to the classification token.
    """

    def __init__(
        self,
        model: Optional[VisionTransformerClassifier] = None,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device if device else torch.device("cpu")
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model = model
        self.is_ready = False
        self.error_message: Optional[str] = None

        if self.model is None:
            self._load_model()
        else:
            self.is_ready = True

    def _load_model(self):
        """Loads trained ViT model weights."""
        if not self.model_path.exists():
            self.is_ready = False
            self.error_message = f"Model checkpoint not found at: {self.model_path}"
            return

        try:
            self.model = build_vit_model(pretrained=False, num_classes=NUM_CLASSES, device=self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.is_ready = True
            self.error_message = None
        except Exception as e:
            self.model = None
            self.is_ready = False
            self.error_message = f"Failed to initialize ViT model for attention rollout: {e}"

    def generate_attention_map(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        alpha: float = 0.45
    ) -> Dict[str, Any]:
        """
        Generates attention rollout heatmap and overlay for an input image.

        Returns:
            dict containing:
            - success (bool)
            - overlay_rgb (np.ndarray)
            - heatmap_raw (np.ndarray normalized [0, 1])
            - output_path (str or None)
            - disclaimer (str)
        """
        disclaimer = (
            "Model visualization for explainability only. Highlights spatial regions weighted by transformer "
            "attention layers; not clinical evidence, margin boundary or diagnostic confirmation."
        )

        if not self.is_ready or self.model is None:
            return {
                "success": False,
                "error": self.error_message or "Attention service not initialized.",
                "disclaimer": disclaimer
            }

        try:
            # 1. Decode original image for blending
            rgb_orig = validate_and_decode_image(image_input)
            orig_h, orig_w = rgb_orig.shape[:2]

            # 2. Preprocess to 224x224 tensor
            tensor = preprocess_for_classification(rgb_orig, apply_hair_removal=False, is_training=False)
            tensor = tensor.unsqueeze(0).to(self.device)

            # 3. Hook transformer attention layers
            attn_matrices: List[torch.Tensor] = []
            fused_states: List[bool] = []
            hook_handles = []

            vit_backbone = getattr(self.model, "vit", self.model)
            blocks = getattr(vit_backbone, "blocks", [])

            if len(blocks) == 0:
                raise RuntimeError("Unable to locate transformer blocks in model architecture.")

            for block in blocks:
                # Save and temporarily disable fused_attn to expose raw attention matrices
                fused_states.append(getattr(block.attn, "fused_attn", False))
                block.attn.fused_attn = False

                def make_hook():
                    def hook_fn(module, inp, outp):
                        # inp[0] is attention weights after softmax: shape (B, num_heads, N, N)
                        if len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                            attn_matrices.append(inp[0].detach().cpu())
                    return hook_fn

                handle = block.attn.attn_drop.register_forward_hook(make_hook())
                hook_handles.append(handle)

            # 4. Forward pass
            try:
                with torch.no_grad():
                    _ = self.model(tensor)
            finally:
                # Always remove hooks and restore fused_attn states
                for handle in hook_handles:
                    handle.remove()
                for block, state in zip(blocks, fused_states):
                    block.attn.fused_attn = state

            if len(attn_matrices) == 0:
                raise RuntimeError("No attention weights were intercepted during forward pass.")

            # 5. Attention Rollout computation
            # Recursively multiply (0.5 * A_l + 0.5 * I) across layers
            num_tokens = attn_matrices[0].shape[-1]  # 197 tokens (1 CLS + 196 patches)
            identity = torch.eye(num_tokens).unsqueeze(0)  # (1, 197, 197)

            rollout = identity.clone()

            for layer_attn in attn_matrices:
                # Average attention across heads: (B, num_heads, N, N) -> (1, N, N)
                avg_heads = torch.mean(layer_attn, dim=1)  # (1, N, N)
                # Residual connection: add identity and normalize rows
                augmented = 0.5 * avg_heads + 0.5 * identity
                augmented = augmented / augmented.sum(dim=-1, keepdim=True)
                # Matrix multiplication
                rollout = torch.matmul(augmented, rollout)

            # 6. Extract CLS token attention to patch tokens
            # Row 0 is the CLS token; columns 1..196 are the 14x14 patch tokens
            cls_attention = rollout[0, 0, 1:].numpy()  # shape (196,)
            grid_size = int(np.sqrt(cls_attention.shape[0]))  # 14
            heatmap_14x14 = cls_attention.reshape(grid_size, grid_size)

            # Normalize to [0.0, 1.0]
            h_min, h_max = heatmap_14x14.min(), heatmap_14x14.max()
            if h_max > h_min:
                heatmap_norm = (heatmap_14x14 - h_min) / (h_max - h_min)
            else:
                heatmap_norm = np.zeros_like(heatmap_14x14)

            # 7. Resize heatmap to original image dimensions
            heatmap_resized = cv2.resize(heatmap_norm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

            # 8. Colorize heatmap (JET colormap)
            heatmap_uint8 = (heatmap_resized * 255.0).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # 9. Blend heatmap with original image
            blend_rgb = cv2.addWeighted(rgb_orig, (1.0 - alpha), heatmap_rgb, alpha, 0)

            # 10. Save overlay if requested
            saved_path_str = None
            if output_path is not None:
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                blend_bgr = cv2.cvtColor(blend_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_p), blend_bgr)
                saved_path_str = str(out_p)

            return {
                "success": True,
                "overlay_rgb": blend_rgb,
                "heatmap_raw": heatmap_resized,
                "output_path": saved_path_str,
                "grid_size": (grid_size, grid_size),
                "disclaimer": disclaimer
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate attention map: {str(e)}",
                "disclaimer": disclaimer
            }


# Singleton instance
_attention_service_instance: Optional[AttentionRolloutService] = None


def get_attention_service(
    model: Optional[VisionTransformerClassifier] = None,
    model_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False
) -> AttentionRolloutService:
    """Returns singleton instance of AttentionRolloutService."""
    global _attention_service_instance
    if _attention_service_instance is None or force_reload or model is not None:
        _attention_service_instance = AttentionRolloutService(model=model, model_path=model_path)
    return _attention_service_instance
