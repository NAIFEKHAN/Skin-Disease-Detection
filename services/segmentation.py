"""
Lesion Segmentation Inference Service.
Phase 3 Implementation.

Features:
- Safe U-Net model loading with graceful missing/incompatible weight handling.
- Input validation and decoding (paths, bytes, numpy arrays).
- Optional DullRazor hair removal preprocessing.
- Binary segmentation prediction and mask resizing to original input dimensions.
- Transparent lesion overlay generation with boundary contour delineation.
- Relative lesion area calculation as a percentage of total image area.
- Explicit labelling of lesion area as relative coverage (not clinical severity).
- Fully CPU-compatible inference.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import cv2
import numpy as np
import torch

from models.unet import build_unet, ResNet18UNet
from services.preprocessing import (
    validate_and_decode_image,
    remove_hair_artefacts,
    preprocess_for_segmentation
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "unet_model.pth"


class SegmentationService:
    """Service class for deep learning lesion segmentation inference."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        default_threshold: float = 0.70
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.device = device if device else torch.device("cpu")
        self.default_threshold = default_threshold
        self.model: Optional[ResNet18UNet] = None
        self.is_ready = False
        self.error_message: Optional[str] = None
        self.optimal_threshold = default_threshold

        self._load_model()

    def _load_model(self):
        """Loads model weights if available, handling missing or incompatible checkpoints."""
        if not self.model_path.exists():
            self.is_ready = False
            self.error_message = (
                f"Model checkpoint not found at: {self.model_path}. "
                "Train the U-Net model using training/train_unet.py or the Google Colab notebook."
            )
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Instantiate model architecture
            self.model = build_unet(pretrained=False, device=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimal_threshold = checkpoint.get("optimal_threshold", self.default_threshold)
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                raise ValueError("Unrecognized checkpoint format.")

            self.model.eval()
            self.is_ready = True
            self.error_message = None

        except Exception as e:
            self.model = None
            self.is_ready = False
            self.error_message = f"Failed to load model weights from {self.model_path}: {str(e)}"

    def segment_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        apply_hair_removal: bool = False,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end segmentation on a dermoscopic image:
        1. Validates and decodes image.
        2. Applies optional DullRazor hair removal.
        3. Preprocesses to 256x256 normalized tensor.
        4. Runs U-Net forward pass.
        5. Computes probability map and binary mask.
        6. Resizes mask back to original input dimensions.
        7. Computes relative lesion area percentage.
        8. Generates transparent green overlay with contour.
        """
        if not self.is_ready:
            raise RuntimeError(self.error_message or "Segmentation service is not initialized.")

        # 1. Validation & decoding
        rgb = validate_and_decode_image(image_input)
        orig_h, orig_w = rgb.shape[:2]

        # 2. Optional hair removal
        if apply_hair_removal:
            rgb = remove_hair_artefacts(rgb)

        # 3. Preprocessing (consistent with training: 256x256, ImageNet normalized)
        tensor = preprocess_for_segmentation(rgb, mask_input=None, is_training=False)
        tensor = tensor.unsqueeze(0).to(self.device)  # Shape (1, 3, 256, 256)

        # 4. Inference
        th = threshold if threshold is not None else self.optimal_threshold
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (256, 256)
            binary_256 = (probs >= th).astype(np.uint8)

        # 5. Resize mask back to original image dimensions (nearest-neighbor)
        mask_orig = cv2.resize(binary_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 6. Calculate relative lesion area percentage
        lesion_pixel_count = int((mask_orig > 0).sum())
        total_pixel_count = orig_h * orig_w
        relative_area_pct = (lesion_pixel_count / total_pixel_count) * 100.0

        # 7. Generate transparent lesion overlay
        overlay = rgb.copy()
        green_tint = np.zeros_like(rgb)
        green_tint[:, :, 1] = 230  # Soft green highlight
        mask_bool = mask_orig > 0

        if mask_bool.any():
            overlay[mask_bool] = cv2.addWeighted(rgb[mask_bool], 0.65, green_tint[mask_bool], 0.35, 0)
            contours, _ = cv2.findContours(mask_orig * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (45, 106, 79), 2)  # Forest green boundary

        return {
            "success": True,
            "relative_lesion_area_pct": round(relative_area_pct, 2),
            "area_coverage_note": "Relative image area coverage (not physical size or clinical severity)",
            "lesion_pixels": lesion_pixel_count,
            "total_pixels": total_pixel_count,
            "threshold_used": th,
            "original_dimensions": [orig_h, orig_w],
            "binary_mask": (mask_orig * 255).astype(np.uint8),
            "overlay_image": overlay
        }


# Global singleton instance
_service_instance: Optional[SegmentationService] = None


def get_segmentation_service(
    model_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False
) -> SegmentationService:
    """Returns singleton instance of SegmentationService."""
    global _service_instance
    if _service_instance is None or model_path is not None or force_reload:
        _service_instance = SegmentationService(model_path=model_path)
    return _service_instance
