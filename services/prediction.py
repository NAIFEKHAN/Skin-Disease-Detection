"""
Classification Inference Service for Dermatological Disease Detection.
Phase 4 Implementation.

Features:
- Trained Vision Transformer (ViT-B/16) checkpoint loading.
- Safe missing/incompatible checkpoint handling.
- Input validation and decoding (paths, bytes, numpy arrays).
- Optional DullRazor hair removal preprocessing.
- Full 7-class probability distribution via softmax.
- Top-3 disease category ranking with confidence percentages.
- Algorithmic confidence scoring and prediction uncertainty flagging.
- Clear medical communication: strictly preliminary decision aid over 7 disease categories.
- Fully CPU-compatible inference.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import cv2
import numpy as np
import torch

from models.vit import (
    build_vit_model,
    VisionTransformerClassifier,
    DISEASE_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_DISPLAY_NAMES,
    NUM_CLASSES,
    VIT_MODEL_NAME
)
from services.preprocessing import (
    validate_and_decode_image,
    remove_hair_artefacts,
    preprocess_for_classification
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "vit_model.pth"


class PredictionService:
    """Service class for deep learning skin disease classification inference."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.device = device if device else torch.device("cpu")
        self.model: Optional[VisionTransformerClassifier] = None
        self.is_ready = False
        self.error_message: Optional[str] = None
        self.classes = DISEASE_CLASSES
        self.class_display_names = CLASS_DISPLAY_NAMES

        self._load_model()

    def _load_model(self):
        """Loads trained ViT model weights if available, handling missing checkpoints gracefully."""
        if not self.model_path.exists():
            self.is_ready = False
            self.error_message = (
                f"Vision Transformer model checkpoint not found at: {self.model_path}. "
                "Train the model using training/train_vit.py or the Google Colab notebook."
            )
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Instantiate model architecture (freeze_backbone=False for inference)
            self.model = build_vit_model(pretrained=False, num_classes=NUM_CLASSES, device=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
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
            self.error_message = f"Failed to load ViT model weights from {self.model_path}: {str(e)}"

    def predict_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        apply_hair_removal: bool = False,
        uncertainty_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes end-to-end 7-class disease classification:
        1. Validates and decodes image.
        2. Applies optional DullRazor hair removal.
        3. Preprocesses to 224x224 ImageNet-normalized tensor.
        4. Runs ViT forward pass to compute softmax probabilities.
        5. Identifies top prediction and top-3 ranked differentials.
        6. Flags uncertain predictions and attaches medical scope disclaimer.
        """
        if not self.is_ready:
            raise RuntimeError(self.error_message or "Classification service is not initialized.")

        # 1. Validation & decoding
        rgb = validate_and_decode_image(image_input)

        # 2. Optional hair removal
        if apply_hair_removal:
            rgb = remove_hair_artefacts(rgb)

        # 3. Preprocessing (224x224, ImageNet normalized)
        tensor = preprocess_for_classification(rgb, apply_hair_removal=False, is_training=False)
        tensor = tensor.unsqueeze(0).to(self.device)  # Shape (1, 3, 224, 224)

        # 4. Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_code = IDX_TO_CLASS[pred_idx]
        pred_name = CLASS_DISPLAY_NAMES[pred_code]

        # 5. Class-wise probability breakdown
        class_probabilities = {}
        for idx, code in enumerate(self.classes):
            class_probabilities[code] = {
                "display_name": self.class_display_names[code],
                "probability": round(float(probs[idx]), 4),
                "percentage": f"{probs[idx] * 100:.1f}%"
            }

        # 6. Top-3 differential ranking
        sorted_indices = np.argsort(probs)[::-1][:3]
        top_3 = []
        for rank, s_idx in enumerate(sorted_indices, start=1):
            s_code = IDX_TO_CLASS[s_idx]
            top_3.append({
                "rank": rank,
                "class_code": s_code.upper(),
                "display_name": self.class_display_names[s_code],
                "probability": round(float(probs[s_idx]), 4),
                "percentage": f"{probs[s_idx] * 100:.1f}%"
            })

        # 7. Uncertainty check
        is_uncertain = confidence < uncertainty_threshold
        uncertainty_note = (
            "Confidence is below 50%; prediction exhibits elevated ambiguity across categories."
            if is_uncertain else "Model prediction exhibits high relative confidence."
        )

        return {
            "success": True,
            "predicted_class_code": pred_code.upper(),
            "predicted_disease_name": pred_name,
            "confidence_score": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "is_uncertain": is_uncertain,
            "uncertainty_note": uncertainty_note,
            "top_3_predictions": top_3,
            "class_probabilities": class_probabilities,
            "clinical_scope_note": (
                "Model score reflects algorithmic classification across 7 defined dermoscopic lesion categories "
                "(MEL, NV, BCC, AKIEC, BKL, DF, VASC) and is NOT a definitive clinical diagnosis. "
                "Always consult a qualified dermatologist."
            )
        }


# Global singleton instance
_prediction_service_instance: Optional[PredictionService] = None


def get_prediction_service(
    model_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False
) -> PredictionService:
    """Returns singleton instance of PredictionService."""
    global _prediction_service_instance
    if _prediction_service_instance is None or model_path is not None or force_reload:
        _prediction_service_instance = PredictionService(model_path=model_path)
    return _prediction_service_instance
