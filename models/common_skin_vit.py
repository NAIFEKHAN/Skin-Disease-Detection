"""
Vision Transformer (ViT-B/16) Architecture for 9-Class Common Skin Disease Classification.
Clinical Photography Classification Model.

Specifications:
- Architecture: Vision Transformer Base (ViT-B/16)
- Model Identifier: 'vit_base_patch16_224' (timm)
- Pretrained Weights: ImageNet-21k fine-tuned on ImageNet-1k
- Input Resolution: 3 x 224 x 224
- Patch Size: 16 x 16 (196 visual tokens + 1 [CLS] token)
- Embedding Dimension: 768, Depth: 12 Transformer Blocks, Heads: 12
- Output: 9-class linear classification head
- Preprocessing: Clinical photography preprocessing (Resize + Center Crop, ImageNet Normalization)
  Strictly avoids dermoscopic-specific hair removal filters.
- Execution: Fully CPU and CUDA GPU compatible
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# Default model configuration
COMMON_SKIN_MODEL_NAME = "vit_base_patch16_224"
INPUT_IMAGE_SIZE = (224, 224)
NUM_COMMON_CLASSES = 9

COMMON_SKIN_CLASSES = [
    "acne",
    "allergic_contact_dermatitis",
    "eczema",
    "impetigo",
    "psoriasis",
    "ringworm",
    "scabies",
    "seborrheic_dermatitis",
    "urticaria"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(COMMON_SKIN_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(COMMON_SKIN_CLASSES)}

CLASS_DISPLAY_NAMES = {
    "acne": "Acne",
    "allergic_contact_dermatitis": "Allergic Contact Dermatitis",
    "eczema": "Eczema",
    "impetigo": "Impetigo",
    "psoriasis": "Psoriasis",
    "ringworm": "Ringworm (Tinea Corporis)",
    "scabies": "Scabies",
    "seborrheic_dermatitis": "Seborrheic Dermatitis",
    "urticaria": "Urticaria (Hives)"
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CommonSkinViTClassifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) for 9-class clinical common skin condition classification.
    """
    def __init__(
        self,
        num_classes: int = NUM_COMMON_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_blocks: int = 2,
        drop_rate: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = COMMON_SKIN_MODEL_NAME

        # Load pretrained ViT-B/16 from timm
        self.vit = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )

        if freeze_backbone:
            self._apply_freezing(unfreeze_blocks=unfreeze_blocks)

    def _apply_freezing(self, unfreeze_blocks: int = 2):
        """
        Freezes patch embedding and lower transformer blocks to prevent overfitting,
        while keeping the top N blocks and classification head trainable.
        """
        if hasattr(self.vit, "patch_embed"):
            for p in self.vit.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            self.vit.cls_token.requires_grad = False
        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
            self.vit.pos_embed.requires_grad = False

        total_blocks = len(self.vit.blocks)
        freeze_until = max(0, total_blocks - unfreeze_blocks)

        for i, block in enumerate(self.vit.blocks):
            requires_grad = (i >= freeze_until)
            for p in block.parameters():
                p.requires_grad = requires_grad

        if hasattr(self.vit, "norm"):
            for p in self.vit.norm.parameters():
                p.requires_grad = True
        if hasattr(self.vit, "head"):
            for p in self.vit.head.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        return self.vit(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax for inference probability distribution."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns list of parameters that require gradient updates."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        """Counts total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen
        }


def get_clinical_transforms(is_training: bool = False) -> transforms.Compose:
    """
    Transforms tailored for ordinary clinical photographs.
    Avoids aggressive dermoscopic-specific artifact filters.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def load_common_skin_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    eval_mode: bool = True
) -> Tuple[CommonSkinViTClassifier, Dict[str, Any]]:
    """
    Loads saved Common Skin ViT model checkpoint and returns model instance and metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", NUM_COMMON_CLASSES)
    model = CommonSkinViTClassifier(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)

    if eval_mode:
        model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "best_macro_f1": checkpoint.get("best_macro_f1", 0.0),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "classes": checkpoint.get("classes", COMMON_SKIN_CLASSES),
        "class_to_idx": checkpoint.get("class_to_idx", CLASS_TO_IDX),
        "idx_to_class": checkpoint.get("idx_to_class", IDX_TO_CLASS)
    }

    return model, metadata


def save_common_skin_checkpoint(
    model: CommonSkinViTClassifier,
    save_path: Union[str, Path],
    epoch: int = 0,
    best_macro_f1: float = 0.0,
    val_metrics: Optional[Dict[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    smoke_test: bool = False
):
    """
    Saves trained model state, metadata, and optimizer state to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_architecture": COMMON_SKIN_MODEL_NAME,
        "num_classes": model.num_classes,
        "classes": COMMON_SKIN_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "input_resolution": INPUT_IMAGE_SIZE,
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "smoke_test": smoke_test,
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
        "val_metrics": val_metrics or {},
        "model_state_dict": model.state_dict()
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path} (Macro F1: {best_macro_f1:.4f})")
