"""
EfficientNet-B2 Architecture for 9-Class Common Skin Disease Classification.
Convolutional Neural Network alternative to Vision Transformer.

Specifications:
- Architecture: EfficientNet-B2 ('efficientnet_b2' via timm / torchvision)
- Pretrained Weights: ImageNet-1k pretrained weights
- Input Resolution: 3 x 260 x 260
- Total Parameters: ~7.7M (over 11x smaller than ViT-B/16's 85.8M)
- Strong inductive bias for small-to-medium clinical datasets (1,000–2,000 images)
- Aspect-Ratio-Preserving Preprocessing: Scale to max dim 260 + neutral symmetric padding
- Moderate Training Augmentations: Horizontal/vertical flips, minor rotation (±15°), mild color jitter (0.1)
- Supports Gradual Backbone Unfreezing & Discriminative Learning Rates
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import timm

EFFICIENTNET_MODEL_NAME = "efficientnet_b2"
INPUT_IMAGE_SIZE = (260, 260)
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


class AspectRatioPreservingResizeAndPad:
    """
    Resizes image maintaining aspect ratio and symmetrically pads to square target size.
    Prevents cropping out peripheral rash patterns and lesion borders.
    """
    def __init__(self, target_size: int = 260, fill_color: Tuple[int, int, int] = (124, 116, 104)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        return TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill_color)


def get_efficientnet_transforms(is_training: bool = False, target_size: int = 260) -> transforms.Compose:
    """
    Constructs preprocessing and augmentation pipeline for EfficientNet-B2.
    Uses aspect-ratio-preserving padding and avoids aggressive color/geometric alterations.
    """
    if is_training:
        return transforms.Compose([
            AspectRatioPreservingResizeAndPad(target_size=target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            AspectRatioPreservingResizeAndPad(target_size=target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class CommonSkinEfficientNetB2(nn.Module):
    """
    EfficientNet-B2 Classifier for 9-Class Clinical Common Skin Conditions.
    """
    def __init__(
        self,
        num_classes: int = NUM_COMMON_CLASSES,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes

        # Create backbone via timm
        self.model = timm.create_model(
            EFFICIENTNET_MODEL_NAME,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Computes softmax probabilities across 9 classes."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def freeze_backbone(self):
        """Freezes all feature extraction layers, leaving only classifier head trainable."""
        for name, param in self.model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_top_blocks(self):
        """Unfreezes top convolutional stages (blocks 5, 6 and head convs)."""
        for name, param in self.model.named_parameters():
            if any(block_id in name for block_id in ["blocks.5", "blocks.6", "conv_head", "bn2", "classifier"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes entire network for end-to-end discriminative fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_discriminative_param_groups(
        self,
        lr_head: float = 3e-4,
        lr_backbone_upper: float = 5e-5,
        lr_backbone_lower: float = 1e-5,
        weight_decay: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Creates parameter groups with discriminative learning rates.
        Lower layers adapt gently; head trains with higher rate.
        """
        head_params = []
        upper_params = []
        lower_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name or "fc" in name:
                head_params.append(param)
            elif any(b in name for b in ["blocks.5", "blocks.6", "conv_head", "bn2"]):
                upper_params.append(param)
            else:
                lower_params.append(param)

        groups = []
        if lower_params:
            groups.append({"params": lower_params, "lr": lr_backbone_lower, "weight_decay": weight_decay})
        if upper_params:
            groups.append({"params": upper_params, "lr": lr_backbone_upper, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

        return groups

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen
        }


def save_efficientnet_checkpoint(
    model: CommonSkinEfficientNetB2,
    save_path: Union[str, Path],
    epoch: int = 0,
    best_macro_f1: float = 0.0,
    val_metrics: Optional[Dict[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    smoke_test: bool = False
):
    """Saves EfficientNet-B2 checkpoint and full metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_architecture": EFFICIENTNET_MODEL_NAME,
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
    print(f"Checkpoint successfully saved to: {save_path} (Macro F1: {best_macro_f1:.4f})")


def load_efficientnet_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    eval_mode: bool = True
) -> Tuple[CommonSkinEfficientNetB2, Dict[str, Any]]:
    """Loads EfficientNet-B2 model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", NUM_COMMON_CLASSES)
    model = CommonSkinEfficientNetB2(
        num_classes=num_classes,
        pretrained=False
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
