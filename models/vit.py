"""
Vision Transformer (ViT-B/16) Architecture for 7-Class Skin Lesion Classification.
Phase 4 Implementation.

Specifications:
- Architecture: Vision Transformer Base (ViT-B/16)
- Model Identifier: 'vit_base_patch16_224' (timm)
- Pretrained Weights: ImageNet-21k fine-tuned on ImageNet-1k (AugReg)
- Input Resolution: 3 x 224 x 224
- Patch Size: 16 x 16 (196 visual tokens + 1 [CLS] token)
- Embedding Dimension: 768, Depth: 12 Transformer Blocks, Heads: 12
- Output: 7-class linear classification head
- Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Transfer Learning: Supports configurable backbone freezing and discriminative fine-tuning
- Execution: Fully CPU and CUDA GPU compatible
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import timm

# Default model configuration
VIT_MODEL_NAME = "vit_base_patch16_224"
INPUT_IMAGE_SIZE = (224, 224)
NUM_CLASSES = 7

# Canonical 7 Disease Classes (Index aligned with trained model weights and dataset manifests)
DISEASE_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASS_TO_IDX = {c: i for i, c in enumerate(DISEASE_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(DISEASE_CLASSES)}
CLASS_DISPLAY_NAMES = {
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis-like Lesions",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "vasc": "Vascular Lesions",
    "df": "Dermatofibroma"
}


class VisionTransformerClassifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) for 7-class dermatological disease classification.
    Wraps timm's vit_base_patch16_224 with configurable transfer learning features.
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_blocks: int = 2,
        drop_rate: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = VIT_MODEL_NAME

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
        Freezes patch embedding and lower transformer blocks to prevent overfitting
        on small medical datasets, while keeping the top N blocks and classification head trainable.
        """
        # Freeze patch embedding and position embeddings
        if hasattr(self.vit, "patch_embed"):
            for p in self.vit.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(self.vit, "cls_token"):
            self.vit.cls_token.requires_grad = False
        if hasattr(self.vit, "pos_embed"):
            self.vit.pos_embed.requires_grad = False

        # ViT has 12 blocks (0 to 11)
        total_blocks = len(self.vit.blocks)
        freeze_until = max(0, total_blocks - unfreeze_blocks)

        for i, block in enumerate(self.vit.blocks):
            requires_grad = (i >= freeze_until)
            for p in block.parameters():
                p.requires_grad = requires_grad

        # Normalization layer before head and head itself remain trainable
        if hasattr(self.vit, "norm"):
            for p in self.vit.norm.parameters():
                p.requires_grad = True
        if hasattr(self.vit, "head"):
            for p in self.vit.head.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (B, num_classes)."""
        return self.vit(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper returning softmax probability distributions of shape (B, num_classes)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference helper returning:
        - predicted_classes: Tensor of predicted class indices (B,)
        - confidences: Tensor of top softmax probabilities (B,)
        - probabilities: Full softmax probability tensor (B, num_classes)
        """
        probs = self.predict_proba(x)
        confs, preds = torch.max(probs, dim=-1)
        return preds, confs, probs

    def get_parameter_count(self) -> Dict[str, int]:
        """Returns total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen
        }

    def get_parameter_groups(self, head_lr: float = 1e-4, backbone_lr: float = 1e-5) -> List[Dict[str, Any]]:
        """
        Returns parameter groups with discriminative learning rates:
        - Higher learning rate for new classification head.
        - Lower learning rate for fine-tuning pretrained transformer backbone.
        """
        head_params = []
        backbone_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": head_lr, "name": "head"}
        ]
        return param_groups


def build_vit_model(
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = False,
    unfreeze_blocks: int = 2,
    device: Optional[torch.device] = None
) -> VisionTransformerClassifier:
    """Factory helper to instantiate the ViT-B/16 classifier."""
    model = VisionTransformerClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        unfreeze_blocks=unfreeze_blocks
    )
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vit_model(pretrained=True, num_classes=7, freeze_backbone=True, unfreeze_blocks=2, device=device)
    p_info = model.get_parameter_count()
    print("=" * 60)
    print(f"ViT-B/16 ({VIT_MODEL_NAME}) initialized on {device}")
    print(f"Total parameters:     {p_info['total_parameters']:,}")
    print(f"Trainable parameters: {p_info['trainable_parameters']:,}")
    print(f"Frozen parameters:    {p_info['frozen_parameters']:,}")
    print("=" * 60)

    # Smoke test forward pass
    dummy = torch.randn(2, 3, 224, 224, device=device)
    logits = model(dummy)
    probs = model.predict_proba(dummy)
    preds, confs, _ = model.predict(dummy)

    print("Logits shape:      ", logits.shape)
    print("Probabilities shape:", probs.shape)
    print("Predictions:       ", preds.tolist())
    print("Confidences:       ", [round(c, 4) for c in confs.tolist()])
    assert logits.shape == (2, 7)
    assert probs.shape == (2, 7)
    print("✔ models/vit.py smoke test passed!")
