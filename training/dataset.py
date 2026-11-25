"""
PyTorch Datasets, DataLoaders, and Training Preparation for:
1. Skin Lesion Classification (Vision Transformer ViT-B/16)
2. Skin Lesion Segmentation (U-Net)

Supports:
- Local and Google Colab environments (configurable data_root).
- Efficient on-the-fly loading and CPU/GPU memory management.
- Class weight computation for imbalanced cross-entropy loss.
- Coordinated augmentations for segmentation image-mask pairs.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from services.preprocessing import (
    preprocess_for_classification,
    preprocess_for_segmentation,
    validate_and_decode_image,
    validate_and_decode_mask
)

# Canonical 7 Disease Classes Mapping (Aligned with manifests and trained model)
CLASS_TO_IDX: Dict[str, int] = {
    "nv": 0,     # Melanocytic Nevi
    "mel": 1,    # Melanoma
    "bkl": 2,    # Benign Keratosis-like Lesions
    "bcc": 3,    # Basal Cell Carcinoma
    "akiec": 4,  # Actinic Keratoses and Intraepithelial Carcinoma
    "vasc": 5,   # Vascular Lesions
    "df": 6      # Dermatofibroma
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis-like Lesions",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "vasc": "Vascular Lesions",
    "df": "Dermatofibroma"
}


def get_default_data_root() -> Path:
    """Returns the dataset root path, honoring DERMA_DATA_ROOT env variable if set."""
    env_root = os.environ.get("DERMA_DATA_ROOT")
    if env_root:
        return Path(env_root)
    # Default to CP-1/data
    return Path(__file__).resolve().parent.parent / "data"


class SkinClassificationDataset(Dataset):
    """
    PyTorch Dataset for 7-class dermatological lesion classification (ViT-B/16).
    Loads images on-the-fly from manifests, applies ViT normalization (224x224),
    and returns (image_tensor, label_int, image_id).
    """
    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        is_training: bool = False,
        apply_hair_removal: bool = False
    ):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else get_default_data_root()
        self.is_training = is_training
        self.apply_hair_removal = apply_hair_removal

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Classification manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        if "label" not in self.df.columns and "dx" in self.df.columns:
            self.df["label"] = self.df["dx"].map(CLASS_TO_IDX)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # Determine image file path
        if "image_path" in row and pd.notna(row["image_path"]):
            rel_path = str(row["image_path"]).replace("\\", "/")
            img_path = self.data_root / rel_path
        else:
            img_path = self.data_root / "classification" / "images" / f"{image_id}.jpg"

        tensor = preprocess_for_classification(
            image_input=img_path,
            apply_hair_removal=self.apply_hair_removal,
            is_training=self.is_training
        )

        label = int(row["label"])
        return tensor, label, image_id


class SkinSegmentationDataset(Dataset):
    """
    PyTorch Dataset for dermatological lesion segmentation (U-Net).
    Loads image and expert mask pairs, applies coordinated resizing (256x256),
    and returns (image_tensor, mask_tensor, image_id).
    """
    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        is_training: bool = False,
        apply_hair_removal: bool = False
    ):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else get_default_data_root()
        self.is_training = is_training
        self.apply_hair_removal = apply_hair_removal

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Segmentation manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        if "image_path" in row and pd.notna(row["image_path"]):
            rel_img = str(row["image_path"]).replace("\\", "/")
            img_path = self.data_root / rel_img
        else:
            img_path = self.data_root / "segmentation" / "images" / f"{image_id}.jpg"

        if "mask_path" in row and pd.notna(row["mask_path"]):
            rel_mask = str(row["mask_path"]).replace("\\", "/")
            mask_path = self.data_root / rel_mask
        else:
            mask_path = self.data_root / "segmentation" / "masks" / f"{image_id}_segmentation.png"

        img_tensor, mask_tensor = preprocess_for_segmentation(
            image_input=img_path,
            mask_input=mask_path,
            apply_hair_removal=self.apply_hair_removal,
            is_training=self.is_training
        )

        return img_tensor, mask_tensor, image_id


def compute_class_weights(
    manifest_path: Union[str, Path],
    num_classes: int = 7
) -> torch.Tensor:
    """
    Computes balanced class weights for CrossEntropyLoss:
    w_c = N / (n_classes * count_c)
    """
    df = pd.read_csv(manifest_path)
    if "label" not in df.columns:
        df["label"] = df["dx"].map(CLASS_TO_IDX)

    counts = df["label"].value_counts().to_dict()
    total_samples = len(df)
    weights = []
    for c in range(num_classes):
        c_count = counts.get(c, 1)
        w = total_samples / (num_classes * c_count)
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def get_classification_dataloaders(
    manifests_dir: Optional[Union[str, Path]] = None,
    data_root: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
    apply_hair_removal: bool = False
) -> Dict[str, DataLoader]:
    """
    Factory function returning train, val, and test DataLoaders for classification.
    """
    if manifests_dir is None:
        manifests_dir = get_default_data_root() / "manifests"
    else:
        manifests_dir = Path(manifests_dir)

    train_ds = SkinClassificationDataset(
        manifest_path=manifests_dir / "classification_train.csv",
        data_root=data_root,
        is_training=True,
        apply_hair_removal=apply_hair_removal
    )
    val_ds = SkinClassificationDataset(
        manifest_path=manifests_dir / "classification_val.csv",
        data_root=data_root,
        is_training=False,
        apply_hair_removal=apply_hair_removal
    )
    test_ds = SkinClassificationDataset(
        manifest_path=manifests_dir / "classification_test.csv",
        data_root=data_root,
        is_training=False,
        apply_hair_removal=apply_hair_removal
    )

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        # Calculate sample weights for weighted random sampling
        class_weights = compute_class_weights(manifests_dir / "classification_train.csv")
        sample_weights = [class_weights[label].item() for label in train_ds.df["label"]]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


def get_segmentation_dataloaders(
    manifests_dir: Optional[Union[str, Path]] = None,
    data_root: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    apply_hair_removal: bool = False
) -> Dict[str, DataLoader]:
    """
    Factory function returning train, val, and test DataLoaders for segmentation.
    """
    if manifests_dir is None:
        manifests_dir = get_default_data_root() / "manifests"
    else:
        manifests_dir = Path(manifests_dir)

    train_ds = SkinSegmentationDataset(
        manifest_path=manifests_dir / "segmentation_train.csv",
        data_root=data_root,
        is_training=True,
        apply_hair_removal=apply_hair_removal
    )
    val_ds = SkinSegmentationDataset(
        manifest_path=manifests_dir / "segmentation_val.csv",
        data_root=data_root,
        is_training=False,
        apply_hair_removal=apply_hair_removal
    )
    test_ds = SkinSegmentationDataset(
        manifest_path=manifests_dir / "segmentation_test.csv",
        data_root=data_root,
        is_training=False,
        apply_hair_removal=apply_hair_removal
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
