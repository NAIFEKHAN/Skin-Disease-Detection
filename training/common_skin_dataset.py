"""
PyTorch Dataset and DataLoader for Common Skin Disease Clinical Classification.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from models.common_skin_vit import (
    get_clinical_transforms,
    COMMON_SKIN_CLASSES,
    CLASS_TO_IDX
)


class CommonSkinDataset(Dataset):
    """
    Dataset wrapper for clinical common skin disease classification.
    Reads manifests containing relative_path, label_idx, case_id, skin_tone.
    """
    def __init__(
        self,
        manifest_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        is_training: bool = False
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.is_training = is_training

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_clinical_transforms(is_training=is_training)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_path = row["relative_path"]
        img_path = self.root_dir / rel_path

        # Fallback if image not found at root_dir / relative_path
        if not img_path.exists():
            img_path = self.manifest_path.parent.parent / "common_skin" / "images" / row["filename"]
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        label = int(row["label_idx"])

        metadata = {
            "case_id": str(row.get("case_id", "")),
            "source": str(row.get("source", "")),
            "target_class": str(row.get("target_class", "")),
            "skin_tone": str(row.get("skin_tone", "UNKNOWN")),
            "filename": str(row.get("filename", ""))
        }

        return tensor, label, metadata


def build_common_skin_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Constructs Train, Validation, and Test DataLoaders.
    """
    if data_dir is None:
        data_dir = Path("data")

    manifest_dir = data_dir / "manifests"
    train_manifest = manifest_dir / "common_skin_train.csv"
    val_manifest = manifest_dir / "common_skin_val.csv"
    test_manifest = manifest_dir / "common_skin_test.csv"

    root_dir = data_dir.parent

    train_dataset = CommonSkinDataset(train_manifest, root_dir=root_dir, is_training=True)
    val_dataset = CommonSkinDataset(val_manifest, root_dir=root_dir, is_training=False)
    test_dataset = CommonSkinDataset(test_manifest, root_dir=root_dir, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    stats = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "batch_size": batch_size,
        "num_classes": len(COMMON_SKIN_CLASSES)
    }

    return train_loader, val_loader, test_loader, stats
