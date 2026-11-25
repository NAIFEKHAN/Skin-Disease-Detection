"""
Colab Package Creator for Vision Transformer (ViT-B/16).
Phase 4 Implementation.

Bundles a compact archive containing:
- models/ (vit.py, __init__.py)
- services/ (preprocessing.py, prediction.py, __init__.py)
- training/ (dataset.py, train_vit.py, train_vit.ipynb, __init__.py)
- data/manifests/ (classification_*.csv, dataset_summary.json)
- data/classification/images/ (1,757 genuine dermoscopic images)

Strictly excludes:
- data/segmentation/
- models/*.pth (preserves existing U-Net checkpoint)
- venv/
- data/*.db
- __pycache__/
- .git/
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = BASE_DIR / "colab_vit_package.zip"


def create_colab_vit_package():
    print("=" * 60)
    print("Creating Compact Google Colab ViT-B/16 Package")
    print("=" * 60)

    include_patterns = [
        "models/__init__.py",
        "models/vit.py",
        "services/__init__.py",
        "services/preprocessing.py",
        "services/prediction.py",
        "training/__init__.py",
        "training/dataset.py",
        "training/train_vit.py",
        "training/train_vit.ipynb",
        "data/manifests/classification_train.csv",
        "data/manifests/classification_val.csv",
        "data/manifests/classification_test.csv",
        "data/manifests/dataset_summary.json"
    ]

    total_files = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zip_out:
        # 1. Add specific code and manifest files
        for rel_path in include_patterns:
            p = BASE_DIR / rel_path
            if p.exists():
                zip_out.write(p, arcname=rel_path)
                total_files += 1
            else:
                print(f"Warning: Missing file {rel_path}")

        # 2. Add classification images
        cls_img_dir = BASE_DIR / "data" / "classification" / "images"
        if cls_img_dir.exists():
            for f in cls_img_dir.glob("*.jpg"):
                rel = f.relative_to(BASE_DIR)
                zip_out.write(f, arcname=str(rel).replace("\\", "/"))
                total_files += 1
        else:
            print("Warning: data/classification/images not found!")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"✔ Archive created successfully at: {OUTPUT_ZIP}")
    print(f"✔ Total bundled files: {total_files}")
    print(f"✔ Archive file size: {size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    create_colab_vit_package()
