"""
Colab Package Creator.
Phase 3 Implementation.

Bundles a compact archive containing:
- models/ (unet.py, __init__.py)
- services/ (preprocessing.py, segmentation.py, __init__.py)
- training/ (dataset.py, train_unet.py, train_unet.ipynb, __init__.py)
- data/manifests/ (segmentation_*.csv, dataset_summary.json)
- data/segmentation/ (images/ and masks/)

Strictly excludes:
- venv/
- data/classification/
- data/*.db
- __pycache__/
- .git/
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = BASE_DIR / "colab_unet_package.zip"


def create_colab_package():
    print("=" * 60)
    print("Creating Compact Google Colab U-Net Package")
    print("=" * 60)

    include_patterns = [
        "models/__init__.py",
        "models/unet.py",
        "services/__init__.py",
        "services/preprocessing.py",
        "services/segmentation.py",
        "training/__init__.py",
        "training/dataset.py",
        "training/train_unet.py",
        "training/train_unet.ipynb",
        "data/manifests/segmentation_train.csv",
        "data/manifests/segmentation_val.csv",
        "data/manifests/segmentation_test.csv",
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

        # 2. Add segmentation images and masks
        seg_dir = BASE_DIR / "data" / "segmentation"
        for folder in ["images", "masks"]:
            folder_path = seg_dir / folder
            if folder_path.exists():
                for f in folder_path.glob("*.*"):
                    rel = f.relative_to(BASE_DIR)
                    zip_out.write(f, arcname=str(rel).replace("\\", "/"))
                    total_files += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"✔ Archive created successfully at: {OUTPUT_ZIP}")
    print(f"✔ Total bundled files: {total_files}")
    print(f"✔ Archive file size: {size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    create_colab_package()
