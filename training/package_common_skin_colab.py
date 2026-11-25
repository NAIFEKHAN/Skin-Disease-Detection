"""
Build Colab Training Package ZIP for Common Skin Disease ViT-B/16 GPU Training.
"""

import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_PATH = ROOT_DIR / "colab_common_skin_package.zip"


def create_colab_package():
    print(f"Creating Colab training package: {PACKAGE_PATH}...")
    if PACKAGE_PATH.exists():
        PACKAGE_PATH.unlink()

    files_to_pack = []

    # 1. Model architecture
    files_to_pack.append(("models/common_skin_vit.py", "models/common_skin_vit.py"))
    files_to_pack.append(("models/__init__.py", "models/__init__.py"))

    # 2. Training scripts & notebook
    files_to_pack.append(("training/common_skin_dataset.py", "training/common_skin_dataset.py"))
    files_to_pack.append(("training/train_common_skin.py", "training/train_common_skin.py"))
    files_to_pack.append(("training/train_common_skin.ipynb", "train_common_skin.ipynb"))
    files_to_pack.append(("training/__init__.py", "training/__init__.py"))

    # 3. Manifests
    manifest_files = [
        "data/manifests/common_skin_train.csv",
        "data/manifests/common_skin_val.csv",
        "data/manifests/common_skin_test.csv",
        "data/manifests/common_skin_all.csv",
        "data/manifests/common_skin_metadata.json"
    ]
    for mf in manifest_files:
        p = ROOT_DIR / mf
        if p.exists():
            files_to_pack.append((str(p.relative_to(ROOT_DIR)), mf))

    # 4. Genuine clinical images referenced in manifests
    import pandas as pd
    all_manifest = ROOT_DIR / "data" / "manifests" / "common_skin_all.csv"
    if all_manifest.exists():
        df = pd.read_csv(all_manifest)
        for _, row in df.iterrows():
            rel_path = row["relative_path"]
            abs_path = ROOT_DIR / rel_path
            if abs_path.exists():
                files_to_pack.append((rel_path, rel_path))

    print(f"Total files to package: {len(files_to_pack)}")

    with zipfile.ZipFile(PACKAGE_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arcname in files_to_pack:
            src_path = ROOT_DIR / src
            if src_path.exists():
                z.write(src_path, arcname)

    size_mb = PACKAGE_PATH.stat().st_size / (1024 * 1024)
    print(f"Colab package created successfully: {PACKAGE_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    create_colab_package()
