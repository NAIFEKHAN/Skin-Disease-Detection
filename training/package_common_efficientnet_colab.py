"""
Build Colab Training Package ZIP for Common Skin Disease EfficientNet-B2 GPU Training.
"""

import os
import zipfile
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_PATH = ROOT_DIR / "colab_common_efficientnet_package.zip"


def create_colab_package():
    print(f"Creating Colab EfficientNet training package: {PACKAGE_PATH}...")
    if PACKAGE_PATH.exists():
        try:
            PACKAGE_PATH.unlink()
        except Exception as e:
            print(f"Warning unlinking existing package: {e}")

    files_to_pack = []

    # 1. Models
    files_to_pack.append(("models/common_skin_efficientnet.py", "models/common_skin_efficientnet.py"))
    files_to_pack.append(("models/__init__.py", "models/__init__.py"))

    # 2. Training scripts & notebook
    files_to_pack.append(("training/train_common_efficientnet.py", "training/train_common_efficientnet.py"))
    files_to_pack.append(("training/evaluate_common_efficientnet.py", "training/evaluate_common_efficientnet.py"))
    files_to_pack.append(("training/train_common_efficientnet.ipynb", "train_common_efficientnet.ipynb"))
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

    # 4. Clinical images referenced in manifests
    all_manifest = ROOT_DIR / "data" / "manifests" / "common_skin_all.csv"
    if all_manifest.exists():
        df = pd.read_csv(all_manifest)
        for _, row in df.iterrows():
            rel_path = row["relative_path"]
            abs_path = ROOT_DIR / rel_path
            if abs_path.exists():
                files_to_pack.append((rel_path, rel_path))

    print(f"Total files to package: {len(files_to_pack)}")

    tmp_path = PACKAGE_PATH.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arcname in files_to_pack:
            src_path = ROOT_DIR / src
            if src_path.exists():
                z.write(src_path, arcname)

    if PACKAGE_PATH.exists():
        PACKAGE_PATH.unlink()
    tmp_path.rename(PACKAGE_PATH)

    size_mb = PACKAGE_PATH.stat().st_size / (1024 * 1024)
    print(f"Colab EfficientNet package created successfully: {PACKAGE_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    create_colab_package()
