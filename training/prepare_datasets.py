"""
Dataset Preparation & Manifest Generator for Phase 2:
- Samples compact, balanced HAM10000 classification dataset (up to 300 per class, total 1,757 genuine images).
- Collects 1,000 genuine ISIC segmentation image-mask pairs with diverse lesion sizes and conditions.
- Performs stratified, lesion-group-aware 70/15/15 splitting with seed 42.
- Verifies zero cross-task test leakage between classification and segmentation.
- Downloads images from official ISIC Archive S3 mirror and validates image integrity.
"""

import os
import random
import urllib.request
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd

# Constants
RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
CLASS_DIR = DATA_DIR / "classification"
SEG_DIR = DATA_DIR / "segmentation"
RAW_MASKS_DIR = DATA_DIR / "raw_masks" / "HAM10000_segmentations_lesion_tschandl"

S3_IMAGE_BASE_URL = "https://isic-archive.s3.us-east-1.amazonaws.com/images/{image_id}.jpg"

# 7 Disease classes in HAM10000
DISEASE_CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASS_NAME_MAP = {
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis-like Lesions",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "vasc": "Vascular Lesions",
    "df": "Dermatofibroma"
}


def sample_classification_dataset(metadata_df: pd.DataFrame, max_per_class: int = 300) -> pd.DataFrame:
    """
    Reproducibly samples up to max_per_class images per disease category,
    keeping lesions grouped where possible.
    """
    rng = random.Random(RANDOM_SEED)
    selected_rows = []

    for dx in DISEASE_CLASSES:
        class_df = metadata_df[metadata_df["dx"] == dx].copy()
        total_available = len(class_df)

        if total_available <= max_per_class:
            selected_rows.append(class_df)
        else:
            # Group by lesion_id
            lesions = list(class_df["lesion_id"].unique())
            rng.shuffle(lesions)

            chosen_lesions = []
            img_count = 0
            for l in lesions:
                l_imgs = class_df[class_df["lesion_id"] == l]
                if img_count + len(l_imgs) <= max_per_class:
                    chosen_lesions.append(l)
                    img_count += len(l_imgs)
                elif img_count == max_per_class:
                    break

            # Fill remainder with single-image lesions if needed to hit exact target
            if img_count < max_per_class:
                remaining = class_df[~class_df["lesion_id"].isin(chosen_lesions)]
                rem_needed = max_per_class - img_count
                single_lesions = [l for l in remaining["lesion_id"].unique() if len(remaining[remaining["lesion_id"] == l]) == 1]
                chosen_lesions.extend(single_lesions[:rem_needed])

            sampled = class_df[class_df["lesion_id"].isin(chosen_lesions)].copy()
            selected_rows.append(sampled)

    all_sampled = pd.concat(selected_rows, ignore_index=True)
    return all_sampled


def split_lesion_group_aware(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a classification dataset into train, val, and test sets using
    stratified, lesion-group-aware partitioning (no lesion straddles splits).
    """
    rng = random.Random(RANDOM_SEED)
    train_records, val_records, test_records = [], [], []

    for dx in DISEASE_CLASSES:
        class_df = df[df["dx"] == dx].copy()
        lesion_groups = class_df.groupby("lesion_id")
        unique_lesions = list(lesion_groups.groups.keys())
        rng.shuffle(unique_lesions)

        total_imgs = len(class_df)
        target_train = int(round(total_imgs * train_ratio))
        target_val = int(round(total_imgs * val_ratio))

        c_train, c_val, c_test = [], [], []
        train_count, val_count = 0, 0

        for l in unique_lesions:
            l_df = lesion_groups.get_group(l)
            count = len(l_df)
            if train_count + count <= target_train or (train_count < target_train and (target_train - train_count) >= count // 2):
                c_train.append(l_df)
                train_count += count
            elif val_count + count <= target_val or (val_count < target_val and (target_val - val_count) >= count // 2):
                c_val.append(l_df)
                val_count += count
            else:
                c_test.append(l_df)

        if c_train:
            train_records.append(pd.concat(c_train, ignore_index=True))
        if c_val:
            val_records.append(pd.concat(c_val, ignore_index=True))
        if c_test:
            test_records.append(pd.concat(c_test, ignore_index=True))

    train_df = pd.concat(train_records, ignore_index=True)
    val_df = pd.concat(val_records, ignore_index=True)
    test_df = pd.concat(test_records, ignore_index=True)
    return train_df, val_df, test_df


def select_segmentation_dataset(
    train_cls: pd.DataFrame,
    val_cls: pd.DataFrame,
    test_cls: pd.DataFrame,
    target_total: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Selects 1,000 genuine image-mask pairs across diverse lesion sizes,
    strictly matching the split assignments of classification images to guarantee
    zero cross-task test leakage.
    Target: 700 Train (70%), 150 Val (15%), 150 Test (15%).
    """
    target_train = 700
    target_val = 150
    target_test = 150

    def calculate_mask_diversity(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        areas = []
        valid_masks = []
        for _, row in df.iterrows():
            img_id = row["image_id"]
            mask_path = RAW_MASKS_DIR / f"{img_id}_segmentation.png"
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    area_pct = float((m > 127).sum() / m.size * 100)
                    areas.append(area_pct)
                    valid_masks.append(True)
                    continue
            areas.append(0.0)
            valid_masks.append(False)
        df["lesion_area_pct"] = areas
        df["valid_mask"] = valid_masks
        df["mask_filename"] = df["image_id"] + "_segmentation.png"
        return df[df["valid_mask"]].copy()

    train_annotated = calculate_mask_diversity(train_cls)
    val_annotated = calculate_mask_diversity(val_cls)
    test_annotated = calculate_mask_diversity(test_cls)

    # Stratified selection by lesion area tiers (small < 10%, medium 10-30%, large > 30%)
    def sample_diverse_masks(df: pd.DataFrame, target_count: int, seed: int) -> pd.DataFrame:
        rng = random.Random(seed)
        df = df.copy()
        small = df[df["lesion_area_pct"] < 10.0]
        medium = df[(df["lesion_area_pct"] >= 10.0) & (df["lesion_area_pct"] <= 30.0)]
        large = df[df["lesion_area_pct"] > 30.0]

        # Allocate proportional quotas
        q_small = int(round(target_count * 0.20))
        q_large = int(round(target_count * 0.30))
        q_medium = target_count - q_small - q_large

        def pick_n(sub_df, n):
            indices = list(sub_df.index)
            rng.shuffle(indices)
            return indices[:n]

        picked_indices = pick_n(small, q_small) + pick_n(medium, q_medium) + pick_n(large, q_large)
        # If any tier is short, fill from remaining
        if len(picked_indices) < target_count:
            remainder = [i for i in df.index if i not in picked_indices]
            rng.shuffle(remainder)
            picked_indices.extend(remainder[:target_count - len(picked_indices)])

        return df.loc[picked_indices].copy()

    seg_train = sample_diverse_masks(train_annotated, target_train, seed=RANDOM_SEED)
    seg_val = sample_diverse_masks(val_annotated, target_val, seed=RANDOM_SEED + 1)
    seg_test = sample_diverse_masks(test_annotated, target_test, seed=RANDOM_SEED + 2)

    return seg_train, seg_val, seg_test


def download_single_image(img_id: str, dest_path: Path) -> bool:
    """Downloads an image from official ISIC S3 and validates decoding."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        # Validate that the file is not corrupted
        img = cv2.imread(str(dest_path))
        if img is not None and img.shape[0] > 50:
            return True

    url = S3_IMAGE_BASE_URL.format(image_id=img_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                data = response.read()
                # Verify decoding before writing
                np_arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Corrupted image data received for {img_id}")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(data)
                return True
        except Exception as e:
            if attempt == 2:
                print(f"[FAIL] Failed to download {img_id}: {e}")
                return False
    return False


def copy_or_link_mask(img_id: str, dest_path: Path) -> bool:
    """Copies the expert-annotated mask to the destination directory."""
    src_mask = RAW_MASKS_DIR / f"{img_id}_segmentation.png"
    if not src_mask.exists():
        print(f"[FAIL] Missing mask for {img_id}")
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        data = src_mask.read_bytes()
        dest_path.write_bytes(data)
    return True


def run_pipeline():
    print("=" * 65)
    print("AI Skin Disease Detection — Phase 2 Dataset Pipeline")
    print("=" * 65)

    metadata_file = DATA_DIR / "HAM10000_metadata.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing {metadata_file}. Run metadata download first.")

    metadata = pd.read_csv(metadata_file)
    print(f"Loaded official metadata: {len(metadata)} total records.")

    # 1. Classification sampling
    print("\n--- 1. Sampling Classification Dataset (Target: up to 300 per category) ---")
    cls_sampled = sample_classification_dataset(metadata, max_per_class=300)
    print(f"Sampled classification images: {len(cls_sampled)}")
    print(cls_sampled["dx"].value_counts())

    # 2. Split classification dataset
    print("\n--- 2. Performing Stratified Lesion-Group-Aware Splitting (70/15/15) ---")
    cls_train, cls_val, cls_test = split_lesion_group_aware(cls_sampled)
    print(f"Classification Train: {len(cls_train)} ({len(cls_train)/len(cls_sampled)*100:.1f}%)")
    print(f"Classification Val:   {len(cls_val)} ({len(cls_val)/len(cls_sampled)*100:.1f}%)")
    print(f"Classification Test:  {len(cls_test)} ({len(cls_test)/len(cls_sampled)*100:.1f}%)")

    # Verify lesion isolation
    train_lesions = set(cls_train["lesion_id"])
    val_lesions = set(cls_val["lesion_id"])
    test_lesions = set(cls_test["lesion_id"])
    assert len(train_lesions & val_lesions) == 0, "Lesion leakage between train and val!"
    assert len(train_lesions & test_lesions) == 0, "Lesion leakage between train and test!"
    assert len(val_lesions & test_lesions) == 0, "Lesion leakage between val and test!"
    print("✔ Classification lesion isolation verified: 0 lesion overlap between splits.")

    # 3. Segmentation dataset selection
    print("\n--- 3. Selecting Segmentation Dataset (Target: 1,000 samples) ---")
    seg_train, seg_val, seg_test = select_segmentation_dataset(cls_train, cls_val, cls_test, target_total=1000)
    print(f"Segmentation Train: {len(seg_train)} (70%)")
    print(f"Segmentation Val:   {len(seg_val)} (15%)")
    print(f"Segmentation Test:  {len(seg_test)} (15%)")
    print(f"Total Segmentation: {len(seg_train) + len(seg_val) + len(seg_test)}")

    # Verify cross-task data leakage
    print("\n--- 4. Cross-Task Evaluation Leakage Verification ---")
    cls_test_imgs = set(cls_test["image_id"])
    seg_train_imgs = set(seg_train["image_id"])
    seg_val_imgs = set(seg_val["image_id"])
    seg_test_imgs = set(seg_test["image_id"])
    cls_train_imgs = set(cls_train["image_id"])
    cls_val_imgs = set(cls_val["image_id"])

    assert len(cls_test_imgs & seg_train_imgs) == 0, "Classification test image found in segmentation training!"
    assert len(cls_test_imgs & seg_val_imgs) == 0, "Classification test image found in segmentation validation!"
    assert len(seg_test_imgs & cls_train_imgs) == 0, "Segmentation test image found in classification training!"
    assert len(seg_test_imgs & cls_val_imgs) == 0, "Segmentation test image found in classification validation!"
    print("✔ Cross-task evaluation leakage verified: ZERO evaluation contamination.")

    # Check shared images
    shared_total = len((cls_train_imgs | cls_val_imgs | cls_test_imgs) & (seg_train_imgs | seg_val_imgs | seg_test_imgs))
    print(f"✔ Shared images between classification & segmentation: {shared_total} (all strictly split-aligned).")

    # 4. Download images
    all_needed_images = sorted(list(set(cls_sampled["image_id"]) | set(seg_train["image_id"]) | set(seg_val["image_id"]) | set(seg_test["image_id"])))
    print(f"\n--- 5. Downloading {len(all_needed_images)} Genuine Images from Official ISIC S3 ---")

    cls_img_dir = CLASS_DIR / "images"
    seg_img_dir = SEG_DIR / "images"
    seg_mask_dir = SEG_DIR / "masks"
    cls_img_dir.mkdir(parents=True, exist_ok=True)
    seg_img_dir.mkdir(parents=True, exist_ok=True)
    seg_mask_dir.mkdir(parents=True, exist_ok=True)

    def process_image(img_id):
        dest_cls = cls_img_dir / f"{img_id}.jpg"
        ok = download_single_image(img_id, dest_cls)
        # If also in segmentation, copy/link to segmentation images
        if img_id in (seg_train_imgs | seg_val_imgs | seg_test_imgs):
            dest_seg = seg_img_dir / f"{img_id}.jpg"
            if not dest_seg.exists() and dest_cls.exists():
                dest_seg.write_bytes(dest_cls.read_bytes())
            dest_mask = seg_mask_dir / f"{img_id}_segmentation.png"
            copy_or_link_mask(img_id, dest_mask)
        return ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(process_image, all_needed_images))

    success_count = sum(results)
    print(f"Downloaded and verified: {success_count} / {len(all_needed_images)} images.")
    if success_count < len(all_needed_images):
        print(f"Warning: {len(all_needed_images) - success_count} images failed to download.")

    # 5. Save manifests
    print("\n--- 6. Saving Reproducible CSV Manifests ---")
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # Add relative filepaths to manifests
    def enrich_cls(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        df = df.copy()
        df["split"] = split_name
        df["image_path"] = df["image_id"].apply(lambda x: f"classification/images/{x}.jpg")
        df["category_name"] = df["dx"].map(CLASS_NAME_MAP)
        # Class index 0-6
        class_to_idx = {c: i for i, c in enumerate(DISEASE_CLASSES)}
        df["label"] = df["dx"].map(class_to_idx)
        return df

    def enrich_seg(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        df = df.copy()
        df["split"] = split_name
        df["image_path"] = df["image_id"].apply(lambda x: f"segmentation/images/{x}.jpg")
        df["mask_path"] = df["image_id"].apply(lambda x: f"segmentation/masks/{x}_segmentation.png")
        return df

    cls_train_out = enrich_cls(cls_train, "train")
    cls_val_out = enrich_cls(cls_val, "val")
    cls_test_out = enrich_cls(cls_test, "test")

    seg_train_out = enrich_seg(seg_train, "train")
    seg_val_out = enrich_seg(seg_val, "val")
    seg_test_out = enrich_seg(seg_test, "test")

    cls_train_out.to_csv(MANIFESTS_DIR / "classification_train.csv", index=False)
    cls_val_out.to_csv(MANIFESTS_DIR / "classification_val.csv", index=False)
    cls_test_out.to_csv(MANIFESTS_DIR / "classification_test.csv", index=False)

    seg_train_out.to_csv(MANIFESTS_DIR / "segmentation_train.csv", index=False)
    seg_val_out.to_csv(MANIFESTS_DIR / "segmentation_val.csv", index=False)
    seg_test_out.to_csv(MANIFESTS_DIR / "segmentation_test.csv", index=False)

    # Save summary report
    summary = {
        "random_seed": RANDOM_SEED,
        "classification": {
            "total_images": len(cls_sampled),
            "classes": DISEASE_CLASSES,
            "class_distribution": cls_sampled["dx"].value_counts().to_dict(),
            "splits": {
                "train": len(cls_train_out),
                "val": len(cls_val_out),
                "test": len(cls_test_out)
            },
            "unique_lesions": cls_sampled["lesion_id"].nunique()
        },
        "segmentation": {
            "total_pairs": len(seg_train_out) + len(seg_val_out) + len(seg_test_out),
            "splits": {
                "train": len(seg_train_out),
                "val": len(seg_val_out),
                "test": len(seg_test_out)
            },
            "diversity_metrics": {
                "mean_lesion_area_pct": float(pd.concat([seg_train_out, seg_val_out, seg_test_out])["lesion_area_pct"].mean()),
                "min_lesion_area_pct": float(pd.concat([seg_train_out, seg_val_out, seg_test_out])["lesion_area_pct"].min()),
                "max_lesion_area_pct": float(pd.concat([seg_train_out, seg_val_out, seg_test_out])["lesion_area_pct"].max())
            }
        },
        "leakage_verification": {
            "lesion_overlap_cls_splits": 0,
            "test_overlap_cls_and_seg_train": 0,
            "shared_images_count": shared_total,
            "split_consistency_guaranteed": True
        }
    }

    with open(MANIFESTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✔ Manifests written successfully to data/manifests/")
    print(f"Summary saved to {MANIFESTS_DIR / 'dataset_summary.json'}")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
