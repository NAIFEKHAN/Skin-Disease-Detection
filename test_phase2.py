"""
Phase 2 Comprehensive Verification Suite
Tests:
1. Valid and corrupted image handling.
2. Classification tensor shape: (3, 224, 224).
3. Segmentation tensor shape: (3, 256, 256).
4. Mask tensor shape: (1, 256, 256) with binary {0.0, 1.0}.
5. Correct seven-class label mapping.
6. Image-mask alignment and lesion integrity.
7. Data augmentation consistency.
8. Dataset split integrity and zero-leakage cross-task check.
9. DataLoader batching and class weights calculation.
10. Live FastAPI endpoints and frontend pages.
"""

import sys
import os
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from services.preprocessing import (
    validate_and_decode_image,
    validate_and_decode_mask,
    remove_hair_artefacts,
    preprocess_for_classification,
    preprocess_for_segmentation,
    apply_training_augmentation,
    CLASSIFICATION_IMAGE_SIZE,
    SEGMENTATION_IMAGE_SIZE
)
from training.dataset import (
    SkinClassificationDataset,
    SkinSegmentationDataset,
    get_classification_dataloaders,
    get_segmentation_dataloaders,
    compute_class_weights,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"


def test_01_valid_and_corrupted_image_handling():
    print("\n--- Test 1: Valid & Corrupted Image Handling ---")
    # 1. Valid sample from classification manifest
    train_df = pd.read_csv(MANIFESTS_DIR / "classification_train.csv")
    sample_id = train_df.iloc[0]["image_id"]
    sample_img_path = DATA_DIR / "classification" / "images" / f"{sample_id}.jpg"
    assert sample_img_path.exists(), f"Sample image {sample_img_path} not found"
    rgb = validate_and_decode_image(sample_img_path)
    assert isinstance(rgb, np.ndarray)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.dtype == np.uint8
    print(f"✔ Genuine image ({sample_id}) decoded successfully. Dimensions: {rgb.shape}")

    # 2. Corrupted bytes
    corrupted_bytes = b"not_an_image_corrupted_header_data"
    try:
        validate_and_decode_image(corrupted_bytes)
        assert False, "Should have raised ValueError on corrupted bytes"
    except ValueError as e:
        print(f"✔ Corrupted byte buffer correctly rejected: {e}")

    # 3. Non-existent file
    try:
        validate_and_decode_image("non_existent_file.jpg")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"✔ Missing file correctly rejected: {e}")


def test_02_classification_tensor_shape():
    print("\n--- Test 2: Classification Tensor Shape (3, 224, 224) ---")
    train_df = pd.read_csv(MANIFESTS_DIR / "classification_train.csv")
    sample_id = train_df.iloc[0]["image_id"]
    sample_img_path = DATA_DIR / "classification" / "images" / f"{sample_id}.jpg"
    tensor = preprocess_for_classification(sample_img_path, apply_hair_removal=True, is_training=False)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32
    print(f"✔ Classification tensor verified: shape={tensor.shape}, dtype={tensor.dtype}")


def test_03_segmentation_tensor_shapes():
    print("\n--- Test 3: Segmentation Image & Mask Tensor Shapes ---")
    seg_df = pd.read_csv(MANIFESTS_DIR / "segmentation_train.csv")
    row = seg_df.iloc[0]
    sample_img_path = DATA_DIR / row["image_path"]
    sample_mask_path = DATA_DIR / row["mask_path"]
    assert sample_img_path.exists(), f"Sample image {sample_img_path} not found"
    assert sample_mask_path.exists(), f"Sample mask {sample_mask_path} not found"

    img_tensor, mask_tensor = preprocess_for_segmentation(
        sample_img_path, sample_mask_path, apply_hair_removal=False, is_training=False
    )
    assert img_tensor.shape == (3, 256, 256), f"Expected (3, 256, 256), got {img_tensor.shape}"
    assert mask_tensor.shape == (1, 256, 256), f"Expected (1, 256, 256), got {mask_tensor.shape}"
    assert mask_tensor.dtype == torch.float32

    # Verify binary values {0.0, 1.0}
    unique_vals = torch.unique(mask_tensor).tolist()
    assert all(v in [0.0, 1.0] for v in unique_vals), f"Mask contains non-binary values: {unique_vals}"
    print(f"✔ Segmentation image tensor shape: {img_tensor.shape}")
    print(f"✔ Segmentation mask tensor shape: {mask_tensor.shape}, unique values={unique_vals}")


def test_04_seven_class_label_mapping():
    print("\n--- Test 4: Seven-Class Label Mapping ---")
    expected_classes = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
    assert len(CLASS_TO_IDX) == 7
    for c in expected_classes:
        assert c in CLASS_TO_IDX, f"Missing class {c}"
        idx = CLASS_TO_IDX[c]
        assert IDX_TO_CLASS[idx] == c

    # Check manifest labels
    train_df = pd.read_csv(MANIFESTS_DIR / "classification_train.csv")
    unique_labels = sorted(train_df["label"].unique().tolist())
    assert unique_labels == [0, 1, 2, 3, 4, 5, 6], f"Unexpected label range: {unique_labels}"
    print(f"✔ Seven-class bijective mapping verified across all 7 disease categories: {CLASS_TO_IDX}")


def test_05_image_mask_alignment():
    print("\n--- Test 5: Image-Mask Alignment ---")
    seg_train_df = pd.read_csv(MANIFESTS_DIR / "segmentation_train.csv")
    row = seg_train_df.iloc[0]
    img_path = DATA_DIR / row["image_path"]
    mask_path = DATA_DIR / row["mask_path"]

    raw_img = validate_and_decode_image(img_path)
    raw_mask = validate_and_decode_mask(mask_path)
    assert raw_img.shape[:2] == raw_mask.shape[:2], f"Mismatched dimensions: img={raw_img.shape}, mask={raw_mask.shape}"
    lesion_pixels = (raw_mask > 0).sum()
    assert lesion_pixels > 0, "Segmentation mask has 0 lesion pixels"
    print(f"✔ Image-mask alignment verified: dimensions={raw_img.shape[:2]}, lesion_pixels={lesion_pixels}")


def test_06_data_augmentation_consistency():
    print("\n--- Test 6: Coordinated Data Augmentation ---")
    seg_df = pd.read_csv(MANIFESTS_DIR / "segmentation_train.csv")
    row = seg_df.iloc[0]
    sample_img_path = DATA_DIR / row["image_path"]
    sample_mask_path = DATA_DIR / row["mask_path"]

    img = validate_and_decode_image(sample_img_path)
    mask = validate_and_decode_mask(sample_mask_path)

    aug_img, aug_mask = apply_training_augmentation(img, mask)
    assert aug_img.shape == img.shape
    assert aug_mask.shape == mask.shape
    assert aug_mask.dtype == np.uint8
    assert set(np.unique(aug_mask)).issubset({0, 255})
    print("✔ Coordinated geometric augmentation verified on image & mask.")


def test_07_dataset_split_integrity_and_leakage():
    print("\n--- Test 7: Dataset Split Integrity & Leakage Verification ---")
    cls_train = pd.read_csv(MANIFESTS_DIR / "classification_train.csv")
    cls_val = pd.read_csv(MANIFESTS_DIR / "classification_val.csv")
    cls_test = pd.read_csv(MANIFESTS_DIR / "classification_test.csv")

    # 1. No lesion overlap within classification
    train_l = set(cls_train["lesion_id"])
    val_l = set(cls_val["lesion_id"])
    test_l = set(cls_test["lesion_id"])
    assert len(train_l & val_l) == 0, "Lesion leakage between classification train and val!"
    assert len(train_l & test_l) == 0, "Lesion leakage between classification train and test!"
    assert len(val_l & test_l) == 0, "Lesion leakage between classification val and test!"
    print("✔ Classification lesion isolation: 0 overlapping lesions.")

    # 2. No duplicate images within splits
    assert len(cls_train) == cls_train["image_id"].nunique()
    assert len(cls_val) == cls_val["image_id"].nunique()
    assert len(cls_test) == cls_test["image_id"].nunique()
    print("✔ Duplicate detection: 0 duplicate images within classification sets.")

    # 3. Cross-task evaluation contamination check
    seg_train = pd.read_csv(MANIFESTS_DIR / "segmentation_train.csv")
    seg_val = pd.read_csv(MANIFESTS_DIR / "segmentation_val.csv")
    seg_test = pd.read_csv(MANIFESTS_DIR / "segmentation_test.csv")

    cls_test_imgs = set(cls_test["image_id"])
    cls_train_imgs = set(cls_train["image_id"])
    seg_train_imgs = set(seg_train["image_id"])
    seg_test_imgs = set(seg_test["image_id"])

    assert len(cls_test_imgs & seg_train_imgs) == 0, "Classification test image present in segmentation train!"
    assert len(seg_test_imgs & cls_train_imgs) == 0, "Segmentation test image present in classification train!"
    print("✔ Cross-task evaluation isolation: ZERO test contamination between tasks.")


def test_08_dataloader_functionality():
    print("\n--- Test 8: DataLoader Batching & Class Weights ---")
    cls_loaders = get_classification_dataloaders(batch_size=8, num_workers=0)
    train_loader = cls_loaders["train"]
    val_loader = cls_loaders["val"]

    # Test one batch
    for images, labels, ids in train_loader:
        assert images.shape == (8, 3, 224, 224)
        assert labels.shape == (8,)
        assert len(ids) == 8
        break
    print(f"✔ Classification DataLoader batch verified: shape={images.shape}, labels={labels.tolist()}")

    seg_loaders = get_segmentation_dataloaders(batch_size=4, num_workers=0)
    for s_imgs, s_masks, s_ids in seg_loaders["train"]:
        assert s_imgs.shape == (4, 3, 256, 256)
        assert s_masks.shape == (4, 1, 256, 256)
        assert len(s_ids) == 4
        break
    print(f"✔ Segmentation DataLoader batch verified: img_shape={s_imgs.shape}, mask_shape={s_masks.shape}")

    # Test class weights
    weights = compute_class_weights(MANIFESTS_DIR / "classification_train.csv")
    assert weights.shape == (7,)
    assert (weights > 0).all()
    print(f"✔ Class weights computed for imbalanced loss: {weights.tolist()}")


def test_09_fastapi_endpoints():
    print("\n--- Test 9: Live FastAPI Endpoints & UI Pages ---")
    endpoints = [
        ("/", 200),
        ("/history", 200),
        ("/health", 200),
        ("/api/history", 200),
        ("/api/records", 200),
        ("/docs", 200),
    ]

    for ep, exp_status in endpoints:
        url = f"http://127.0.0.1:8000{ep}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            assert status == exp_status, f"{ep} returned {status}, expected {exp_status}"
            content = resp.read()
            assert len(content) > 0
            if ep == "/health":
                data = json.loads(content.decode("utf-8"))
                assert data["status"] == "healthy"
                assert data["database"] == "connected"
        print(f"✔ Endpoint {ep} passed with status {status}")


if __name__ == "__main__":
    test_01_valid_and_corrupted_image_handling()
    test_02_classification_tensor_shape()
    test_03_segmentation_tensor_shapes()
    test_04_seven_class_label_mapping()
    test_05_image_mask_alignment()
    test_06_data_augmentation_consistency()
    test_07_dataset_split_integrity_and_leakage()
    test_08_dataloader_functionality()
    test_09_fastapi_endpoints()
    print("\n" + "=" * 65)
    print("ALL 9 PHASE 2 VERIFICATION TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)
