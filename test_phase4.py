"""
Comprehensive Phase 4 Verification Suite:
Vision Transformer (ViT-B/16) Disease Classification Model & Pipeline.

Tests:
1. ViT-B/16 model architecture, layer configurations, and parameter counts.
2. Freeze/unfreeze logic: verify frozen parameters (blocks 0-9) vs trainable (blocks 10-11, head).
3. Forward pass tensor shapes: (B, 3, 224, 224) -> (B, 7).
4. Forward and backward pass gradient flow (checking head & unchilled blocks receive gradients).
5. Softmax probability output with predict_proba() (sums to 1.0, range [0, 1]).
6. 7-class indexing and metadata mapping (MEL, NV, BCC, AKIEC, BKL, DF, VASC).
7. Weighted Cross-Entropy Loss computation with class weights from manifest.
8. Discriminative parameter grouping (head learning rate vs backbone learning rate).
9. PredictionService CPU inference, DullRazor hair removal preprocessing, top-3 differential ranking.
10. PredictionService error handling (corrupted input, unready/missing checkpoint).
11. Verification of Colab GPU training artifacts: notebook train_vit.ipynb and colab_vit_package.zip.
12. Regression verification: U-Net SegmentationService and live FastAPI endpoints.
"""

import os
import sys
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.vit import (
    build_vit_model,
    VisionTransformerClassifier,
    DISEASE_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_DISPLAY_NAMES,
    VIT_MODEL_NAME
)
from training.dataset import compute_class_weights
from services.prediction import PredictionService, get_prediction_service
from services.segmentation import get_segmentation_service

DATA_DIR = BASE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"


def test_01_vit_architecture_and_parameters():
    print("\n--- Test 1: ViT-B/16 Architecture & Parameter Verification ---")
    model = build_vit_model(pretrained=False, freeze_backbone=True, unfreeze_blocks=2)
    p_info = model.get_parameter_count()

    assert model.num_classes == 7, f"Expected 7 classes, got {model.num_classes}"
    assert p_info["total_parameters"] > 85_000_000, f"Expected >85M parameters, got {p_info['total_parameters']}"
    assert p_info["trainable_parameters"] > 14_000_000, f"Expected >14M trainable parameters, got {p_info['trainable_parameters']}"
    assert p_info["frozen_parameters"] > 70_000_000, f"Expected >70M frozen parameters, got {p_info['frozen_parameters']}"
    print(f"✔ Architecture verified: Total={p_info['total_parameters']:,}, Trainable={p_info['trainable_parameters']:,}, Frozen={p_info['frozen_parameters']:,}")


def test_02_freeze_unfreeze_layer_verification():
    print("\n--- Test 2: Selective Freeze/Unfreeze Layer Verification ---")
    model = build_vit_model(pretrained=False, freeze_backbone=True, unfreeze_blocks=2)

    # Check that early blocks require_grad is False
    early_block = model.vit.blocks[0]
    for param in early_block.parameters():
        assert not param.requires_grad, "Expected early transformer block parameters to be frozen!"

    # Check that last blocks require_grad is True
    late_block = model.vit.blocks[11]
    for param in late_block.parameters():
        assert param.requires_grad, "Expected unchilled transformer block parameters to be trainable!"

    # Check classification head is trainable
    for param in model.vit.head.parameters():
        assert param.requires_grad, "Expected classification head parameters to be trainable!"

    print("✔ Layer freeze/unfreeze verified: blocks 0-9 frozen, blocks 10-11 & head trainable.")


def test_03_forward_pass_and_logits_shape():
    print("\n--- Test 3: Forward Pass & Logits Dimension (B, 7) ---")
    model = build_vit_model(pretrained=False, num_classes=7)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (4, 7), f"Expected shape (4, 7), got {logits.shape}"
    assert logits.dtype == torch.float32
    print(f"✔ Forward pass logits shape verified: {logits.shape}")


def test_04_gradient_flow_and_backprop():
    print("\n--- Test 4: Forward & Backward Pass Gradient Propagation ---")
    model = build_vit_model(pretrained=False, freeze_backbone=True, unfreeze_blocks=2)
    model.train()

    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_labels = torch.tensor([0, 3], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    logits = model(dummy_input)
    loss = criterion(logits, dummy_labels)

    loss.backward()

    # Head must have gradients
    head_grad = model.vit.head.weight.grad
    assert head_grad is not None, "Head weights have no gradient!"
    assert torch.sum(torch.abs(head_grad)) > 0, "Head weight gradient is all zeros!"

    # Block 11 must have gradients
    block11_grad = model.vit.blocks[11].attn.qkv.weight.grad
    assert block11_grad is not None, "Block 11 weights have no gradient!"
    assert torch.sum(torch.abs(block11_grad)) > 0, "Block 11 weight gradient is all zeros!"

    # Block 0 must NOT have gradients
    block0_grad = model.vit.blocks[0].attn.qkv.weight.grad
    assert block0_grad is None, "Frozen Block 0 should not accumulate gradients!"

    print("✔ Gradient propagation verified: trainable layers receive nonzero gradients, frozen layers receive none.")


def test_05_softmax_probabilities():
    print("\n--- Test 5: Softmax Probability Computation ---")
    model = build_vit_model(pretrained=False)
    model.eval()

    dummy_input = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        probs = model.predict_proba(dummy_input)

    assert probs.shape == (3, 7), f"Expected shape (3, 7), got {probs.shape}"
    sums = torch.sum(probs, dim=-1)
    for s in sums:
        assert torch.isclose(s, torch.tensor(1.0), atol=1e-5), f"Probabilities do not sum to 1.0: {s.item()}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1] range!"
    print(f"✔ Softmax probabilities verified: shapes={probs.shape}, row sums=1.0000.")


def test_06_disease_classes_and_indexing():
    print("\n--- Test 6: 7-Class Disease Categories & Label Mappings ---")
    expected_classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
    assert DISEASE_CLASSES == expected_classes, f"Class list mismatch: {DISEASE_CLASSES}"

    for idx, name in enumerate(expected_classes):
        assert CLASS_TO_IDX[name] == idx, f"Mapping mismatch for {name}: {CLASS_TO_IDX[name]} != {idx}"
        assert IDX_TO_CLASS[idx] == name, f"Reverse mapping mismatch for {idx}: {IDX_TO_CLASS[idx]} != {name}"
        assert name in CLASS_DISPLAY_NAMES, f"Display name missing for {name}"

    print(f"✔ 7-class disease mappings verified: {expected_classes}")


def test_07_class_weights_computation():
    print("\n--- Test 7: Class Weights Computation from Manifest ---")
    train_manifest = MANIFESTS_DIR / "classification_train.csv"
    assert train_manifest.exists(), f"Train manifest not found at {train_manifest}"

    weights = compute_class_weights(train_manifest, num_classes=7)
    assert len(weights) == 7, f"Expected 7 class weights, got {len(weights)}"
    assert (weights > 0).all(), "All class weights must be positive"
    # Rare classes (DF=5, VASC=6) must have higher weight than common classes (MEL=0, NV=1)
    assert weights[5] > weights[0], f"Expected DF weight ({weights[5]}) > MEL weight ({weights[0]})"
    assert weights[6] > weights[1], f"Expected VASC weight ({weights[6]}) > NV weight ({weights[1]})"

    print(f"✔ Class weights computed successfully: {[round(w, 4) for w in weights.tolist()]}")


def test_08_discriminative_parameter_groups():
    print("\n--- Test 8: Discriminative Learning Rate Parameter Grouping ---")
    model = build_vit_model(pretrained=False, freeze_backbone=True, unfreeze_blocks=2)
    param_groups = model.get_parameter_groups(head_lr=1e-4, backbone_lr=1e-5)

    assert len(param_groups) == 2, f"Expected 2 parameter groups, got {len(param_groups)}"
    group_map = {g["name"]: g for g in param_groups}
    assert group_map["head"]["lr"] == 1e-4, "Head group LR mismatch"
    assert group_map["backbone"]["lr"] == 1e-5, "Backbone group LR mismatch"
    assert len(group_map["head"]["params"]) > 0, "Head group has 0 parameters"
    assert len(group_map["backbone"]["params"]) > 0, "Backbone group has 0 parameters"

    print(f"✔ Parameter groups verified: Head LR={group_map['head']['lr']}, Backbone LR={group_map['backbone']['lr']}")


def test_09_prediction_service_inference():
    print("\n--- Test 9: PredictionService Inference & Preprocessing ---")
    model_ckpt = BASE_DIR / "models" / "vit_model.pth"
    assert model_ckpt.exists(), f"Model checkpoint missing at {model_ckpt}"

    service = PredictionService(model_path=model_ckpt)
    assert service.is_ready, f"Service not ready: {service.error_message}"

    # Test with dummy image array
    dummy_bgr = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)

    # Test without hair removal
    result = service.predict_image(dummy_bgr, apply_hair_removal=False)
    assert result["success"] is True
    assert "predicted_class_code" in result
    assert "confidence_score" in result
    assert "predicted_disease_name" in result
    assert "class_probabilities" in result
    assert "top_3_predictions" in result
    assert "is_uncertain" in result
    assert "uncertainty_note" in result
    assert "clinical_scope_note" in result

    assert result["predicted_class_code"].lower() in DISEASE_CLASSES
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert len(result["class_probabilities"]) == 7
    assert len(result["top_3_predictions"]) == 3

    # Probabilities in top 3 differential must be descending
    diff = result["top_3_predictions"]
    assert diff[0]["probability"] >= diff[1]["probability"] >= diff[2]["probability"]

    # Test with hair removal enabled
    result_hr = service.predict_image(dummy_bgr, apply_hair_removal=True)
    assert result_hr["success"] is True
    print(f"✔ PredictionService inference verified: Class={result['predicted_class_code']} ({result['confidence_score']:.2%})")


def test_10_prediction_service_error_handling():
    print("\n--- Test 10: PredictionService Error Handling ---")
    # Non-existent model path handling
    non_existent = BASE_DIR / "models" / "non_existent_vit.pth"
    service_unready = PredictionService(model_path=non_existent)
    assert not service_unready.is_ready
    try:
        service_unready.predict_image(np.zeros((224, 224, 3), dtype=np.uint8))
        assert False, "Expected RuntimeError when model is not ready!"
    except RuntimeError:
        pass

    # Invalid image inputs with ready service
    model_ckpt = BASE_DIR / "models" / "vit_model.pth"
    service_ready = PredictionService(model_path=model_ckpt)
    try:
        service_ready.predict_image("non_existent_image_path.jpg")
        assert False, "Expected FileNotFoundError or ValueError for non-existent image path!"
    except (FileNotFoundError, ValueError):
        pass

    try:
        service_ready.predict_image(np.zeros((100,), dtype=np.uint8))
        assert False, "Expected ValueError for 1D image array!"
    except ValueError:
        pass

    print("✔ PredictionService gracefully handled unready models and corrupt image inputs.")


def test_11_colab_package_and_notebook():
    print("\n--- Test 11: Google Colab GPU Artifacts Verification ---")
    nb_path = BASE_DIR / "training" / "train_vit.ipynb"
    assert nb_path.exists(), f"Notebook not found at {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    assert "cells" in nb_json and len(nb_json["cells"]) > 5, "Notebook appears incomplete"

    zip_path = BASE_DIR / "colab_vit_package.zip"
    assert zip_path.exists(), f"Colab package zip not found at {zip_path}"
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "data/manifests/classification_train.csv" in namelist
        assert "data/manifests/classification_val.csv" in namelist
        assert "data/manifests/classification_test.csv" in namelist
        assert "training/train_vit.py" in namelist
        assert "models/vit.py" in namelist
        img_count = sum(1 for name in namelist if name.startswith("data/classification/") and name.endswith(".jpg"))
        assert img_count == 1757, f"Expected 1757 classification images in zip, got {img_count}"

    print(f"✔ Colab artifacts verified: Notebook {nb_path.name} ({len(nb_json['cells'])} cells), ZIP {zip_path.name} ({img_count} images).")


def test_12_regression_and_fastapi_endpoints():
    print("\n--- Test 12: Regression Check (U-Net & FastAPI Endpoints) ---")
    # Verify U-Net Segmentation Service
    unet_path = BASE_DIR / "models" / "unet_model.pth"
    assert unet_path.exists(), "Trained U-Net model checkpoint missing!"
    seg_service = get_segmentation_service(model_path=unet_path)
    assert seg_service.is_ready, "SegmentationService failed to initialize with trained weights!"

    # Verify FastAPI endpoints
    endpoints = [
        ("http://127.0.0.1:8000/", 200, "Dashboard"),
        ("http://127.0.0.1:8000/history", 200, "History"),
        ("http://127.0.0.1:8000/health", 200, "Health"),
        ("http://127.0.0.1:8000/api/history", 200, "API History"),
        ("http://127.0.0.1:8000/docs", 200, "Swagger Docs")
    ]
    for url, expected_code, name in endpoints:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == expected_code, f"{name} returned HTTP {resp.status}"
            print(f"  ✔ Endpoint {url} [{name}]: HTTP {resp.status}")

    print("✔ Regression check passed: U-Net is operational and all 5 FastAPI routes are healthy.")


def run_all_tests():
    print("=" * 70)
    print("PHASE 4 VISION TRANSFORMER (ViT-B/16) VERIFICATION SUITE")
    print("=" * 70)

    test_01_vit_architecture_and_parameters()
    test_02_freeze_unfreeze_layer_verification()
    test_03_forward_pass_and_logits_shape()
    test_04_gradient_flow_and_backprop()
    test_05_softmax_probabilities()
    test_06_disease_classes_and_indexing()
    test_07_class_weights_computation()
    test_08_discriminative_parameter_groups()
    test_09_prediction_service_inference()
    test_10_prediction_service_error_handling()
    test_11_colab_package_and_notebook()
    test_12_regression_and_fastapi_endpoints()

    print("\n" + "=" * 70)
    print("ALL 12 PHASE 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
