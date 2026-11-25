"""
Comprehensive Phase 3 Verification Suite:
Tests:
1. U-Net model initialization and parameter count.
2. Encoder-decoder output tensor dimensions: (B, 1, 256, 256).
3. Combined BCE + Dice loss computation.
4. Forward and backward pass gradient propagation.
5. CPU inference compatibility.
6. Binary mask generation and probability thresholding.
7. SegmentationService overlay and contour generation.
8. Relative lesion area calculation and labelling.
9. Missing model weight graceful error handling.
10. Live FastAPI endpoints and frontend pages.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.unet import build_unet, ResNet18UNet
from training.train_unet import CombinedBCEDiceLoss, DiceLoss, compute_metrics
from services.segmentation import SegmentationService, get_segmentation_service

DATA_DIR = BASE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"


def test_01_model_initialization_and_params():
    print("\n--- Test 1: Model Initialization & Parameter Counts ---")
    model = build_unet(pretrained=False)
    params = model.get_parameter_count()
    assert params["total_parameters"] > 14_000_000, f"Expected >14M params, got {params['total_parameters']}"
    assert params["trainable_parameters"] == params["total_parameters"]
    print(f"✔ U-Net initialized successfully: {params['total_parameters']:,} parameters.")


def test_02_encoder_decoder_output_dimensions():
    print("\n--- Test 2: Encoder-Decoder Output Dimensions (B, 1, 256, 256) ---")
    model = build_unet(pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"
    assert out.dtype == torch.float32
    print(f"✔ Output dimensions verified: shape={out.shape}, dtype={out.dtype}")


def test_03_loss_calculation():
    print("\n--- Test 3: Combined BCE + Dice Loss Computation ---")
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    logits = torch.randn(4, 1, 256, 256)
    targets = (torch.rand(4, 1, 256, 256) > 0.5).float()

    total_loss, bce, dice = criterion(logits, targets)

    assert total_loss.ndim == 0, "Total loss must be a scalar"
    assert total_loss.item() > 0, "Total loss must be positive"
    assert bce.item() > 0, "BCE loss must be positive"
    assert dice.item() >= 0, "Dice loss must be non-negative"

    # Verify combined formula
    expected_total = 0.5 * bce.item() + 0.5 * dice.item()
    assert abs(total_loss.item() - expected_total) < 1e-5
    print(f"✔ Loss verified: Total={total_loss.item():.4f} (BCE={bce.item():.4f}, Dice={dice.item():.4f})")


def test_04_forward_backward_passes():
    print("\n--- Test 4: Forward and Backward Passes ---")
    model = build_unet(pretrained=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CombinedBCEDiceLoss()

    images = torch.randn(2, 3, 256, 256)
    masks = torch.ones(2, 1, 256, 256)

    optimizer.zero_grad()
    logits = model(images)
    total_loss, _, _ = criterion(logits, masks)
    total_loss.backward()

    # Verify that both encoder and decoder parameters have gradients
    has_encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder1.parameters())
    has_decoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder1.parameters())
    assert has_encoder_grad, "Encoder has no gradients"
    assert has_decoder_grad, "Decoder has no gradients"

    optimizer.step()
    print("✔ Full forward and backward pass with gradient updates verified across encoder & decoder.")


def test_05_cpu_inference_compatibility():
    print("\n--- Test 5: CPU-Compatible Inference ---")
    device = torch.device("cpu")
    model = build_unet(pretrained=False, device=device)
    model.eval()

    inp = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        out = model(inp)
    assert out.device.type == "cpu"
    assert out.shape == (1, 1, 256, 256)
    print("✔ CPU inference verified without requiring CUDA acceleration.")


def test_06_binary_mask_generation():
    print("\n--- Test 6: Binary Mask Generation & Thresholding ---")
    model = build_unet(pretrained=False)
    dummy_input = torch.randn(2, 3, 256, 256)

    for th in [0.3, 0.5, 0.7]:
        probs, binary = model.predict_mask(dummy_input, threshold=th)
        assert probs.shape == (2, 1, 256, 256)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert binary.shape == (2, 1, 256, 256)
        unique_vals = set(torch.unique(binary).tolist())
        assert unique_vals.issubset({0.0, 1.0}), f"Non-binary mask values: {unique_vals}"
        print(f"✔ Threshold {th:.1f} verified: unique values = {unique_vals}")


def test_07_segmentation_service_and_overlay():
    print("\n--- Test 7: Segmentation Inference Service & Overlay Generation ---")
    # Use existing genuine sample from test manifest
    seg_test = pd.read_csv(MANIFESTS_DIR / "segmentation_test.csv")
    sample_img_path = DATA_DIR / seg_test.iloc[0]["image_path"]
    assert sample_img_path.exists(), f"Sample image {sample_img_path} not found"

    service = get_segmentation_service(force_reload=True)
    result = service.segment_image(sample_img_path, apply_hair_removal=True, threshold=0.5)

    assert result["success"] is True
    assert "relative_lesion_area_pct" in result
    assert "binary_mask" in result
    assert "overlay_image" in result
    assert isinstance(result["overlay_image"], np.ndarray)

    orig_h, orig_w = result["original_dimensions"]
    assert result["binary_mask"].shape == (orig_h, orig_w)
    assert result["overlay_image"].shape == (orig_h, orig_w, 3)

    print(f"✔ Service output verified on genuine image ({seg_test.iloc[0]['image_id']}):")
    print(f"   Original Dimensions: {result['original_dimensions']}")
    print(f"   Relative Lesion Area: {result['relative_lesion_area_pct']}%")
    print(f"   Mask Shape: {result['binary_mask'].shape}")
    print(f"   Overlay Shape: {result['overlay_image'].shape}")


def test_08_lesion_area_calculation_and_labelling():
    print("\n--- Test 8: Lesion Area Calculation & Labelling ---")
    service = get_segmentation_service(force_reload=True)
    seg_test = pd.read_csv(MANIFESTS_DIR / "segmentation_test.csv")
    sample_img_path = DATA_DIR / seg_test.iloc[0]["image_path"]

    result = service.segment_image(sample_img_path)

    # Verify area calculation
    expected_pct = (result["lesion_pixels"] / result["total_pixels"]) * 100.0
    assert abs(result["relative_lesion_area_pct"] - expected_pct) < 0.05
    assert "not physical size or clinical severity" in result["area_coverage_note"]
    print(f"✔ Relative lesion area verified: {result['relative_lesion_area_pct']}% of total image.")
    print(f"✔ Explicit clinical disclaimer note verified: '{result['area_coverage_note']}'")


def test_09_missing_model_weight_handling():
    print("\n--- Test 9: Missing Model Weight Graceful Error Handling ---")
    bogus_path = BASE_DIR / "models" / "non_existent_fake_weights.pth"
    bad_service = SegmentationService(model_path=bogus_path)

    assert bad_service.is_ready is False
    assert "not found" in bad_service.error_message
    print(f"✔ Gracefully flagged missing weights: '{bad_service.error_message[:65]}...'")

    try:
        bad_service.segment_image(np.zeros((100, 100, 3), dtype=np.uint8))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"✔ RuntimeError correctly raised on inference attempt: '{str(e)[:65]}...'")


def test_10_fastapi_endpoints_and_frontend():
    print("\n--- Test 10: Live FastAPI Endpoints & UI Pages ---")
    endpoints = ["/", "/history", "/health", "/api/history", "/docs"]
    for ep in endpoints:
        url = f"http://127.0.0.1:8000{ep}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200, f"Endpoint {ep} returned {resp.status}"
        print(f"✔ Endpoint {ep} passed with status 200")


if __name__ == "__main__":
    test_01_model_initialization_and_params()
    test_02_encoder_decoder_output_dimensions()
    test_03_loss_calculation()
    test_04_forward_backward_passes()
    test_05_cpu_inference_compatibility()
    test_06_binary_mask_generation()
    test_07_segmentation_service_and_overlay()
    test_08_lesion_area_calculation_and_labelling()
    test_09_missing_model_weight_handling()
    test_10_fastapi_endpoints_and_frontend()
    print("\n" + "=" * 65)
    print("ALL 10 PHASE 3 VERIFICATION TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)
