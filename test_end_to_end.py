"""
Comprehensive End-to-End System Integration Test Suite
Skin Disease Detection with Treatment Cost Estimation

Tests:
1. System Health & Lifespan Readiness (GET /health)
2. Image Quality Assessment Endpoint (POST /api/check-quality)
3. Educational Disease Library Endpoint (GET /api/disease-library)
4. Invalid Upload Rejection (HTTP 400)
5. Full Multi-Modal AI Inference Pipeline (POST /api/predict)
   - U-Net Segmentation (threshold 0.70)
   - ViT-B/16 Classification (7 categories)
   - XAI Attention Rollout Map
   - Prototype Severity & Concern Indicator
   - Indicative Treatment Cost Breakdown (INR)
   - Educational Disease Info Attachment
   - ReportLab Redesigned PDF Report
6. Verification of Output Artifacts on Disk (PNGs + PDF)
7. SQLite Database Persistence
8. Reports / History API (GET /api/history & GET /api/records)
9. Single Record Detail API (GET /api/results/{id})
10. PDF Report Download Endpoint (GET /api/report/{id})
11. Web Navigation Views (/, /library, /reports, /history, /about)
12. Safe Record Deletion Endpoint (DELETE /api/records/{id})
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from main import app
from database import DB_PATH, get_db_connection

client = None


def test_1_health_endpoint():
    print("\n--- Test 1: GET /health ---")
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "models" in data
    assert data["models"]["unet_segmentation"]["ready"] is True
    assert data["models"]["vit_classification"]["ready"] is True
    assert data["models"]["attention_rollout"]["ready"] is True
    print(f"  [PASS] Health check passed: models ready, threshold={data['models']['unet_segmentation']['optimal_threshold']}")


def test_2_quality_check_endpoint():
    print("\n--- Test 2: POST /api/check-quality ---")
    sample_images = list((ROOT_DIR / "data" / "classification" / "images").glob("*.jpg"))
    assert len(sample_images) > 0, "No sample images found!"
    test_img_path = sample_images[0]

    with open(test_img_path, "rb") as f:
        files = {"file": (test_img_path.name, f, "image/jpeg")}
        response = client.post("/api/check-quality", files=files)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    res = response.json()
    assert "status" in res
    assert "details" in res
    assert "sharpness_score" in res["details"]
    assert "brightness_score" in res["details"]
    print(f"  [PASS] Image quality assessed: status={res['status']}, sharpness={res['details']['sharpness_score']}")


def test_3_disease_library_endpoint():
    print("\n--- Test 3: GET /api/disease-library ---")
    response = client.get("/api/disease-library")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "diseases" in data
    assert len(data["diseases"]) in [7, 16]
    codes = {d["code"] for d in data["diseases"]}
    assert {"MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"}.issubset(codes)
    print(f"  [PASS] Disease library verified across {len(codes)} categories (including all 7 dermoscopic classes): {codes}")


def test_4_invalid_upload_rejection():
    print("\n--- Test 4: Invalid Upload Rejection (HTTP 400) ---")
    files = {"file": ("test.txt", b"This is not an image", "text/plain")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print(f"  [PASS] Non-image rejected properly: {response.json()}")


def test_5_end_to_end_prediction():
    print("\n--- Test 5: POST /api/predict on Genuine Dermoscopic Image ---")
    sample_images = list((ROOT_DIR / "data" / "classification" / "images").glob("*.jpg"))
    test_img_path = sample_images[0]
    print(f"  Testing with image: {test_img_path.name}")

    with open(test_img_path, "rb") as f:
        files = {"file": (test_img_path.name, f, "image/jpeg")}
        response = client.post("/api/predict", files=files)

    assert response.status_code == 200, f"Prediction failed with {response.status_code}: {response.text}"
    res = response.json()
    
    assert res["success"] is True
    assert "record_id" in res
    assert "reference_id" in res
    assert res["reference_id"].startswith("REF-")

    # Classification check
    pred = res["prediction"]
    assert "disease_name" in pred
    assert "disease_code" in pred
    assert "confidence_percentage" in pred
    assert len(pred["top_3_predictions"]) == 3
    print(f"  Prediction: {pred['disease_name']} ({pred['disease_code']}) with {pred['confidence_percentage']} confidence")

    # Segmentation check
    seg = res["segmentation"]
    assert "relative_lesion_area_percentage" in seg
    assert seg["threshold_applied"] == 0.70
    print(f"  Segmentation: Area coverage = {seg['relative_lesion_area_percentage']:.2f}% (Threshold: {seg['threshold_applied']})")

    # Attention rollout check
    attn = res["attention_map"]
    assert attn["available"] is True
    assert attn["attention_path"] is not None

    # Severity check
    sev = res["severity"]
    assert sev["level"] in ["Low Concern", "Moderate Concern", "High Concern", "Undetermined"]
    print(f"  Severity: {sev['level']} - {sev['summary']}")

    # Cost estimation check
    cost = res["cost_estimation"]
    assert "formatted_total_range" in cost
    assert "breakdown" in cost
    assert len(cost["breakdown"]) == 5
    print(f"  Treatment Cost: {cost['formatted_total_range']}")

    # Educational disease info check
    assert "disease_info" in res
    if res["disease_info"]:
        assert "short_description" in res["disease_info"]
        print(f"  [PASS] Disease info attached: {res['disease_info']['name']}")

    # Disk file verification
    img_paths = res["image_paths"]
    for key in ["original", "mask", "overlay", "attention"]:
        web_path = img_paths[key]
        assert web_path.startswith("uploads/"), f"Unexpected path: {web_path}"
        disk_path = ROOT_DIR / web_path
        assert disk_path.exists(), f"Output file does not exist on disk: {disk_path}"
        assert disk_path.stat().st_size > 0, f"File is 0 bytes: {disk_path}"
    print("  [PASS] All 4 output images verified on disk.")

    # PDF verification
    assert "report_url" in res
    ref_id = res["reference_id"]
    pdf_disk = ROOT_DIR / "reports" / f"report_{ref_id}.pdf"
    assert pdf_disk.exists(), f"PDF report does not exist on disk: {pdf_disk}"
    assert pdf_disk.stat().st_size > 5000, f"PDF report too small: {pdf_disk.stat().st_size} bytes"
    print(f"  [PASS] ReportLab PDF report verified on disk ({pdf_disk.stat().st_size} bytes).")

    return res


def test_6_database_and_history(pred_result):
    print("\n--- Test 6: Database & History Verification ---")
    rec_id = pred_result["record_id"]
    ref_id = pred_result["reference_id"]

    # Direct SQLite check
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, reference_id, predicted_disease, severity_grade FROM analysis_records WHERE id = ?", (rec_id,))
        row = cursor.fetchone()
        assert row is not None, f"Record {rec_id} not found in database!"
        assert row["reference_id"] == ref_id
        assert row["predicted_disease"] == pred_result["prediction"]["disease_name"]
        print(f"  [PASS] SQLite row verified: ID={row['id']}, Ref={row['reference_id']}, Disease={row['predicted_disease']}")

    # GET /api/history check
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    records = hist_data["records"]
    matching = [r for r in records if r["id"] == rec_id]
    assert len(matching) == 1, f"Record {rec_id} not found in history endpoint output!"
    print(f"  [PASS] GET /api/history verified ({hist_data['total']} total records).")

    # GET /api/results/{id} check
    det_res = client.get(f"/api/results/{rec_id}")
    assert det_res.status_code == 200
    det_data = det_res.json()
    assert det_data["id"] == rec_id
    assert det_data["reference_id"] == ref_id
    print("  [PASS] GET /api/results/{id} verified.")

    # GET /api/report/{id} check
    rep_res = client.get(f"/api/report/{rec_id}")
    assert rep_res.status_code == 200
    assert rep_res.headers["content-type"] == "application/pdf"
    assert len(rep_res.content) > 5000
    print(f"  [PASS] GET /api/report/{rec_id} successfully downloaded {len(rep_res.content)} bytes of PDF.")


def test_7_web_pages_navigation():
    print("\n--- Test 7: Web Navigation Views ---")
    pages = [
        ("/", 200, "Skin Analysis"),
        ("/library", 200, "Disease Library"),
        ("/reports", 200, "Analysis Reports"),
        ("/history", 200, "reports"),  # Aliased or redirect
        ("/about", 200, "About the Project"),
        ("/docs", 200, "Swagger UI")
    ]
    for url, exp_code, kw in pages:
        res = client.get(url)
        assert res.status_code == exp_code, f"{url} returned {res.status_code}, expected {exp_code}"
        assert kw.lower() in res.text.lower(), f"{kw} not found in {url} body"
        print(f"  ✔ View {url} returned HTTP {res.status_code} with keyword '{kw}'")


def test_8_safe_record_deletion():
    print("\n--- Test 8: Safe Record Deletion (DELETE /api/records/{id}) ---")
    # First create a disposable test record
    sample_images = list((ROOT_DIR / "data" / "classification" / "images").glob("*.jpg"))
    test_img_path = sample_images[1] if len(sample_images) > 1 else sample_images[0]

    with open(test_img_path, "rb") as f:
        files = {"file": (test_img_path.name, f, "image/jpeg")}
        create_res = client.post("/api/predict", files=files)

    assert create_res.status_code == 200
    created_data = create_res.json()
    del_id = created_data["record_id"]
    del_ref = created_data["reference_id"]
    print(f"  Created temporary test record: ID={del_id}, Ref={del_ref}")

    # Check generated files exist before deletion
    pdf_path = ROOT_DIR / "reports" / f"report_{del_ref}.pdf"
    assert pdf_path.exists(), f"PDF did not generate: {pdf_path}"

    # Call DELETE endpoint
    del_res = client.delete(f"/api/records/{del_id}")
    assert del_res.status_code == 200, f"Expected 200, got {del_res.status_code}: {del_res.text}"
    del_json = del_res.json()
    assert del_json["success"] is True
    print(f"  DELETE response: {del_json['message']}")

    # Verify SQLite row was deleted
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM analysis_records WHERE id = ?", (del_id,))
        assert cursor.fetchone() is None, f"Record {del_id} still exists in DB after deletion!"

    # Verify associated PDF file was unlinked
    assert not pdf_path.exists(), f"PDF file was not unlinked on deletion: {pdf_path}"

    # Verify shared models and dataset are intact
    assert (ROOT_DIR / "models" / "unet_model.pth").exists(), "CRITICAL: unet_model.pth was accidentally deleted!"
    assert (ROOT_DIR / "models" / "vit_model.pth").exists(), "CRITICAL: vit_model.pth was accidentally deleted!"
    assert (ROOT_DIR / "data" / "treatment_costs.json").exists(), "CRITICAL: treatment_costs.json missing!"
    print("  [PASS] Record deleted safely: database row removed, associated output files unlinked, shared models preserved.")


def run_all_tests():
    print("======================================================================")
    print("Skin Disease Detection — Comprehensive End-to-End System Test")
    print("======================================================================")
    with TestClient(app) as test_client:
        global client
        client = test_client
        test_1_health_endpoint()
        test_2_quality_check_endpoint()
        test_3_disease_library_endpoint()
        test_4_invalid_upload_rejection()
        pred_res = test_5_end_to_end_prediction()
        test_6_database_and_history(pred_res)
        test_7_web_pages_navigation()
        test_8_safe_record_deletion()
    print("\n======================================================================")
    print("ALL 8 INTEGRATION TESTS PASSED SUCCESSFULLY! (100% OPERATIONAL)")
    print("======================================================================")


if __name__ == "__main__":
    run_all_tests()
