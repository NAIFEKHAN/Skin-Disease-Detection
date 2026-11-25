"""
Main FastAPI Monolithic Application.
AI-Based Skin Disease Detection with Treatment Cost Estimation.
Complete Pipeline: U-Net Lesion Segmentation, ViT-B/16 Classification,
XAI Attention Rollout, Prototype Severity, INR Cost Estimation, SQLite, and PDF Reporting.
"""

import os
import cv2
import uuid
import json
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Union, List

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Core application modules
from database import (
    init_db,
    save_analysis,
    get_all_records,
    get_record_by_id,
    get_record_by_ref_or_id,
    update_record_report_path,
    delete_record_and_files
)
from services.segmentation import get_segmentation_service, SegmentationService
from services.prediction import get_prediction_service, PredictionService
from services.attention import get_attention_service, AttentionRolloutService
from services.severity import get_severity_service, SeverityService
from services.cost_estimation import get_cost_service, CostEstimationService
from services.report import get_report_generator, ReportGenerator
from services.quality_check import assess_image_quality

# Directory configuration
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
INPUTS_DIR = UPLOADS_DIR / "inputs"
OUTPUTS_DIR = UPLOADS_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure all working directories exist
for folder in [FRONTEND_DIR, UPLOADS_DIR, INPUTS_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Service singletons
seg_service: Optional[SegmentationService] = None
pred_service: Optional[PredictionService] = None
attn_service: Optional[AttentionRolloutService] = None
sev_service: Optional[SeverityService] = None
cost_service: Optional[CostEstimationService] = None
report_gen: Optional[ReportGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager: pre-loads trained deep learning models once
    at startup on CPU to guarantee fast, zero-delay inference during HTTP requests.
    """
    global seg_service, pred_service, attn_service, sev_service, cost_service, report_gen

    print("\n" + "=" * 70)
    print("STARTING APPLICATION SERVICES & PRE-LOADING DEEP LEARNING MODELS")
    print("=" * 70)

    # 1. Initialize SQLite Database
    init_db()

    # 2. Preload U-Net Lesion Segmentation Service
    print("• Loading ResNet18-UNet Lesion Segmentation Service (threshold = 0.70)...")
    seg_service = get_segmentation_service()
    if seg_service.is_ready:
        print("  ✔ U-Net weights loaded successfully.")
    else:
        print(f"  ⚠ U-Net warning: {seg_service.error_message}")

    # 3. Preload ViT-B/16 Disease Classification Service
    print("• Loading Vision Transformer (ViT-B/16) Classification Service...")
    pred_service = get_prediction_service()
    if pred_service.is_ready:
        print("  ✔ ViT-B/16 weights loaded successfully.")
    else:
        print(f"  ⚠ ViT warning: {pred_service.error_message}")

    # 4. Preload XAI Attention Rollout Service (sharing ViT backbone)
    print("• Initializing XAI Attention Rollout Service...")
    attn_service = get_attention_service(model=pred_service.model if pred_service else None)
    print("  ✔ XAI Attention Rollout Service ready.")

    # 5. Initialize Rule-Based Severity & Cost Estimation Services
    sev_service = get_severity_service()
    cost_service = get_cost_service()
    report_gen = get_report_generator()

    print("✔ All system services initialized successfully.")
    print("=" * 70 + "\n")

    yield

    print("Shutting down application services...")


# Create FastAPI app
app = FastAPI(
    title="AI-Based Skin Disease Detection with Treatment Cost Estimation",
    description="Monolithic academic prototype integrating deep learning lesion segmentation, Vision Transformer disease classification, Explainable AI attention rollout, prototype severity rating, and illustrative INR treatment cost estimation.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file endpoints
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


# ==============================================================================
# SYSTEM & HEALTH ENDPOINTS
# ==============================================================================

@app.get("/health", tags=["System"])
def health_check():
    """System health check endpoint verifying server status, models and database."""
    db_status = "connected"
    try:
        _ = get_all_records(limit=1)
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "app_name": "AI-Based Skin Disease Detection with Treatment Cost Estimation",
        "version": "2.0.0",
        "database": db_status,
        "models": {
            "unet_segmentation": {
                "ready": seg_service.is_ready if seg_service else False,
                "optimal_threshold": getattr(seg_service, "optimal_threshold", 0.70) if seg_service else 0.70
            },
            "vit_classification": {
                "ready": pred_service.is_ready if pred_service else False,
                "classes": getattr(pred_service, "classes", []) if pred_service else []
            },
            "attention_rollout": {
                "ready": attn_service.is_ready if attn_service else False
            },
            "common_skin_classification": {
                "installed": (BASE_DIR / "models" / "common_skin_model.pth").exists(),
                "status": "pending_approval",
                "ready": False,
                "classes": [
                    "acne", "allergic_contact_dermatitis", "eczema", "impetigo",
                    "psoriasis", "ringworm", "scabies", "seborrheic_dermatitis", "urticaria"
                ]
            }
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ==============================================================================
# FRONTEND VIEWS
# ==============================================================================

@app.get("/", tags=["Frontend"])
def serve_dashboard():
    """Serves the main Skin Image Analysis dashboard page."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(str(index_file))


@app.get("/library", tags=["Frontend"])
def serve_library():
    """Serves the Disease Library page."""
    library_file = FRONTEND_DIR / "library.html"
    if not library_file.exists():
        raise HTTPException(status_code=404, detail="Disease Library library.html not found")
    return FileResponse(str(library_file))


@app.get("/reports", tags=["Frontend"])
@app.get("/history", tags=["Frontend"])
def serve_reports():
    """Serves the Reports page (with backward-compatible /history route)."""
    reports_file = FRONTEND_DIR / "reports.html"
    if not reports_file.exists():
        # Fallback to history.html if reports.html is pending
        history_file = FRONTEND_DIR / "history.html"
        if history_file.exists():
            return FileResponse(str(history_file))
        raise HTTPException(status_code=404, detail="Reports page not found")
    return FileResponse(str(reports_file))


@app.get("/about", tags=["Frontend"])
def serve_about():
    """Serves the About and System Specifications page."""
    about_file = FRONTEND_DIR / "about.html"
    if not about_file.exists():
        raise HTTPException(status_code=404, detail="About page not found")
    return FileResponse(str(about_file))


# ==============================================================================
# PREDICTION & ANALYSIS PIPELINE
# ==============================================================================

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.post("/api/predict", tags=["Analysis"])
async def predict_lesion(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    apply_hair_removal: bool = Form(False)
):
    """
    End-to-end multimodal dermatological screening analysis pipeline:
    1. Validates and saves uploaded dermoscopic image.
    2. Runs U-Net lesion segmentation (mask, boundary overlay, relative coverage %).
    3. Runs Vision Transformer (ViT-B/16) disease classification (probabilities, top-3 differential).
    4. Generates genuine XAI Attention Rollout heatmap.
    5. Determines prototype severity risk indicator.
    6. Calculates illustrative INR treatment cost breakdown.
    7. Stores analysis in SQLite database.
    8. Pre-compiles downloadable PDF clinical report.
    """
    upload = file or image
    if not upload or not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided. Please upload a valid JPG, JPEG, or PNG dermoscopic image."
        )

    ext = Path(upload.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only JPG, JPEG, and PNG dermoscopic images are accepted."
        )

    # Read and validate payload size
    content = await upload.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum upload size limit of 15 MB ({len(content)/(1024*1024):.1f} MB)."
        )

    # Generate unique analysis reference
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_uuid = uuid.uuid4().hex[:6].upper()
    ref_id = f"REF-{datetime.now().strftime('%Y%m%d')}-{ref_uuid}"

    input_filename = f"{timestamp_str}_{ref_uuid}_{upload.filename}"
    input_path = INPUTS_DIR / input_filename

    # Save original uploaded image
    with open(input_path, "wb") as f:
        f.write(content)

    # Verify OpenCV decoding
    img_test = cv2.imread(str(input_path))
    if img_test is None:
        if input_path.exists():
            input_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image data. The file cannot be decoded as an image."
        )

    # 1. Run U-Net Lesion Segmentation
    if not seg_service or not seg_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="U-Net segmentation service is unavailable. Check model checkpoints."
        )

    mask_filename = f"{ref_id}_mask.png"
    overlay_filename = f"{ref_id}_overlay.png"
    mask_path = OUTPUTS_DIR / mask_filename
    overlay_path = OUTPUTS_DIR / overlay_filename

    try:
        seg_result = seg_service.segment_image(
            image_input=input_path,
            apply_hair_removal=apply_hair_removal,
            threshold=0.70
        )
        # Save mask and overlay images
        cv2.imwrite(str(mask_path), seg_result["binary_mask"])
        cv2.imwrite(str(overlay_path), cv2.cvtColor(seg_result["overlay_image"], cv2.COLOR_RGB2BGR))
        relative_area = seg_result["relative_lesion_area_pct"]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lesion segmentation failed: {str(e)}"
        )

    # 2. Run ViT-B/16 Disease Classification
    if not pred_service or not pred_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vision Transformer classification service is unavailable. Check model checkpoints."
        )

    try:
        pred_result = pred_service.predict_image(
            image_input=input_path,
            apply_hair_removal=apply_hair_removal
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Disease classification failed: {str(e)}"
        )

    # 3. Run XAI Attention Rollout
    attention_filename = f"{ref_id}_attention.png"
    attention_path = OUTPUTS_DIR / attention_filename
    attention_relative_path = f"uploads/outputs/{attention_filename}"

    try:
        if attn_service and attn_service.is_ready:
            attn_result = attn_service.generate_attention_map(
                image_input=input_path,
                output_path=attention_path
            )
            if not attn_result.get("success"):
                attention_relative_path = None
        else:
            attention_relative_path = None
    except Exception:
        attention_relative_path = None

    # 4. Compute Prototype Severity
    sev_result = sev_service.estimate_severity(
        predicted_class=pred_result["predicted_class_code"],
        confidence_score=pred_result["confidence_score"],
        top_3_predictions=pred_result["top_3_predictions"],
        relative_lesion_area=relative_area
    )

    # 5. Compute Illustrative Treatment Cost Breakdown (INR)
    cost_result = cost_service.estimate_cost(predicted_class=pred_result["predicted_class_code"])

    # Relative paths for frontend serving
    rel_input_path = f"uploads/inputs/{input_filename}"
    rel_mask_path = f"uploads/outputs/{mask_filename}"
    rel_overlay_path = f"uploads/outputs/{overlay_filename}"

    # 6. Save in SQLite Database
    record = save_analysis(
        reference_id=ref_id,
        image_path=rel_input_path,
        mask_path=rel_mask_path,
        overlay_path=rel_overlay_path,
        attention_map_path=attention_relative_path,
        predicted_disease=pred_result["predicted_disease_name"],
        predicted_class_code=pred_result["predicted_class_code"],
        confidence_score=pred_result["confidence_score"],
        severity_grade=sev_result["level"],
        severity_details=sev_result,
        lesion_area_percentage=relative_area,
        top_3_predictions=pred_result["top_3_predictions"],
        all_probabilities=pred_result["class_probabilities"],
        estimated_cost_min=cost_result["total_min"],
        estimated_cost_max=cost_result["total_max"],
        currency="INR",
        cost_breakdown=cost_result,
        report_path=None
    )

    # 7. Generate PDF Report
    pdf_filename = f"report_{ref_id}.pdf"
    try:
        pdf_full_path = report_gen.generate_report(
            record_data={
                **record,
                "predicted_disease_name": pred_result["predicted_disease_name"],
                "predicted_class_code": pred_result["predicted_class_code"],
                "confidence_percentage": pred_result["confidence_percentage"],
                "relative_lesion_area": relative_area,
                "severity": sev_result,
                "top_3_predictions": pred_result["top_3_predictions"],
                "cost_estimate": cost_result,
                "image_path": rel_input_path,
                "mask_path": rel_mask_path,
                "overlay_path": rel_overlay_path,
                "attention_path": attention_relative_path
            },
            filename=pdf_filename
        )
        rel_pdf_path = f"reports/{pdf_filename}"
        update_record_report_path(record["id"], rel_pdf_path)
        record["report_path"] = rel_pdf_path
    except Exception as e:
        print(f"Warning: PDF generation failed: {e}")
        rel_pdf_path = None

    # 8. Load educational disease info
    disease_educational_info = None
    try:
        info_file = BASE_DIR / "data" / "disease_info.json"
        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                d_data = json.load(f)
                for d in d_data.get("diseases", []):
                    if d["code"].upper() == pred_result["predicted_class_code"].upper():
                        disease_educational_info = d
                        break
    except Exception as e:
        print(f"Warning: Failed to load disease info: {e}")

    # Return structured comprehensive response
    return {
        "success": True,
        "reference_id": ref_id,
        "record_id": record["id"],
        "created_at": record.get("created_at"),
        "prediction": {
            "disease_code": pred_result["predicted_class_code"],
            "disease_name": pred_result["predicted_disease_name"],
            "confidence_score": pred_result["confidence_score"],
            "confidence_percentage": pred_result["confidence_percentage"],
            "is_uncertain": pred_result["is_uncertain"],
            "uncertainty_note": pred_result["uncertainty_note"],
            "top_3_predictions": pred_result["top_3_predictions"],
            "class_probabilities": pred_result["class_probabilities"]
        },
        "segmentation": {
            "relative_lesion_area_percentage": relative_area,
            "mask_path": rel_mask_path,
            "overlay_path": rel_overlay_path,
            "threshold_applied": 0.70
        },
        "attention_map": {
            "available": attention_relative_path is not None,
            "attention_path": attention_relative_path,
            "disclaimer": "Model explainability rollout visualization; not clinical diagnostic validation."
        },
        "severity": sev_result,
        "cost_estimation": cost_result,
        "disease_info": disease_educational_info,
        "image_paths": {
            "original": rel_input_path,
            "mask": rel_mask_path,
            "overlay": rel_overlay_path,
            "attention": attention_relative_path
        },
        "report_url": f"/api/report/{record['id']}",
        "medical_disclaimer": (
            "For educational use only. Not a medical diagnosis. Consult a qualified dermatologist. "
            "On held-out dermoscopic evaluation, overall accuracy is 64.59% and Melanoma recall is 47.73%."
        )
    }


# ==============================================================================
# QUALITY CHECK & EDUCATIONAL LIBRARY ENDPOINTS
# ==============================================================================

@app.post("/api/check-quality", tags=["Quality"])
async def check_image_quality(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Lightweight image quality checker evaluating blur, illumination, and resolution
    using OpenCV heuristics to provide constructive user feedback before analysis.
    """
    upload = file or image
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No image file provided for quality check.")

    contents = await upload.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 15 MB size limit.")

    result = assess_image_quality(contents)
    return result


@app.get("/api/disease-library", tags=["Library"])
def get_disease_library(mode: Optional[str] = None):
    """Returns educational summaries for supported skin lesion and condition categories."""
    info_file = BASE_DIR / "data" / "disease_info.json"
    if not info_file.exists():
        raise HTTPException(status_code=404, detail="Disease library information not found.")
    try:
        with open(info_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        diseases = data.get("diseases", [])
        if mode == "dermoscopy":
            diseases = [d for d in diseases if d.get("mode") == "dermoscopy"]
        elif mode == "common_skin":
            diseases = [d for d in diseases if d.get("mode") == "common_skin"]
        return {"diseases": diseases, "total_categories": len(diseases)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read disease library: {str(e)}")


# ==============================================================================
# HISTORY & RECORD RETRIEVAL ENDPOINTS
# ==============================================================================

@app.get("/api/history", tags=["History"])
@app.get("/api/records", tags=["History"])
def get_history_records(limit: int = 50, offset: int = 0):
    """Retrieves standardized historical analysis records from SQLite."""
    try:
        records = get_all_records(limit=limit, offset=offset)
        return {
            "total": len(records),
            "records": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.get("/api/results/{record_id}", tags=["Analysis"])
@app.get("/api/records/{record_id}", tags=["Records"])
def get_analysis_result(record_id: Union[int, str]):
    """Retrieves detailed results for a specific analysis record by ID or Reference ID."""
    record = get_record_by_ref_or_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Analysis record '{record_id}' not found.")
    return record


@app.delete("/api/records/{record_id}", tags=["Records"])
@app.delete("/api/history/{record_id}", tags=["History"])
def delete_record_endpoint(record_id: Union[int, str]):
    """
    Safely deletes a report record from SQLite and unlinks its associated
    generated files (mask, overlay, attention map, and PDF report).
    """
    deleted = delete_record_and_files(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Analysis record '{record_id}' not found.")
    return {
        "success": True,
        "message": f"Report '{record_id}' and associated artifacts deleted successfully.",
        "record_id": deleted["id"],
        "reference_id": deleted.get("reference_id"),
        "deleted_files": deleted.get("deleted_files", [])
    }


@app.get("/api/report/{record_id}", tags=["Reports"])
def download_pdf_report(record_id: Union[int, str]):
    """Downloads or serves the pre-compiled PDF report for an analysis record."""
    record = get_record_by_ref_or_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    pdf_rel = record.get("report_path")
    needs_generation = True
    if pdf_rel:
        pdf_path = BASE_DIR / pdf_rel
        if pdf_path.exists():
            # Check if PDF template version is up-to-date (v2.4)
            try:
                with open(pdf_path, "rb") as f:
                    header_bytes = f.read(3000)
                    if b"Template v2.4" in header_bytes:
                        needs_generation = False
            except Exception:
                needs_generation = True

            if not needs_generation:
                return FileResponse(
                    str(pdf_path),
                    media_type="application/pdf",
                    filename=f"Report_{record.get('reference_id', 'Skin_Analysis')}.pdf"
                )

    # Re-generate PDF on-the-fly with updated v2.4 template
    try:
        ref_id = record.get("reference_id", f"REC{record['id']}")
        pdf_filename = f"report_{ref_id}.pdf"
        pdf_path = report_gen.generate_report(record, filename=pdf_filename)
        update_record_report_path(record["id"], f"reports/{pdf_filename}")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"Report_{ref_id}.pdf"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF report: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
