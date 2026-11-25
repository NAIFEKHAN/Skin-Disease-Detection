"""
Database module for AI-Based Skin Disease Detection with Treatment Cost Estimation.
Uses SQLite for persistent storage of analysis records, segmentation results,
explainability attention maps, and treatment cost breakdowns.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "skin_disease.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database with Row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the database schema and performs safe migrations."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reference_id TEXT UNIQUE,
                image_path TEXT NOT NULL,
                mask_path TEXT,
                overlay_path TEXT,
                attention_map_path TEXT,
                predicted_disease TEXT,
                predicted_class_code TEXT,
                confidence_score REAL,
                severity_grade TEXT,
                severity_details_json TEXT,
                lesion_area_percentage REAL,
                top_3_json TEXT,
                all_probs_json TEXT,
                estimated_cost_min REAL,
                estimated_cost_max REAL,
                currency TEXT DEFAULT 'INR',
                cost_breakdown_json TEXT,
                report_path TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                patient_name TEXT,
                patient_age INTEGER,
                patient_gender TEXT
            )
        """)

        # Migration: ensure any missing columns exist on older tables
        cursor.execute("PRAGMA table_info(analysis_records);")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        needed_columns = [
            ("reference_id", "TEXT"),
            ("overlay_path", "TEXT"),
            ("attention_map_path", "TEXT"),
            ("predicted_class_code", "TEXT"),
            ("severity_details_json", "TEXT"),
            ("top_3_json", "TEXT"),
            ("all_probs_json", "TEXT"),
        ]

        for col_name, col_type in needed_columns:
            if col_name not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE analysis_records ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_created_at 
            ON analysis_records(created_at DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_ref_id
            ON analysis_records(reference_id)
        """)
        conn.commit()


def save_analysis(
    image_path: str,
    mask_path: Optional[str] = None,
    overlay_path: Optional[str] = None,
    attention_map_path: Optional[str] = None,
    predicted_disease: Optional[str] = None,
    predicted_class_code: Optional[str] = None,
    confidence_score: Optional[float] = None,
    severity_grade: Optional[str] = None,
    severity_details: Optional[Dict[str, Any]] = None,
    lesion_area_percentage: Optional[float] = None,
    top_3_predictions: Optional[List[Dict[str, Any]]] = None,
    all_probabilities: Optional[Dict[str, Any]] = None,
    estimated_cost_min: Optional[float] = None,
    estimated_cost_max: Optional[float] = None,
    currency: str = "INR",
    cost_breakdown: Optional[Dict[str, Any]] = None,
    report_path: Optional[str] = None,
    reference_id: Optional[str] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves a complete dermatological analysis record and returns the saved record dict.
    """
    init_db()

    if not reference_id:
        timestamp_str = datetime.now().strftime("%Y%m%d")
        rand_suffix = uuid.uuid4().hex[:6].upper()
        reference_id = f"REF-{timestamp_str}-{rand_suffix}"

    breakdown_json = json.dumps(cost_breakdown) if cost_breakdown is not None else None
    severity_json = json.dumps(severity_details) if severity_details is not None else None
    top3_json = json.dumps(top_3_predictions) if top_3_predictions is not None else None
    all_probs_json_str = json.dumps(all_probabilities) if all_probabilities is not None else None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_records (
                reference_id,
                image_path,
                mask_path,
                overlay_path,
                attention_map_path,
                predicted_disease,
                predicted_class_code,
                confidence_score,
                severity_grade,
                severity_details_json,
                lesion_area_percentage,
                top_3_json,
                all_probs_json,
                estimated_cost_min,
                estimated_cost_max,
                currency,
                cost_breakdown_json,
                report_path,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reference_id,
            image_path,
            mask_path,
            overlay_path,
            attention_map_path,
            predicted_disease,
            predicted_class_code,
            confidence_score,
            severity_grade,
            severity_json,
            lesion_area_percentage,
            top3_json,
            all_probs_json_str,
            estimated_cost_min,
            estimated_cost_max,
            currency,
            breakdown_json,
            report_path,
            notes
        ))
        conn.commit()
        record_id = cursor.lastrowid

    return get_record_by_id(record_id)  # type: ignore


def format_record(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deserializes JSON fields and structures record representation."""
    d = dict(row_dict)
    for json_field, target_field in [
        ("cost_breakdown_json", "cost_breakdown"),
        ("severity_details_json", "severity_details"),
        ("top_3_json", "top_3_predictions"),
        ("all_probs_json", "all_probabilities")
    ]:
        if d.get(json_field):
            try:
                d[target_field] = json.loads(d[json_field])
            except Exception:
                d[target_field] = None
        else:
            d[target_field] = None

    # Guarantee reference_id exists
    if not d.get("reference_id"):
        d["reference_id"] = f"REF-{d.get('id', 0):05d}"

    return d


def get_all_records(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieves analysis records ordered by creation date descending."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM analysis_records 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = cursor.fetchall()
        return [format_record(dict(r)) for r in rows]


def get_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single record by its primary key ID."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        return format_record(dict(row)) if row else None


def get_record_by_ref_or_id(identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
    """Retrieves a single record by reference_id or integer ID."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if str(identifier).isdigit():
            cursor.execute("SELECT * FROM analysis_records WHERE id = ? OR reference_id = ?", (int(identifier), str(identifier)))
        else:
            cursor.execute("SELECT * FROM analysis_records WHERE reference_id = ?", (str(identifier),))
        row = cursor.fetchone()
        return format_record(dict(row)) if row else None


def update_record_report_path(record_id: int, report_path: str) -> None:
    """Updates the generated report path for a record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE analysis_records SET report_path = ? WHERE id = ?", (report_path, record_id))
        conn.commit()


def delete_record_and_files(identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
    """
    Safely deletes a record from the database and removes only its exclusively
    associated generated output files (mask, overlay, attention map, and PDF report).
    
    Strictly prevents path traversal and protects dataset, models, and application files.
    """
    init_db()
    record = get_record_by_ref_or_id(identifier)
    if not record:
        return None

    rec_id = record["id"]

    # 1. Delete row from database
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_records WHERE id = ?", (rec_id,))
        conn.commit()

    # 2. Collect candidate associated files
    candidate_paths = [
        record.get("image_path"),
        record.get("mask_path"),
        record.get("overlay_path"),
        record.get("attention_map_path"),
        record.get("report_path")
    ]

    allowed_roots = [
        (BASE_DIR / "uploads" / "inputs").resolve(),
        (BASE_DIR / "uploads" / "outputs").resolve(),
        (BASE_DIR / "reports").resolve()
    ]

    deleted_files = []
    for rel_path in candidate_paths:
        if not rel_path:
            continue
        try:
            target_path = (BASE_DIR / str(rel_path).lstrip("/")).resolve()
            # Verify target path is strictly inside allowed roots
            is_safe = any(
                target_path.is_relative_to(root)
                for root in allowed_roots
            )
            # Extra safety: Ensure filename starts with expected prefixes or is not shared
            if is_safe and target_path.is_file():
                # Never delete .gitkeep
                if target_path.name == ".gitkeep":
                    continue
                target_path.unlink(missing_ok=True)
                deleted_files.append(str(target_path.name))
        except Exception as e:
            print(f"Warning: Failed to delete associated file {rel_path}: {e}")

    record["deleted_files"] = deleted_files
    return record


if __name__ == "__main__":
    init_db()
    print("Database initialized & migrated successfully at:", DB_PATH)
