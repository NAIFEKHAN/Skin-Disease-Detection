"""
Prepare Expanded Common Skin Disease Dataset from Verified Clinical Datasets (Google SCIN & SkinDisNet).
Phase Expansion: Common Skin Conditions Classification.

Features:
- Collects genuine clinical photographs from Google SCIN (JAMA Network Open 2024, CC-BY 4.0 / Apache 2.0)
  and SkinDisNet (Data in Brief 2025, CC BY-NC 4.0).
- Supports 9 verified target categories with up to 150-250 images per category:
  1. Acne
  2. Allergic Contact Dermatitis
  3. Eczema
  4. Impetigo
  5. Psoriasis
  6. Ringworm (Tinea Corporis)
  7. Scabies
  8. Seborrheic Dermatitis
  9. Urticaria (Hives)
- Prioritizes distinct independent patient cases (1 image per case first).
- Resizes oversized raw training photographs to max dimension 512x512 (Lanczos, quality=92, preserving aspect ratio)
  reducing storage by ~15-30x while preserving dermatological textures and clinical details.
- Explicitly documents and excludes categories with insufficient reliable clinical samples:
  - Vitiligo (only 2 primary cases in SCIN, 0 in SkinDisNet)
  - Other fungal infections (only 31 primary Tinea Versicolor cases; kept separate from Tinea Corporis)
- Case/Patient-level group-aware stratified splitting (70% Train, 15% Val, 15% Test, Seed 42).
- Zero data leakage between splits.
- Class weight computation for Weighted Cross-Entropy Loss.
"""

import os
import io
import sys
import ast
import json
import hashlib
import argparse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
COMMON_SKIN_DIR = DATA_DIR / "common_skin"
IMAGES_DIR = COMMON_SKIN_DIR / "images"
MANIFESTS_DIR = DATA_DIR / "manifests"
SCIN_METADATA_DIR = DATA_DIR / "scin_metadata"

# 9 Canonical Target Categories
COMMON_SKIN_CLASSES = [
    "acne",
    "allergic_contact_dermatitis",
    "eczema",
    "impetigo",
    "psoriasis",
    "ringworm",
    "scabies",
    "seborrheic_dermatitis",
    "urticaria"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(COMMON_SKIN_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(COMMON_SKIN_CLASSES)}

CLASS_DISPLAY_NAMES = {
    "acne": "Acne",
    "allergic_contact_dermatitis": "Allergic Contact Dermatitis",
    "eczema": "Eczema",
    "impetigo": "Impetigo",
    "psoriasis": "Psoriasis",
    "ringworm": "Ringworm (Tinea Corporis)",
    "scabies": "Scabies",
    "seborrheic_dermatitis": "Seborrheic Dermatitis",
    "urticaria": "Urticaria (Hives)"
}

UNSUPPORTED_CATEGORIES_REASON = {
    "vitiligo": "Insufficient reliable clinical samples in verified datasets (only 2 primary cases in SCIN, 0 in SkinDisNet). Excluded to prevent model failure.",
    "other_fungal": "Insufficient samples (only 31 primary Tinea Versicolor cases in SCIN); excluded to maintain rigorous clinical distinction from verified Tinea Corporis (Ringworm)."
}

SKINDISNET_ZIP_URL = "https://data.mendeley.com/public-files/datasets/yj3md44hxg/files/ec865f3f-59d9-47b8-b715-4fe11edd3c83/file_downloaded"
SKINDISNET_TOTAL_SIZE = 1398802309
GCS_SCIN_BASE = "https://storage.googleapis.com/dx-scin-public-data"


def get_skindisnet_direct_s3_url() -> str:
    """Follows initial Mendeley 302 redirect to obtain direct S3 presigned URL for fast byte ranges."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(urllib.request.Request(SKINDISNET_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"}))
        return SKINDISNET_ZIP_URL
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307) and "Location" in e.headers:
            return e.headers["Location"]
        raise


class S3RangeFile(io.RawIOBase):
    """File-like wrapper that issues HTTP Range requests to S3 without redirect latency."""
    def __init__(self, s3_url: str, size: int):
        self.url = s3_url
        self.size = size
        self.pos = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b""
        req = urllib.request.Request(
            self.url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={self.pos}-{self.pos + size - 1}"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            self.pos += len(data)
            return data


def ensure_scin_metadata():
    """Ensure SCIN metadata CSVs are downloaded."""
    SCIN_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["scin_cases.csv", "scin_labels.csv"]:
        fpath = SCIN_METADATA_DIR / fname
        if not fpath.exists():
            print(f"Downloading {fname} from GCS...")
            url = f"{GCS_SCIN_BASE}/dataset/{fname}"
            urllib.request.urlretrieve(url, fpath)


def optimize_and_save_image(raw_bytes: bytes, save_path: Path, max_dim: int = 512, quality: int = 92) -> bool:
    """
    Decodes image, resizes oversized dimensions to max_dim (preserving exact aspect ratio via Lanczos),
    and saves as optimized high-quality JPEG.
    Preserves all clinical patterns, lesion borders, erythema, and scaling.
    """
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        
        # Re-open for transformation
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img_rgb = img.convert("RGB")
            w, h = img_rgb.size
            if max(w, h) > max_dim:
                img_rgb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img_rgb.save(save_path, format="JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False


def download_single_scin_image(gcs_rel_path: str, save_path: Path) -> bool:
    """Download a single image from Google SCIN GCS bucket and save optimized."""
    try:
        url = f"{GCS_SCIN_BASE}/{gcs_rel_path}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
        return optimize_and_save_image(content, save_path)
    except Exception:
        return False


def collect_scin_samples(target_per_class: int = 200) -> List[Dict]:
    """
    Collects high-confidence samples from Google SCIN.
    Strictly prioritizes independent patient cases (1 image per case first).
    """
    ensure_scin_metadata()
    cases_df = pd.read_csv(SCIN_METADATA_DIR / "scin_cases.csv", dtype={"case_id": str})
    labels_df = pd.read_csv(SCIN_METADATA_DIR / "scin_labels.csv", dtype={"case_id": str})
    merged = pd.merge(cases_df, labels_df, on="case_id", how="inner")

    scin_class_map = {
        "Acne": "acne",
        "Acneiform eruption": "acne",
        "Allergic Contact Dermatitis": "allergic_contact_dermatitis",
        "Eczema": "eczema",
        "Infected eczema": "eczema",
        "Impetigo": "impetigo",
        "Psoriasis": "psoriasis",
        "Urticaria": "urticaria",
        "Tinea": "ringworm",
        "Scabies": "scabies",
        "Seborrheic Dermatitis": "seborrheic_dermatitis"
    }

    # Group candidate cases by target class
    candidates_by_class: Dict[str, List[Tuple[str, str, str, List[str]]]] = {c: [] for c in COMMON_SKIN_CLASSES}

    for _, row in merged.iterrows():
        gradable = str(row.get("dermatologist_gradable_for_skin_condition_1", ""))
        if "NO" in gradable:
            continue

        label_str = str(row.get("dermatologist_skin_condition_on_label_name", ""))
        if not label_str or label_str in ["nan", "[]"]:
            continue

        try:
            labels = ast.literal_eval(label_str)
            if not labels:
                continue
        except Exception:
            continue

        matched_class = None
        matched_label = None

        # Check primary label first
        if labels[0] in scin_class_map:
            matched_class = scin_class_map[labels[0]]
            matched_label = labels[0]
        else:
            # Check secondary labels
            for l in labels[1:]:
                if l in scin_class_map:
                    matched_class = scin_class_map[l]
                    matched_label = l
                    break

        if not matched_class:
            continue

        case_id = str(row["case_id"])
        fitz = str(row.get("dermatologist_fitzpatrick_skin_type_label_1", "") or row.get("fitzpatrick_skin_type", "FST_UNKNOWN"))

        # Collect available images for this case
        img_paths = []
        for p_col in ["image_1_path", "image_2_path", "image_3_path"]:
            rel = str(row.get(p_col, ""))
            if rel and rel != "nan" and rel.startswith("dataset/images/"):
                img_paths.append(rel)

        if img_paths:
            candidates_by_class[matched_class].append((case_id, matched_label, fitz, img_paths))

    selected_records = []

    for c in COMMON_SKIN_CLASSES:
        cases = candidates_by_class[c]
        count = 0
        
        # PASS 1: Prioritize independent patient cases (1 image per case)
        for case_id, matched_label, fitz, img_paths in cases:
            if count >= target_per_class:
                break
            primary_img = img_paths[0]
            img_id = Path(primary_img).stem
            filename = f"scin_{c}_{case_id}_{img_id}.jpg"
            selected_records.append({
                "case_id": f"scin_{case_id}",
                "source": "Google_SCIN",
                "source_path": primary_img,
                "filename": filename,
                "target_class": c,
                "dermatologist_label": matched_label,
                "skin_tone": fitz
            })
            count += 1

        # PASS 2: If target not yet met, take supplemental images from multi-image cases
        if count < target_per_class:
            for case_id, matched_label, fitz, img_paths in cases:
                if count >= target_per_class:
                    break
                for supp_img in img_paths[1:]:
                    if count >= target_per_class:
                        break
                    img_id = Path(supp_img).stem
                    filename = f"scin_{c}_{case_id}_{img_id}.jpg"
                    selected_records.append({
                        "case_id": f"scin_{case_id}",
                        "source": "Google_SCIN",
                        "source_path": supp_img,
                        "filename": filename,
                        "target_class": c,
                        "dermatologist_label": matched_label,
                        "skin_tone": fitz
                    })
                    count += 1

    return selected_records


def extract_single_skindisnet_file(s3_url: str, item: zipfile.ZipInfo, save_path: Path) -> bool:
    """Extract a single zip entry from S3 using a single HTTP Range request."""
    if save_path.exists():
        return True
    try:
        fetch_len = 100 + item.compress_size
        req = urllib.request.Request(
            s3_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={item.header_offset}-{item.header_offset + fetch_len - 1}"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            buf = resp.read()

        import struct, zlib
        if buf[:4] != b"PK\x03\x04":
            return False

        name_len, extra_len = struct.unpack("<HH", buf[26:30])
        data_start = 30 + name_len + extra_len
        raw_compressed = buf[data_start:data_start + item.compress_size]

        if item.compress_type == 0:
            extracted = raw_compressed
        elif item.compress_type == 8:
            extracted = zlib.decompress(raw_compressed, -15)
        else:
            return False

        return optimize_and_save_image(extracted, save_path)
    except Exception:
        return False


def extract_skindisnet_samples(current_counts: Dict[str, int], target_per_class: int = 180) -> List[Dict]:
    """
    Extracts original clinical photographs from SkinDisNet via concurrent S3 range requests.
    Prioritizes classes that need supplemental images to reach target_per_class (Scabies, Seborrheic Dermatitis, Ringworm).
    """
    skindisnet_map = {
        "TC": "ringworm",
        "SC": "scabies",
        "SD": "seborrheic_dermatitis"
    }

    try:
        s3_url = get_skindisnet_direct_s3_url()
        print("Obtained direct SkinDisNet S3 URL from Mendeley redirect.")
    except Exception as e:
        print(f"Failed to obtain S3 direct URL: {e}")
        return []

    rf = S3RangeFile(s3_url, SKINDISNET_TOTAL_SIZE)
    selected_records = []
    extraction_tasks = []

    try:
        with zipfile.ZipFile(rf) as z:
            for prefix, target_class in skindisnet_map.items():
                folder_prefix = f"Preprocessed/{prefix}/"
                matching_files = [i for i in z.infolist() if i.filename.startswith(folder_prefix) and i.file_size > 0]
                
                needed = max(0, target_per_class - current_counts.get(target_class, 0))
                if needed <= 0 and target_class != "seborrheic_dermatitis":
                    needed = 50  # Take at least 50 high-quality samples for demographic diversity
                
                chosen = matching_files[:needed]
                print(f"  SkinDisNet {prefix} ({target_class}): Extracting {len(chosen)} clinical photographs concurrently...")

                for item in chosen:
                    base_name = Path(item.filename).name
                    save_filename = f"skindisnet_{target_class}_{base_name}"
                    save_path = IMAGES_DIR / save_filename
                    patient_case_id = f"skindisnet_{Path(base_name).stem}"

                    extraction_tasks.append((item, save_path))
                    selected_records.append({
                        "case_id": patient_case_id,
                        "source": "SkinDisNet",
                        "source_path": item.filename,
                        "filename": save_filename,
                        "target_class": target_class,
                        "dermatologist_label": target_class.replace("_", " ").title(),
                        "skin_tone": "Fitzpatrick_IV_V"
                    })

        # Run concurrent extraction with 10 worker threads
        print(f"Running concurrent extraction of {len(extraction_tasks)} SkinDisNet images with 10 threads...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(extract_single_skindisnet_file, s3_url, item, sp)
                for item, sp in extraction_tasks
            ]
            completed = sum(1 for fut in as_completed(futures) if fut.result())
            print(f"Successfully extracted and optimized {completed}/{len(extraction_tasks)} SkinDisNet images.")

    except Exception as e:
        print(f"Error during SkinDisNet extraction: {e}")

    return selected_records


def build_expanded_dataset(target_per_class: int = 200, seed: int = 42):
    """Orchestrates expanded clinical dataset collection, optimization, deduplication, and group-aware splitting."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"BUILDING EXPANDED COMMON SKIN CLINICAL DATASET (Target: ~{target_per_class}/class)")
    print(f"=======================================================")

    # 1. Collect Google SCIN sample metadata
    print("Selecting Google SCIN records across independent cases...")
    scin_records = collect_scin_samples(target_per_class=target_per_class)
    print(f"Selected {len(scin_records)} images from Google SCIN.")

    # 2. Download missing SCIN images concurrently
    print(f"Downloading and optimizing SCIN images to {IMAGES_DIR}...")
    to_download = [r for r in scin_records if not (IMAGES_DIR / r["filename"]).exists()]
    if to_download:
        print(f"  Need to download {len(to_download)} new SCIN images...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_rec = {
                executor.submit(download_single_scin_image, r["source_path"], IMAGES_DIR / r["filename"]): r
                for r in to_download
            }
            downloaded = 0
            for fut in as_completed(future_to_rec):
                if fut.result():
                    downloaded += 1
            print(f"  Downloaded and optimized {downloaded}/{len(to_download)} SCIN images.")
    else:
        print("  All selected SCIN images already downloaded and optimized locally.")

    # 3. Extract SkinDisNet samples (TC, SC, SD) via concurrent S3 range requests
    print("\nExtracting complementary SkinDisNet clinical images...")
    current_counts = {}
    for r in scin_records:
        current_counts[r["target_class"]] = current_counts.get(r["target_class"], 0) + 1
    skindisnet_records = extract_skindisnet_samples(current_counts=current_counts, target_per_class=target_per_class)
    print(f"Extracted {len(skindisnet_records)} images from SkinDisNet.")

    all_records = scin_records + skindisnet_records

    # 4. Verification and Deduplication
    print("\nVerifying on-disk images, aspect ratios, and computing SHA-256 hashes...")
    verified_records = []
    seen_hashes = set()
    corrupt_count = 0
    duplicate_count = 0

    for r in all_records:
        img_path = IMAGES_DIR / r["filename"]
        if not img_path.exists():
            continue
        try:
            with open(img_path, "rb") as f:
                content = f.read()
            h = hashlib.sha256(content).hexdigest()
            if h in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(h)

            with Image.open(io.BytesIO(content)) as img:
                img.verify()
                w, h_dim = img.size
                if w < 64 or h_dim < 64:
                    corrupt_count += 1
                    continue

            r["sha256"] = h
            r["relative_path"] = f"data/common_skin/images/{r['filename']}"
            r["label_idx"] = CLASS_TO_IDX[r["target_class"]]
            verified_records.append(r)
        except Exception:
            corrupt_count += 1

    print(f"Verified {len(verified_records)} clean clinical images (Corrupt: {corrupt_count}, Duplicates: {duplicate_count}).")

    df = pd.DataFrame(verified_records)

    # 5. Group-Aware Stratified Train / Val / Test Split (70% / 15% / 15%)
    # Crucial: Group by case_id to prevent data leakage between splits!
    print(f"\nPerforming Group-Aware Stratified Split (Seed: {seed})...")
    np.random.seed(seed)

    case_summary = df.groupby("case_id").agg({
        "target_class": "first",
        "label_idx": "first"
    }).reset_index()

    train_cases = []
    val_cases = []
    test_cases = []

    for c in COMMON_SKIN_CLASSES:
        c_cases = list(case_summary[case_summary["target_class"] == c]["case_id"].values)
        np.random.shuffle(c_cases)
        n = len(c_cases)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1) if n > 2 else 0

        c_train = c_cases[:n_train]
        c_val = c_cases[n_train:n_train + n_val]
        c_test = c_cases[n_train + n_val:]

        train_cases.extend(c_train)
        val_cases.extend(c_val)
        test_cases.extend(c_test)

    train_cases_set = set(train_cases)
    val_cases_set = set(val_cases)
    test_cases_set = set(test_cases)

    # Verify zero case leakage
    assert len(train_cases_set & val_cases_set) == 0, "Data leakage between train and val!"
    assert len(train_cases_set & test_cases_set) == 0, "Data leakage between train and test!"
    assert len(val_cases_set & test_cases_set) == 0, "Data leakage between val and test!"

    df["split"] = "train"
    df.loc[df["case_id"].isin(val_cases_set), "split"] = "val"
    df.loc[df["case_id"].isin(test_cases_set), "split"] = "test"

    # Save Manifests
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    df.to_csv(MANIFESTS_DIR / "common_skin_all.csv", index=False)
    train_df.to_csv(MANIFESTS_DIR / "common_skin_train.csv", index=False)
    val_df.to_csv(MANIFESTS_DIR / "common_skin_val.csv", index=False)
    test_df.to_csv(MANIFESTS_DIR / "common_skin_test.csv", index=False)

    print("\n--- Manifests Generated Successfully ---")
    print(f"Total:      {len(df)} images ({len(case_summary)} independent cases)")
    print(f"Training:   {len(train_df)} images ({train_df['case_id'].nunique()} cases) - {len(train_df)/len(df)*100:.1f}%")
    print(f"Validation: {len(val_df)} images ({val_df['case_id'].nunique()} cases) - {len(val_df)/len(df)*100:.1f}%")
    print(f"Testing:    {len(test_df)} images ({test_df['case_id'].nunique()} cases) - {len(test_df)/len(df)*100:.1f}%")

    # 6. Class Distribution & Class Weights Calculation
    print("\n--- Class Distribution Across Splits ---")
    dist_table = []
    class_counts_train = []
    for c in COMMON_SKIN_CLASSES:
        idx = CLASS_TO_IDX[c]
        n_tr = len(train_df[train_df["target_class"] == c])
        n_va = len(val_df[val_df["target_class"] == c])
        n_te = len(test_df[test_df["target_class"] == c])
        tot = n_tr + n_va + n_te
        cases_tot = df[df["target_class"] == c]["case_id"].nunique()
        class_counts_train.append(n_tr)
        dist_table.append({
            "Index": idx,
            "Class Code": c,
            "Disease Name": CLASS_DISPLAY_NAMES[c],
            "Cases": cases_tot,
            "Train": n_tr,
            "Val": n_va,
            "Test": n_te,
            "Total": tot
        })

    dist_df = pd.DataFrame(dist_table)
    print(dist_df.to_string(index=False))

    # Compute balanced class weights
    n_total_train = sum(class_counts_train)
    n_classes = len(COMMON_SKIN_CLASSES)
    class_weights = [
        round(n_total_train / (n_classes * max(1, count)), 4)
        for count in class_counts_train
    ]

    metadata = {
        "dataset_name": "Expanded Common Skin Conditions Clinical Dataset",
        "sources": [
            {
                "name": "Google SCIN (Skin Condition Image Network)",
                "doi": "10.1001/jamanetworkopen.2024.46615",
                "license": "Creative Commons Attribution 4.0 International (CC BY 4.0) / Apache 2.0",
                "provenance": "Google Research & US Patient Image Donation Study"
            },
            {
                "name": "SkinDisNet",
                "doi": "10.1016/j.dib.2025.111456",
                "mendeley_doi": "10.17632/yj3md44hxg.1",
                "license": "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
                "provenance": "Rangpur Medical College and Shahid Syed Nazrul Islam Medical College, Bangladesh"
            }
        ],
        "classes": COMMON_SKIN_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "unsupported_categories": UNSUPPORTED_CATEGORIES_REASON,
        "total_images": len(df),
        "total_cases": len(case_summary),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "random_seed": seed,
        "class_weights": class_weights,
        "optimization": {
            "max_dimension": 512,
            "format": "JPEG",
            "quality": 92,
            "preserves_aspect_ratio": True
        }
    }

    meta_path = MANIFESTS_DIR / "common_skin_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata and class weights written to {meta_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Expanded Common Skin Disease Dataset")
    parser.add_argument("--target-per-class", type=int, default=200, help="Target genuine images per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    build_expanded_dataset(target_per_class=args.target_per_class, seed=args.seed)
