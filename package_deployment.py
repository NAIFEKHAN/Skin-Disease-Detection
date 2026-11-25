"""
Automated Deployment Packaging Script: CP-1 Skin Disease Detection.
Creates a clean, minimal production archive 'ec2_deployment_package.zip' for AWS EC2.
Includes:
  - Core app (main.py, database.py, requirements-prod.txt, README.md)
  - Models: genuine trained U-Net & 7-class ViT checkpoints and code
  - Services: all inference, segmentation, cost estimation, XAI, reporting
  - Frontend: modern forest green & white UI, mobile drawer, camera, animated charts
  - Data: disease_info.json (16 categories), treatment_costs.json
  - Deploy: systemd service, Nginx conf, DuckDNS script, server setup script
  - Empty persistent directories: uploads, reports
Excludes:
  - Experimental common skin model checkpoints (saves >550MB)
  - Smoke test weights and backups
  - Training datasets (HAM10000, SCIN, segmentation masks)
  - Colab zip archives (*.zip)
  - Local virtual environments, caches, and test runs
"""

import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ZIP = BASE_DIR / "ec2_deployment_package.zip"

ALLOWED_EXTENSIONS = {
    ".py", ".json", ".html", ".css", ".js", ".png", ".jpg", ".jpeg",
    ".svg", ".webp", ".service", ".conf", ".sh", ".txt", ".md", ".gitkeep", ".pth"
}

ALLOWED_CHECKPOINTS = {
    "unet_model.pth",
    "vit_model.pth"
}

EXCLUDED_DIRS = {
    "venv", ".venv", "__pycache__", ".pytest_cache", ".git", ".github",
    "training", "scin_metadata", "manifests", "classification",
    "common_skin", "segmentation", "smoke_test"
}

EXCLUDED_FILES = {
    "HAM10000_metadata.csv",
    "Mini_Project_Skin_Disease_Detection_1_ViT.pptx",
    "package_deployment.py"
}

def should_include(rel_path: Path) -> bool:
    parts = rel_path.parts
    filename = rel_path.name

    # Exclude zip, tmp, and log files
    if filename.endswith(".zip") or filename.endswith(".tmp") or filename.endswith(".log"):
        return False

    # Exclude designated directories
    for part in parts:
        if part in EXCLUDED_DIRS or part.startswith("."):
            return False

    # Check filename exclusions
    if filename in EXCLUDED_FILES:
        return False
    if filename.startswith("test_") and filename.endswith(".py"):
        return False

    # Extension whitelist
    suffix = rel_path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS and filename != ".gitkeep":
        return False

    # Check model checkpoints specifically
    if "models" in parts and filename.endswith(".pth"):
        if filename not in ALLOWED_CHECKPOINTS:
            return False

    # Check uploads and reports: only include .gitkeep
    if parts[0] in ("uploads", "reports"):
        if filename != ".gitkeep":
            return False

    # In data: only include disease_info.json, treatment_costs.json, .gitkeep
    if parts[0] == "data":
        if filename not in ("disease_info.json", "treatment_costs.json", ".gitkeep"):
            return False

    return True

def create_deployment_package():
    print("=" * 60)
    print(" Building Production EC2 Deployment Package")
    print("=" * 60)

    # Ensure placeholder directories have .gitkeep
    for folder in [
        BASE_DIR / "uploads",
        BASE_DIR / "uploads" / "inputs",
        BASE_DIR / "uploads" / "outputs",
        BASE_DIR / "reports",
        BASE_DIR / "data"
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        gitkeep = folder / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    # Create temporary zip in system temp directory to prevent recursive self-inclusion
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    total_files = 0
    total_uncompressed_bytes = 0

    try:
        print("Compressing project files into archive...")
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, dirs, files in os.walk(BASE_DIR):
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

                for file in files:
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(BASE_DIR)

                    if should_include(rel_path):
                        arcname = str(rel_path).replace("\\", "/")
                        zf.write(full_path, arcname)
                        file_size = full_path.stat().st_size
                        total_uncompressed_bytes += file_size
                        total_files += 1

        # Replace destination zip
        if OUTPUT_ZIP.exists():
            OUTPUT_ZIP.unlink()
        
        # Move from tmp to target
        shutil.move(str(tmp_path), str(OUTPUT_ZIP))

        zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
        raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

        print("=" * 60)
        print("✔ Deployment package created successfully!")
        print(f"Archive: {OUTPUT_ZIP.name}")
        print(f"Files included: {total_files}")
        print(f"Uncompressed size: {raw_size_mb:.2f} MB")
        print(f"Compressed size:   {zip_size_mb:.2f} MB")
        print("=" * 60)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

if __name__ == "__main__":
    create_deployment_package()
