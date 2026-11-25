# AI-Based Skin Disease Detection with Treatment Cost Estimation
## Download trained models and complete project

The source is on `main`. GitHub Release [v1.0.0](https://github.com/NAIFEKHAN/Skin-Disease-Detection/releases/tag/v1.0.0) contains the two active trained model checkpoints and `Skin-Disease-Detection-v1.0.0-complete.zip`. To run inference from a source checkout, download `unet_model.pth` and `vit_model.pth` into `models/`. The common skin classifiers are experimental and are currently disabled in the deployed application.

The release archive includes the source and both active checkpoints. It excludes credentials, patient uploads and reports, databases, raw datasets, local environments, and caches.

An integrated deep learning and image processing decision-support platform designed to classify dermatological lesions, evaluate condition severity, and predict treatment cost breakdowns (INR) to promote healthcare transparency and early clinical screening.

---

## Ã¢Å¡Â Ã¯Â¸Â Medical Disclaimer
> **IMPORTANT:** This system is an academic and educational research project. It does **NOT** provide a formal medical diagnosis and is **NOT** a replacement for a certified dermatologist or qualified healthcare professional. It is intended strictly as a preliminary screening decision-support aid. Always seek professional clinical diagnosis for skin concerns.

---

## Ã°Å¸â€ºÂ Ã¯Â¸Â Technology Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Deep Learning & Computer Vision:** PyTorch, torchvision, timm (Vision Transformers), OpenCV
- **Data & Machine Learning:** NumPy, Pandas, Scikit-learn
- **Database:** SQLite
- **Report Generation:** ReportLab (PDF)
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (Forest green and white theme)
- **Architecture:** Modular Monolithic Architecture

---

## Ã°Å¸â€œÂ Project Directory Structure

```text
CP-1/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ data/                                 # Datasets, manifests, and persistent database
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ classification/
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ images/                       # 1,757 genuine dermoscopic images (.gitignore)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ segmentation/
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ images/                       # 1,000 genuine paired dermoscopic images (.gitignore)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ masks/                        # 1,000 expert-annotated binary masks (.gitignore)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ manifests/                        # Reproducible split manifests (tracked)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ classification_train.csv      # 1,233 samples (70.2%)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ classification_val.csv        # 267 samples (15.2%)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ classification_test.csv       # 257 samples (14.6%)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ segmentation_train.csv        # 700 pairs (70.0%)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ segmentation_val.csv          # 150 pairs (15.0%)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ segmentation_test.csv         # 150 pairs (15.0%)
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ dataset_summary.json          # Pipeline statistics & leakage verification
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ skin_disease.db                   # SQLite database (migrated with attention_map_path)
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ .gitkeep
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ frontend/                             # Web interface
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ css/
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ style.css                     # Forest green and white theme styles
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ js/
Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ app.js                        # Client interactions, health check & history loader
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ history.html                      # Analysis history view (standardized /api/history)
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ index.html                        # Main dashboard & medical disclaimer
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ models/                               # ViT and U-Net model architecture definitions
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ reports/                              # Generated PDF diagnostic reports
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ .gitkeep
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ services/                             # Business logic & image preprocessing
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ attention.py                      # ViT Attention Rollout Explainable AI (XAI) service
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ cost_estimation.py               # Benchmark Indian treatment cost estimation engine (INR)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ preprocessing.py                  # DullRazor, ViT & U-Net preprocessors, augmentation
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ prediction.py                     # ViT-B/16 inference, probability distribution & top-3
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ report.py                         # ReportLab comprehensive PDF medical report compiler
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ segmentation.py                   # U-Net lesion segmentation inference service (threshold 0.70)
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ severity.py                       # Rule-based clinical prototype severity indicator
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ data/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ treatment_costs.json              # Indicative treatment cost bands (INR) across 5 care tiers
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ ...
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ training/                             # Training datasets, pipelines & evaluators
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ dataset.py                        # PyTorch Dataset classes, DataLoaders & class weights
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ prepare_datasets.py               # Dataset sampler, split generator & S3 downloader
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ uploads/                              # Stored uploaded skin lesion images
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ .gitkeep
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ .gitignore                            # Excludes venv, large images, DB, weights & caches
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ database.py                           # SQLite schema, migrations & CRUD operations
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ main.py                               # FastAPI routes (/health, /api/history, static)
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ README.md                             # Comprehensive project documentation
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ requirements.txt                      # Pinned Python dependencies
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ test_phase2.py                        # Automated 9-test verification suite
```

---

## Ã°Å¸â€œÅ  Dataset Specifications (Phase 2)

### 1. Classification Dataset Ã¢â‚¬â€ HAM10000
- **Source:** Harvard Dataverse (`doi:10.7910/DVN/DBW86T`) / Official ISIC Archive AWS S3 Mirror.
- **License:** Creative Commons Attribution-NonCommercial (CC BY-NC 4.0) / CC-0.
- **Total Genuine Images:** 1,757 (compact, balanced subset sampled with random seed 42).
- **Disease-wise Class Distribution:**

| Class Code | Disease Category Name | Available | Sampled Images |
| :--- | :--- | :--- | :--- |
| `NV` | Melanocytic Nevi | 6,705 | 300 |
| `MEL` | Melanoma | 1,113 | 300 |
| `BKL` | Benign Keratosis-like Lesions | 1,099 | 300 |
| `BCC` | Basal Cell Carcinoma | 514 | 300 |
| `AKIEC` | Actinic Keratoses / Intraepithelial Carcinoma | 327 | 300 |
| `VASC` | Vascular Lesions | 142 | 142 |
| `DF` | Dermatofibroma | 115 | 115 |
| **Total** | | **10,015** | **1,757** |

- **Stratified Lesion-Group-Aware Splitting (Seed 42):**
  - **Train:** 1,233 images (70.2%)
  - **Validation:** 267 images (15.2%)
  - **Test:** 257 images (14.6%)
  - **Zero Lesion Leakage:** Images belonging to the same lesion (`lesion_id`) are strictly confined to a single split.

---

### 2. Segmentation Dataset Ã¢â‚¬â€ ISIC
- **Source:** ISIC 2018 Task 1 / Tschandl et al. expert lesion segmentations.
- **License:** CC BY-NC 4.0.
- **Total Genuine Image-Mask Pairs:** 1,000 genuine pairs.
- **Lesion Area Diversity:**
  - Mean coverage: 22.8% of image area (range: 1.1% to 68.4%).
  - Covers small (<10%), medium (10Ã¢â‚¬â€œ30%), and large (>30%) lesions across all skin types.
- **Split Distribution (Strictly Split-Aligned):**
  - **Train:** 700 pairs (70.0%)
  - **Validation:** 150 pairs (15.0%)
  - **Test:** 150 pairs (15.0%)

---

### 3. Cross-Task Evaluation Isolation
To prevent evaluation contamination across the integrated pipeline:
- **Zero Test Contamination:** No classification test image appears in segmentation training or validation. No segmentation test image appears in classification training or validation.
- All shared images (1,000 pairs) maintain 100% split alignment across tasks.

---

## Ã°Å¸â€Â¬ Image Preprocessing & Augmentation (`services/preprocessing.py`)

1. **Validation & Decoding:**
   - Supports file paths, byte streams, and raw NumPy arrays.
   - Enforces 3-channel RGB uint8 conversion and rejects corrupted images.
2. **Hair Artefact Removal (DullRazor):**
   - Applies Morphological BlackHat filtering (17Ãƒâ€”17 kernel) on grayscale lesion views.
   - Binary thresholding isolates hair strands.
   - Fast marching Telea inpainting restores occluded lesion regions.
3. **Classification Preprocessing (ViT-B/16):**
   - Image resizing to `224Ãƒâ€”224` (`cv2.INTER_AREA`).
   - Standard ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
   - Output shape: `(3, 224, 224)` Float32 PyTorch tensor.
4. **Segmentation Preprocessing (U-Net):**
   - Image resizing to `256Ãƒâ€”256` (`cv2.INTER_AREA`) with ImageNet normalization: `(3, 256, 256)`.
   - Mask resizing to `256Ãƒâ€”256` using **nearest-neighbor interpolation** (`cv2.INTER_NEAREST`) to preserve sharp binary boundaries.
   - Mask normalization to binary `{0.0, 1.0}`: `(1, 256, 256)` Float32 PyTorch tensor.
5. **Data Augmentation:**
   - Applied **strictly to training sets** (validation and test sets are never augmented).
   - Coordinated geometric transforms applied identically to both image and mask: horizontal/vertical flips, minor rotations (Ã‚Â±15Ã‚Â°).
   - Photometric adjustments (brightness/contrast scaling 0.9Ã¢â‚¬â€œ1.1) applied only to images.

---

## Ã°Å¸â€œÂ¦ PyTorch Datasets & DataLoaders (`training/dataset.py`)

- **`SkinClassificationDataset`:** Reads manifests, loads images on-the-fly, returns `(image_tensor, label_int, image_id)`.
- **`SkinSegmentationDataset`:** Reads manifests, returns `(image_tensor, mask_tensor, image_id)`.
- **`compute_class_weights()`:** Computes balanced weights for `nn.CrossEntropyLoss(weight=weights)` to handle class imbalance:
  - `MEL`: 0.831 | `NV`: 0.839 | `BCC`: 0.839 | `AKIEC`: 0.835 | `BKL`: 0.839 | `DF`: 1.779 | `VASC`: 2.175
- **Google Colab Support:** Dataset paths are dynamically configurable via `data_root` parameter or `DERMA_DATA_ROOT` environment variable.

---

## Ã°Å¸Å¡â‚¬ Local Execution Instructions

### 1. Activate Virtual Environment
**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Verify or Regenerate Datasets & Manifests
```powershell
python training/prepare_datasets.py
```

### 3. Run Automated Verification Suites
```powershell
# Phase 2 verification (dataset integrity, shapes, and splits)
python test_phase2.py

# Phase 3 verification (U-Net architecture, loss, CPU inference & service)
python test_phase3.py

# Phase 4 verification (ViT architecture, transfer learning, inference service)
python test_phase4.py
```

### 4. Run Model Training Smoke Tests Locally
```powershell
# U-Net segmentation smoke test
python training/train_unet.py --smoke-test

# ViT classification smoke test
python training/train_vit.py --smoke-test --output-dir models/smoke_test
```

### 5. Google Colab GPU Training

#### U-Net Lesion Segmentation
1. Upload `training/train_unet.ipynb` and `colab_unet_package.zip` (25.68 MB) to [Google Colab](https://colab.research.google.com/).
2. Run on T4 GPU runtime.
3. Save resulting `unet_model.pth` (173 MB) to `models/`.

#### Vision Transformer (ViT-B/16) Classification
1. Upload `training/train_vit.ipynb` and `colab_vit_package.zip` (43.09 MB) to [Google Colab](https://colab.research.google.com/).
2. Set runtime to **GPU (T4)**.
3. Run all cells in `train_vit.ipynb` (approx. 8Ã¢â‚¬â€œ12 minutes on T4 GPU).
4. Download the trained `vit_model.pth` and `vit_metrics.json` back to your local `models/` and `training/results/` directories.

### 6. Start the Application Server
```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
- **Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **History View:** [http://127.0.0.1:8000/history](http://127.0.0.1:8000/history)
- **Standardized API History:** [http://127.0.0.1:8000/api/history](http://127.0.0.1:8000/api/history)
- **Health Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Ã°Å¸Â§Â  U-Net Model Architecture & Service (`models/unet.py`, `services/segmentation.py`)

- **Encoder:** Pretrained ResNet18 (Torchvision weights `DEFAULT`).
- **Decoder:** 4-stage progressive bilateral upsampling with double-convolution blocks.
- **Skip Connections:** Preserves spatial lesion context across scales (128Ãƒâ€”128, 64Ãƒâ€”64, 32Ãƒâ€”32, 16Ãƒâ€”16).
- **Parameters:** 14,404,257 total trainable parameters.
- **Loss:** $\mathcal{L}_{total} = 0.5 \cdot \mathcal{L}_{BCEWithLogits} + 0.5 \cdot \mathcal{L}_{Dice}$.
- **Inference Service:**
  - Auto-loads `models/unet_model.pth` with CPU compatibility.
  - Automatically handles missing checkpoints gracefully.
  - Computes relative lesion area percentage: $\frac{\text{lesion pixels}}{\text{total pixels}} \times 100\%$.
  - Generates transparent green overlays with contour boundary delineation.

### Actual Final GPU Training & Evaluation Results (Google Colab T4 GPU, 20 Epochs)

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Test Loss (BCE + Dice)** | **0.1913** | Evaluated on 150 held-out genuine ISIC test samples |
| **Test Dice Score (F1)** | **0.8838** (88.38%) | Lesion boundary overlap accuracy |
| **Test IoU (Jaccard Index)** | **0.7953** (79.53%) | Intersection-over-Union on held-out test data |
| **Test Pixel Precision** | **0.8984** (89.84%) | Low false-positive rate on healthy skin regions |
| **Test Pixel Recall** | **0.8715** (87.15%) | True-positive sensitivity identifying lesion margins |
| **Best Validation Dice** | **0.8891** (88.91%) | Achieved at Epoch 19/20 |
| **Optimal Threshold** | **0.70** | Dynamically optimized on validation set |
| **Total GPU Training Time** | **315.2 seconds** (~5.25 min) | Fast, efficient training on Colab T4 GPU |

Artifacts stored at:
- Model Weights: `models/unet_model.pth` (173 MB)
- Metrics Summary: `training/results/unet_metrics.json`
- Epoch History: `training/results/unet_training_history.csv`
- Curves: `training/results/plots/loss_curve.png`, `dice_curve.png`, `iou_curve.png`
- Visual Comparisons: `training/results/samples/test_sample_*.png`

---

## Ã°Å¸â€Â¬ Vision Transformer Model Architecture & Inference Service (`models/vit.py`, `services/prediction.py`)

- **Architecture:** Pretrained Vision Transformer `vit_base_patch16_224` (via `timm`), ImageNet-21k fine-tuned on ImageNet-1k.
- **Input Specifications:** 3Ãƒâ€”224Ãƒâ€”224 RGB tensors, normalized with standard ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Classification Head:** Linear projection from 768-dimensional token embedding to 7 disease classes:
  - `NV` (0) Ã¢â‚¬â€ Melanocytic Nevi
  - `MEL` (1) Ã¢â‚¬â€ Melanoma
  - `BKL` (2) Ã¢â‚¬â€ Benign Keratosis-like Lesions
  - `BCC` (3) Ã¢â‚¬â€ Basal Cell Carcinoma
  - `AKIEC` (4) Ã¢â‚¬â€ Actinic Keratoses and Intraepithelial Carcinoma
  - `VASC` (5) Ã¢â‚¬â€ Vascular Lesions
  - `DF` (6) Ã¢â‚¬â€ Dermatofibroma
- **Transfer Learning & Freezing Strategy:**
  - Patch embedding, class token, position embeddings, and blocks 0Ã¢â‚¬â€œ9 frozen (71,621,376 frozen parameters).
  - Top 2 transformer blocks (blocks 10 and 11), layernorm, and linear classification head remain trainable (14,182,663 trainable parameters).
  - Total parameters: 85,804,039.
- **Discriminative Learning Rates & Optimizer:**
  - AdamW optimizer with classification head learning rate $1\times 10^{-4}$ and backbone learning rate $1\times 10^{-5}$.
  - Linear warmup (2 epochs) followed by Cosine Annealing learning rate schedule ($\eta_{min} = 1\times 10^{-6}$).
  - Class-weighted Cross-Entropy loss balancing rare classes (DF: 2.175, VASC: 1.779) against frequent categories.
- **Inference Service (`services/prediction.py`):**
  - Preprocesses input dermoscopy images with optional DullRazor hair removal.
  - Safe model weight loading (`models/vit_model.pth` or fallback path) with CPU-compatible inference.
  - Outputs predicted disease class, confidence score, full 7-class probability breakdown, and top-3 ranked differential diagnoses.
  - Flags predictions under 50% confidence as uncertain.
  - Attaches strict medical scope disclaimer to prevent misuse as a standalone clinical diagnosis.

### Actual Final GPU Training & Evaluation Results (Google Colab T4 GPU, 20 Epochs)

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Test Loss (Weighted CE)** | **0.9510** | Evaluated on 257 held-out genuine HAM10000 test samples |
| **Overall Accuracy** | **64.59%** (0.6459) | Multiclass classification accuracy across 7 disease categories |
| **Balanced Accuracy** | **67.91%** (0.6791) | Macro-averaged recall balancing across class sizes |
| **Macro Precision** | **70.29%** (0.7029) | Unweighted mean precision across all 7 categories |
| **Macro Recall** | **67.91%** (0.6791) | Unweighted mean sensitivity across all 7 categories |
| **Macro F1-Score** | **68.68%** (0.6868) | Primary optimization metric across classes |
| **Weighted F1-Score** | **65.31%** (0.6531) | Sample-weighted F1 across test set |
| **Multiclass ROC-AUC (OvR)** | **92.20%** (0.9220) | One-vs-Rest macro-averaged area under the ROC curve |
| **Best Validation Macro F1** | **64.82%** (0.6482) | Achieved at Epoch 12/20 |
| **Total GPU Training Time** | **395.8 seconds** (~6.6 min) | Fast, efficient fine-tuning on Colab T4 GPU |

#### Per-Class Test Performance (Held-Out Test Set, N=257)

| Disease Class | Code | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Melanocytic Nevi** | `NV` | 79.49% | 75.61% | 77.50% | 41 |
| **Melanoma** | `MEL` | 46.67% | 47.73% | 47.19% | 44 |
| **Benign Keratosis-like Lesions** | `BKL` | 40.32% | 55.56% | 46.73% | 45 |
| **Basal Cell Carcinoma** | `BCC` | 80.00% | 74.42% | 77.11% | 43 |
| **Actinic Keratoses / IEC** | `AKIEC` | 72.73% | 53.33% | 61.54% | 45 |
| **Vascular Lesions** | `VASC` | 90.48% | 86.36% | 88.37% | 22 |
| **Dermatofibroma** | `DF` | 82.35% | 82.35% | 82.35% | 17 |

Artifacts stored at:
- Model Weights: `models/vit_model.pth` (456.76 MB)
- Metrics Summary: `training/results/vit_metrics.json`
- Epoch History: `training/results/vit_training_history.csv`
- Curves: `training/results/vit_plots/loss_curve.png`, `acc_curve.png`, `f1_curve.png`, `confusion_matrix.png`, `roc_curves.png`
- Visual Predictions: `training/results/vit_samples/test_*.png`

### Google Colab GPU Training Package
- **Notebook:** `training/train_vit.ipynb` Ã¢â‚¬â€ 19-cell ready-to-run notebook for Google Colab GPU (T4).
- **Archive:** `colab_vit_package.zip` (43.09 MB, 1,770 files) containing only genuine HAM10000 images (`1,233` train, `267` val, `257` test), manifests, and training code.

---

## Ã°Å¸Â§Â  Explainable AI (XAI) Attention Rollout (`services/attention.py`)

- **Theoretical Basis:** Implements Attention Rollout (Abnar & Zuidema, 2020) to recursively trace information flow from the `[CLS]` token across all 12 self-attention transformer blocks.
- **Hooking Architecture:** Dynamically intercepts attention weight matrices at `block.attn.attn_drop` (disabling `fused_attn` temporarily) to capture the genuine $(1, 12, 197, 197)$ attention tensors.
- **Recursive Formulation:**
  $$\tilde{A}^{(l)} = 0.5 \cdot A^{(l)} + 0.5 \cdot I$$
  $$R^{(l)} = \tilde{A}^{(l)} \cdot R^{(l-1)}$$
- **Spatial Mapping:** Extracts `[CLS]` token attention over the 196 patch tokens ($14 \times 14$ grid), normalizes min-max, upsamples to original image dimensions via bicubic interpolation (`cv2.INTER_CUBIC`), and blends with OpenCV `COLORMAP_JET` (60% original image, 40% heatmap).
- **Scope Note:** Visualizations reflect mathematical attention distribution across transformer tokens and do **NOT** represent clinical histological boundaries or pathological confirmation.

---

## Ã°Å¸Â©Âº Prototype Severity Grading & Treatment Cost Estimation (`services/severity.py`, `services/cost_estimation.py`)

### 1. Prototype Clinical Severity Triage
- Heuristic rule-based categorization evaluating disease pathology, model confidence score, and top-3 differential presence:
  - **High Concern (Red):** Predicted Melanoma (`MEL`) or active Melanoma in differentials with $>15\%$ probability. Immediate dermatological evaluation, dermoscopy, and excisional biopsy advised.
  - **Moderate Concern (Orange):** Basal Cell Carcinoma (`BCC`) or Actinic Keratoses (`AKIEC`). Locally invasive or precancerous lesions requiring timely clinical consultation.
  - **Low Concern (Green):** Typically benign lesions (`NV`, `BKL`, `DF`, `VASC`) with high model confidence ($>50\%$). Routine monitoring and periodic skin exams advised.
  - **Undetermined (Amber):** Any prediction where model confidence $<50\%$ or where high-risk lesions appear as competing differentials. Mandatory direct clinical consultation required.
- **Notice:** This is an algorithmic triage prototype and does **NOT** determine pathological staging, tumor thickness (Breslow depth), or definitive prognosis.

### 2. Benchmark Indian Healthcare Treatment Cost Breakdown (`data/treatment_costs.json`)
- Provides indicative price bands in Indian Rupees (INR) for private dermatological facilities across 5 care tiers:
  1. **Specialist Consultation:** Initial evaluation, full-body dermoscopy, digital record keeping (Ã¢â€šÂ¹500 Ã¢â‚¬â€œ Ã¢â€šÂ¹1,500).
  2. **Diagnostic Workup:** Histopathological biopsy, dermoscopic epiluminescence, immunohistochemistry (Ã¢â€šÂ¹1,500 Ã¢â‚¬â€œ Ã¢â€šÂ¹8,000).
  3. **Prescription Medications:** Topical chemotherapeutic agents, immunomodulators, healing ointments (Ã¢â€šÂ¹300 Ã¢â‚¬â€œ Ã¢â€šÂ¹4,000).
  4. **Procedural Intervention:** Excisional biopsy, cryotherapy, electrosurgery, or laser ablation (Ã¢â€šÂ¹1,500 Ã¢â‚¬â€œ Ã¢â€šÂ¹35,000).
  5. **Follow-Up & Monitoring:** Routine surveillance review and periodic photographic re-evaluation (Ã¢â€šÂ¹400 Ã¢â‚¬â€œ Ã¢â€šÂ¹1,200).
- **Notice:** Cost figures are illustrative academic benchmarks reflecting typical Indian private clinic and hospital tariffs; they are **NOT** binding quotations, insurance claims, or medical prescriptions.

---

## Ã°Å¸â€œâ€˜ Automated Medical Report Generation (`services/report.py`)

- Built using **ReportLab 5.1.0** without external headless browser dependencies.
- Generates publication-quality, multi-section clinical summary PDFs (`reports/report_<reference_id>.pdf`):
  - **Header & Reference:** Reference ID (`REF-YYYYMMDD-XXXXXX`), timestamp, patient confidentiality notice.
  - **Diagnostic Summary Table:** Primary classification, confidence %, prototype severity triage indicator, and U-Net lesion area coverage %.
  - **Top-3 Differential Diagnoses Table:** Ranked differentials with model probability percentages.
  - **4-Grid Multi-Modal Image Matrix:** (A) Original Dermoscopic Image, (B) U-Net Binary Mask, (C) Lesion Boundary Overlay, and (D) ViT Attention Rollout Heatmap.
  - **Treatment Cost Breakdown Table:** Itemized care categories and benchmark INR price ranges with total estimate.
  - **Prominent Medical & Academic Limitations Notice:** Clearly stating measured test accuracy (64.59%), melanoma recall (47.73%), and non-diagnostic research scope.

---

## Ã°Å¸Å’Â FastAPI REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves interactive web dashboard with image upload & results viewer |
| `GET` | `/history` | Serves persistent historical analysis records table and detail modal |
| `GET` | `/health` | Live system health check verifying U-Net, ViT, XAI services and SQLite DB |
| `POST` | `/api/predict` | Multipart form upload (`file`, `apply_dullrazor`) returning full diagnostic JSON |
| `GET` | `/api/history` | Standardized JSON endpoint retrieving all historical analysis records |
| `GET` | `/api/records` | Backward-compatible alias for `/api/history` |
| `GET` | `/api/results/{id}` | Detailed JSON representation for a single record by ID or Reference ID |
| `GET` | `/api/report/{id}` | Direct PDF download of the generated ReportLab clinical report |
| `GET` | `/docs` | Interactive OpenAPI Swagger UI documentation |

---

## Ã°Å¸Â§Âª Comprehensive Verification & Test Suites

The project includes 4 automated verification suites covering 100% of the monolithic architecture:

1. **End-to-End System Integration Test (`test_end_to_end.py`):**
   - Health check lifespan validation.
   - Corrupted and non-image rejection (HTTP 400).
   - Full inference execution: U-Net (threshold 0.70) + ViT-B/16 + XAI Attention Rollout.
   - On-disk verification of generated PNGs and PDF reports.
   - SQLite database row insertion, history retrieval, and PDF download verification.
2. **Phase 2 Regression Suite (`test_phase2.py`):**
   - 9 test suites verifying dataset splits, zero-leakage, coordinated augmentation, and live endpoints.
3. **Phase 3 U-Net Regression Suite (`test_phase3.py`):**
   - 10 test suites verifying U-Net architecture, forward/backward passes, CPU inference, and segmentation threshold 0.70.
4. **Phase 4 ViT Regression Suite (`test_phase4.py`):**
   - 12 test suites verifying ViT-B/16 architecture, freezing strategy, class mappings, prediction service, and live endpoints.

```bash
# Run the complete test suite
.\venv\Scripts\python.exe test_end_to_end.py
.\venv\Scripts\python.exe test_phase2.py
.\venv\Scripts\python.exe test_phase3.py
.\venv\Scripts\python.exe test_phase4.py
```

---

## Ã°Å¸Å¡â‚¬ Local Execution Instructions

### 1. Prerequisites
- Python 3.11 installed.
- Windows, macOS, or Linux environment.

### 2. Activate Virtual Environment & Install Dependencies
```bash
# Windows
.\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Launch the Application Server
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 4. Access the Application
- **Interactive Web Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Analysis History:** [http://127.0.0.1:8000/history](http://127.0.0.1:8000/history)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Ã¢Å¡Â Ã¯Â¸Â Known Model Performance & Clinical Limitations

1. **Classification Accuracy (64.59%):** While achieving strong ROC-AUC ($92.20\%$), the multiclass accuracy reflects the inherent difficulty of fine-grained dermatological discrimination across 7 classes on a compact genuine dataset (1,757 images).
2. **Melanoma Sensitivity (47.73% Recall):** Benign keratoses (`BKL`) and early stage melanomas (`MEL`) exhibit high morphological overlap in dermoscopic views. In ambiguous cases, the algorithmic severity service conservatively triggers `Undetermined` or flags Melanoma presence in the differential.
3. **Attention Maps as Interpretability Visualizations:** The attention rollout maps illustrate where the Vision Transformer self-attention focuses; they do **NOT** delineate pathological cell margins and must not be used as a replacement for histopathology.
4. **Treatment Costs as Indicative Benchmarks:** Cost bands reflect typical Indian private clinical rates in INR and are intended solely for preliminary financial transparency and patient education.

---

## Ã°Å¸â€œÅ’ Development Roadmap Status

- **Phase 1 (Completed):** Project structure, virtual environment, dependencies, database schema, web server, dashboard and history placeholders with medical disclaimer.
- **Phase 2 (Completed):** Dataset collection & balancing (HAM10000 & ISIC), DullRazor hair removal, image preprocessing, data augmentations, PyTorch Datasets & DataLoaders, split manifests, and endpoint standardization.
- **Phase 3 (Completed):** ResNet18-UNet architecture development, loss function (BCE + Dice), GPU training, test evaluation metrics (Dice 88.38%, IoU 79.53%), segmentation inference service (threshold 0.70).
- **Phase 4 (Completed):** Vision Transformer (ViT-B/16) transfer learning model development, GPU training (Acc 64.59%, Macro F1 68.68%, ROC-AUC 92.20%), prediction inference service with top-3 differentials.
- **Phase 5 (Completed):** Explainable AI Attention Rollout mechanism generating attention heatmaps across 12 transformer blocks.
- **Phase 6 (Completed):** Rule-based clinical prototype severity indicator (Low, Moderate, High, Undetermined).
- **Phase 7 (Completed):** Treatment cost estimation engine in INR across 5 care tiers (`data/treatment_costs.json`).
- **Phase 8 (Completed):** ReportLab comprehensive multi-element PDF medical report generator.
- **Phase 9 (Completed):** SQLite schema migrations for reference IDs, attention maps, overlays, and cost breakdowns.
- **Phase 10 (Completed):** Unified FastAPI backend pipeline with lifespan service pre-loading.
- **Phase 11 (Completed):** Interactive, responsive frontend UI (Dashboard, History table with detail modal) in forest green and white theme.
- **Phase 12 (Completed):** Comprehensive end-to-end integration test suite and full regression verification.

