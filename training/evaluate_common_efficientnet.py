"""
Standalone Evaluation Script for Trained EfficientNet-B2 Model.
Evaluates genuine checkpoint on held-out test clinical photographs without retraining.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.common_skin_efficientnet import (
    CommonSkinEfficientNetB2,
    load_efficientnet_checkpoint,
    get_efficientnet_transforms,
    COMMON_SKIN_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_DISPLAY_NAMES,
    NUM_COMMON_CLASSES
)

RESULTS_DIR = ROOT_DIR / "training" / "results"
PLOTS_DIR = RESULTS_DIR / "common_efficientnet_plots"
MODELS_DIR = ROOT_DIR / "models"


class HeldOutTestDataset(Dataset):
    """Dataset wrapper for held-out clinical test images."""
    def __init__(self, manifest_path: Path, root_dir: Path):
        self.df = pd.read_csv(manifest_path)
        self.root_dir = root_dir
        self.transform = get_efficientnet_transforms(is_training=False, target_size=260)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_path = row["relative_path"]
        img_path = self.root_dir / rel_path

        if not img_path.exists():
            img_path = self.root_dir / "data" / "common_skin" / "images" / row["filename"]
            if not img_path.exists():
                raise FileNotFoundError(f"Test image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        label = int(row["label_idx"])

        metadata = {
            "case_id": str(row.get("case_id", "")),
            "target_class": str(row.get("target_class", "")),
            "skin_tone": str(row.get("skin_tone", "UNKNOWN")),
            "filename": str(row.get("filename", ""))
        }
        return tensor, label, metadata


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Plots and saves normalized confusion matrix."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    short_names = [CLASS_DISPLAY_NAMES[c] for c in class_names]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short_names,
        yticklabels=short_names,
        title="Held-Out Test Confusion Matrix (EfficientNet-B2)",
        ylabel="True Disease Category",
        xlabel="Predicted Disease Category"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8.5
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")


def measure_cpu_inference_latency(model: nn.Module, num_runs: int = 50) -> Dict[str, float]:
    """Measures single-image CPU inference latency across 50 runs."""
    model.eval()
    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 260, 260, device="cpu")

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "num_runs": num_runs
    }


def evaluate_trained_efficientnet() -> Dict[str, Any]:
    checkpoint_path = MODELS_DIR / "common_skin_efficientnet.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Ensure output directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("EFFICIENTNET-B2 HELD-OUT TEST EVALUATION")
    print(f"Checkpoint: {checkpoint_path}")
    print("=======================================================")

    device = torch.device("cpu")
    model, metadata = load_efficientnet_checkpoint(checkpoint_path, device=device, eval_mode=True)
    print(f"Loaded checkpoint trained to Epoch: {metadata.get('epoch')}")
    print(f"Best Validation Macro F1 recorded in checkpoint: {metadata.get('best_macro_f1', 0.0)*100:.2f}%")

    # Recover genuine validation metrics from checkpoint if present
    if metadata.get("val_metrics"):
        val_metrics_path = RESULTS_DIR / "common_skin_efficientnet_val_metrics.json"
        with open(val_metrics_path, "w") as f:
            json.dump({
                "model": "efficientnet_b2",
                "best_epoch": metadata.get("epoch"),
                "best_macro_f1": metadata.get("best_macro_f1"),
                "val_metrics": metadata.get("val_metrics")
            }, f, indent=2)
        print(f"Recovered genuine validation metrics saved to: {val_metrics_path}")

    test_manifest = ROOT_DIR / "data" / "manifests" / "common_skin_test.csv"
    test_dataset = HeldOutTestDataset(test_manifest, root_dir=ROOT_DIR)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Evaluating {len(test_dataset)} genuine held-out clinical photographs...")

    test_trues, test_preds, test_probs = [], [], []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            test_preds.extend(preds)
            test_trues.extend(targets.numpy())
            test_probs.extend(probs)

    y_true = np.array(test_trues)
    y_pred = np.array(test_preds)
    y_prob = np.array(test_probs)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(NUM_COMMON_CLASSES))
    )

    per_class = {}
    for i, c in enumerate(COMMON_SKIN_CLASSES):
        per_class[c] = {
            "display_name": CLASS_DISPLAY_NAMES[c],
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1_score": float(f1_per[i]),
            "support": int(support[i])
        }

    # Multiclass ROC-AUC (OvR) with explicit labels
    try:
        roc_auc = float(roc_auc_score(
            y_true, y_prob,
            multi_class="ovr",
            average="macro",
            labels=list(range(NUM_COMMON_CLASSES))
        ))
    except Exception as e:
        print(f"ROC-AUC computation notice: {e}")
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_COMMON_CLASSES)))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    cm_plot_path = PLOTS_DIR / "common_skin_efficientnet_confusion_matrix.png"
    plot_confusion_matrix(cm, COMMON_SKIN_CLASSES, cm_plot_path)

    print("\nMeasuring CPU inference latency (50 single-image evaluations)...")
    latency = measure_cpu_inference_latency(model, num_runs=50)

    final_evaluation = {
        "model_architecture": "efficientnet_b2",
        "num_classes": NUM_COMMON_CLASSES,
        "classes": COMMON_SKIN_CLASSES,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "input_resolution": [260, 260],
        "preprocessing": "aspect_ratio_preserving_pad_260",
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_samples": len(y_true),
        "best_training_epoch": metadata.get("epoch"),
        "best_val_macro_f1": metadata.get("best_macro_f1"),
        "overall_accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro": roc_auc,
        "expected_calibration_error": ece,
        "cpu_latency": latency,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist()
    }

    metrics_json_path = RESULTS_DIR / "common_skin_efficientnet_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(final_evaluation, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_json_path}")

    print("\n=======================================================")
    print("EFFICIENTNET-B2 EVALUATION RESULTS")
    print("=======================================================")
    print(f"Overall Accuracy:     {acc*100:.2f}%")
    print(f"Balanced Accuracy:    {bal_acc*100:.2f}%")
    print(f"Macro Precision:      {macro_p*100:.2f}%")
    print(f"Macro Recall:         {macro_r*100:.2f}%")
    print(f"Macro F1-Score:       {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score:    {weighted_f1*100:.2f}%")
    print(f"Multiclass ROC-AUC:   {roc_auc:.4f}" if roc_auc else "Multiclass ROC-AUC:   N/A")
    print(f"Calibration (ECE):    {ece*100:.2f}%")
    print(f"CPU Latency (mean):   {latency['mean_latency_ms']:.2f} ms")
    print("=======================================================")

    return final_evaluation


if __name__ == "__main__":
    evaluate_trained_efficientnet()
