"""
EfficientNet-B2 Training and Evaluation Pipeline for 9-Class Common Skin Disease Classification.

Specifications:
- Model Architecture: EfficientNet-B2 ('efficientnet_b2' via timm)
- ImageNet-1k pretrained weights
- Input Resolution: 260 x 260 with Aspect-Ratio-Preserving padding
- Optimization: AdamW (weight_decay=0.01) with CosineAnnealingLR
- Loss Function: Weighted Cross-Entropy Loss
- Staged Gradual Unfreezing:
  * Stage 1 (Epochs 1-3): Feature extractor frozen; train classification head only
  * Stage 2 (Epochs 4-8): Unfreeze top convolutional blocks (blocks 5, 6, conv_head)
  * Stage 3 (Epochs 9+): Unfreeze full backbone with discriminative learning rates
- Precision: CUDA Mixed Precision (torch.amp.autocast) with GradScaler
- Early Stopping: Patience = 5 epochs monitoring validation Macro F1-score
- Comprehensive Evaluation: Accuracy, Balanced Acc, Macro F1, Per-Class metrics,
  Confusion Matrix, Multiclass ROC-AUC (OvR with explicit labels), Calibration, CPU Latency
- Rapid CPU Smoke Test: --smoke-test flag for safe local pipeline verification
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

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
    get_efficientnet_transforms,
    save_efficientnet_checkpoint,
    COMMON_SKIN_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_DISPLAY_NAMES,
    NUM_COMMON_CLASSES
)

RESULTS_DIR = ROOT_DIR / "training" / "results"
PLOTS_DIR = RESULTS_DIR / "common_efficientnet_plots"
MODELS_DIR = ROOT_DIR / "models"


class CommonSkinDataset(Dataset):
    """Dataset wrapper reading manifests and applying EfficientNet 260x260 transforms."""
    def __init__(
        self,
        manifest_path: Path,
        root_dir: Optional[Path] = None,
        is_training: bool = False
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.root_dir = Path(root_dir) if root_dir else ROOT_DIR
        self.is_training = is_training
        self.transform = get_efficientnet_transforms(is_training=is_training, target_size=260)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_path = row["relative_path"]
        img_path = self.root_dir / rel_path

        if not img_path.exists():
            img_path = self.manifest_path.parent.parent / "common_skin" / "images" / row["filename"]
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")

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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics including OvR ROC-AUC with explicit labels."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(class_names)))
    )

    per_class = {}
    for i, c in enumerate(class_names):
        per_class[c] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1_score": float(f1_per[i]),
            "support": int(support[i])
        }

    # Multiclass ROC-AUC (OvR) with explicit labels
    roc_auc = None
    try:
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 1 and y_prob.shape[1] == len(class_names):
            roc_auc = float(roc_auc_score(
                y_true, y_prob,
                multi_class="ovr",
                average="macro",
                labels=list(range(len(class_names)))
            ))
    except Exception as e:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Confidence calibration metric: Expected Calibration Error (ECE)
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro": roc_auc,
        "expected_calibration_error": ece,
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) measuring confidence reliability."""
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


def plot_training_history(history: pd.DataFrame, save_path: Path):
    """Plots training and validation loss and Macro F1 curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="#00AFE0", lw=2)
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="#FF8500", lw=2)
    axes[0].set_title("Cross-Entropy Loss Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Macro F1
    axes[1].plot(history["epoch"], history["val_f1"], label="Val Macro F1", color="#10B981", lw=2)
    axes[1].plot(history["epoch"], history["val_acc"], label="Val Accuracy", color="#6366F1", lw=1.5, linestyle="--")
    axes[1].set_title("Validation Macro F1 & Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved training history curves to: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Plots normalized confusion matrix heatmap."""
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
        title="Test Confusion Matrix (EfficientNet-B2)",
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


def measure_cpu_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int] = (260, 260),
    num_runs: int = 25
) -> Dict[str, float]:
    """
    Benchmarks CPU inference latency per photograph in milliseconds.
    Safely transitions model to CPU and restores original device.
    """
    original_device = next(model.parameters()).device
    model.to(torch.device("cpu"))
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size, device="cpu")

    # Warm-up runs
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    # Restore original device
    if original_device.type != "cpu":
        model.to(original_device)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95))
    }


def train_common_efficientnet(
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 3e-4,
    patience: int = 5,
    smoke_test: bool = False,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes staged transfer learning training for EfficientNet-B2.
    """
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_checkpoint_dir = MODELS_DIR / "smoke_test" if smoke_test else MODELS_DIR
    target_results_dir = RESULTS_DIR / "smoke_test" if smoke_test else RESULTS_DIR
    target_plots_dir = target_results_dir / "common_efficientnet_plots"

    # Automatically ensure all required output directories exist
    target_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_results_dir.mkdir(parents=True, exist_ok=True)
    target_plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file_name = "common_skin_efficientnet_smoke.pth" if smoke_test else "common_skin_efficientnet.pth"
    best_checkpoint_path = target_checkpoint_dir / checkpoint_file_name
    metrics_json_path = target_results_dir / ("common_skin_efficientnet_smoke_metrics.json" if smoke_test else "common_skin_efficientnet_metrics.json")
    history_csv_path = target_results_dir / ("common_skin_efficientnet_smoke_history.csv" if smoke_test else "common_skin_efficientnet_history.csv")
    curves_plot_path = target_plots_dir / "common_skin_efficientnet_training_curves.png"
    cm_plot_path = target_plots_dir / "common_skin_efficientnet_confusion_matrix.png"

    print("\n=======================================================")
    print("COMMON SKIN DISEASE EfficientNet-B2 TRAINING PIPELINE")
    print(f"Device: {device} | Smoke Test: {smoke_test}")
    print(f"Target Checkpoint: {best_checkpoint_path}")
    print("=======================================================")

    # 1. Dataloaders
    manifest_dir = ROOT_DIR / "data" / "manifests"
    train_manifest = manifest_dir / "common_skin_train.csv"
    val_manifest = manifest_dir / "common_skin_val.csv"
    test_manifest = manifest_dir / "common_skin_test.csv"

    effective_batch_size = 2 if smoke_test else batch_size
    train_dataset = CommonSkinDataset(train_manifest, root_dir=ROOT_DIR, is_training=True)
    val_dataset = CommonSkinDataset(val_manifest, root_dir=ROOT_DIR, is_training=False)
    test_dataset = CommonSkinDataset(test_manifest, root_dir=ROOT_DIR, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)

    print(f"Data: {len(train_dataset)} Train | {len(val_dataset)} Val | {len(test_dataset)} Test (Batch size: {effective_batch_size})")

    # 2. Weighted Cross-Entropy Loss
    class_weights_tensor = None
    meta_path = manifest_dir / "common_skin_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        weights = meta.get("class_weights")
        if weights and len(weights) == NUM_COMMON_CLASSES:
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            print("Loaded class weights for Weighted Cross-Entropy Loss.")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 3. Model Instantiation
    print(f"Instantiating EfficientNet-B2 (efficientnet_b2 via timm)...")
    model = CommonSkinEfficientNetB2(
        num_classes=NUM_COMMON_CLASSES,
        pretrained=not smoke_test,
        drop_rate=0.3,
        drop_path_rate=0.1
    ).to(device)

    # Initial Stage 1: Freeze backbone, train only classification head
    if not smoke_test:
        model.freeze_backbone()
        print("Stage 1 Initialized: Backbone frozen. Training classification head only.")

    param_counts = model.count_parameters()
    print(f"Parameters: Total={param_counts['total_parameters']:,} | Trainable={param_counts['trainable_parameters']:,}")

    # Initial Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_macro_f1 = 0.0
    patience_counter = 0
    history_records = []

    actual_epochs = 1 if smoke_test else epochs
    max_train_batches = 3 if smoke_test else len(train_loader)
    max_val_batches = 2 if smoke_test else len(val_loader)

    start_time = time.time()

    for epoch in range(1, actual_epochs + 1):
        # Staged unfreezing schedule (when not smoke testing)
        if not smoke_test:
            if epoch == 4:
                print("\n>>> Stage 2: Unfreezing top convolutional blocks (blocks 5, 6, conv_head)...")
                model.unfreeze_top_blocks()
                param_groups = model.get_discriminative_param_groups(
                    lr_head=lr * 0.8,
                    lr_backbone_upper=lr * 0.2,
                    lr_backbone_lower=1e-5
                )
                optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch + 1, eta_min=1e-6)

            elif epoch == 9:
                print("\n>>> Stage 3: Unfreezing entire backbone with discriminative learning rates...")
                model.unfreeze_all()
                param_groups = model.get_discriminative_param_groups(
                    lr_head=lr * 0.4,
                    lr_backbone_upper=lr * 0.1,
                    lr_backbone_lower=lr * 0.03
                )
                optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch + 1, eta_min=1e-6)

        # Train Epoch
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for b_idx, (images, targets, _) in enumerate(train_loader):
            if b_idx >= max_train_batches:
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += images.size(0)

        scheduler.step()

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Validation Epoch
        model.eval()
        val_loss = 0.0
        val_trues, val_preds, val_probs = [], [], []

        with torch.no_grad():
            for b_idx, (images, targets, _) in enumerate(val_loader):
                if b_idx >= max_val_batches:
                    break
                images, targets = images.to(device), targets.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                val_preds.extend(preds)
                val_trues.extend(targets.cpu().numpy())
                val_probs.extend(probs)

        val_total = len(val_trues)
        epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_metrics = compute_metrics(
            np.array(val_trues), np.array(val_preds), np.array(val_probs), COMMON_SKIN_CLASSES
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{actual_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val F1: {val_metrics['macro_f1']*100:.2f}% | "
            f"Val Acc: {val_metrics['accuracy']*100:.1f}% | LR: {current_lr:.6f}"
        )

        history_records.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "lr": current_lr
        })

        # Save Best Checkpoint
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            save_efficientnet_checkpoint(
                model=model,
                save_path=best_checkpoint_path,
                epoch=epoch,
                best_macro_f1=best_macro_f1,
                val_metrics=val_metrics,
                optimizer=optimizer,
                smoke_test=smoke_test
            )
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"\n[Early Stopping] No Macro F1 improvement for {patience} consecutive epochs.")
                break

    # Save training history
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training history to: {history_csv_path}")

    # Plot curves
    plot_training_history(history_df, curves_plot_path)

    # --- HELD-OUT TEST EVALUATION ---
    print("\n--- Evaluating Best Model on Held-Out Test Set ---")
    # Load best checkpoint if saved
    if best_checkpoint_path.exists():
        from models.common_skin_efficientnet import load_efficientnet_checkpoint
        model, _ = load_efficientnet_checkpoint(best_checkpoint_path, device=device, eval_mode=True)
    else:
        model.eval()

    test_trues, test_preds, test_probs = [], [], []
    max_test_batches = 2 if smoke_test else len(test_loader)

    with torch.no_grad():
        for b_idx, (images, targets, _) in enumerate(test_loader):
            if b_idx >= max_test_batches:
                break
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            test_preds.extend(preds)
            test_trues.extend(targets.cpu().numpy())
            test_probs.extend(probs)

    test_metrics = compute_metrics(
        np.array(test_trues), np.array(test_preds), np.array(test_probs), COMMON_SKIN_CLASSES
    )

    # Plot Confusion Matrix
    cm_arr = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm_arr, COMMON_SKIN_CLASSES, cm_plot_path)

    # CPU Latency Benchmark (guaranteed safe CPU evaluation)
    latency_info = measure_cpu_inference_latency(model, input_size=(260, 260), num_runs=25)

    final_evaluation = {
        "model_architecture": "efficientnet_b2",
        "num_classes": NUM_COMMON_CLASSES,
        "classes": COMMON_SKIN_CLASSES,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "input_resolution": [260, 260],
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_samples": len(test_trues),
        "smoke_test": smoke_test,
        "overall_accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "roc_auc_macro": test_metrics["roc_auc_macro"],
        "expected_calibration_error": test_metrics["expected_calibration_error"],
        "cpu_latency": latency_info,
        "per_class_metrics": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"]
    }

    with open(metrics_json_path, "w") as f:
        json.dump(final_evaluation, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_json_path}")

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f"TRAINING COMPLETE IN {elapsed:.1f}s")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"CPU Latency (mean): {latency_info['mean_latency_ms']:.2f} ms")
    print(f"=======================================================")

    return final_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Common Skin EfficientNet-B2 Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid CPU smoke test")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu, cuda)")
    args = parser.parse_args()

    train_common_efficientnet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        smoke_test=args.smoke_test,
        device_str=args.device
    )
