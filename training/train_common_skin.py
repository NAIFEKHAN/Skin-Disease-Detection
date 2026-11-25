"""
Training and Evaluation Pipeline for 9-Class Common Skin Disease ViT-B/16 Model.

Specifications:
- Architecture: Vision Transformer Base (ViT-B/16, 'vit_base_patch16_224')
- Optimizer: AdamW (weight_decay=0.01)
- Loss: Weighted Cross-Entropy Loss
- Learning Rate Schedule: CosineAnnealingLR or StepLR
- Precision: CUDA Mixed Precision (torch.amp.autocast) with GradScaler
- Checkpointing: Best checkpoint selected via validation Macro F1-score
- Early Stopping: Patience = 5 epochs
- Local CPU Smoke Test: --smoke-test flag for rapid verification
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
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.common_skin_vit import (
    CommonSkinViTClassifier,
    save_common_skin_checkpoint,
    COMMON_SKIN_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_DISPLAY_NAMES,
    NUM_COMMON_CLASSES
)
from training.common_skin_dataset import build_common_skin_dataloaders

RESULTS_DIR = ROOT_DIR / "training" / "results"
PLOTS_DIR = RESULTS_DIR / "common_skin_plots"
SAMPLES_DIR = RESULTS_DIR / "common_skin_samples"
MODELS_DIR = ROOT_DIR / "models"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
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

    # Multiclass ROC-AUC (OvR)
    roc_auc = None
    try:
        # Check if at least 2 distinct classes present in y_true
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            ))
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

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
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


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
    axes[1].plot(history["epoch"], history["train_f1"], label="Train Macro F1", color="#00AFE0", lw=2)
    axes[1].plot(history["epoch"], history["val_f1"], label="Val Macro F1", color="#14532D", lw=2)
    axes[1].set_title("Validation Macro F1 Score Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Plots normalized confusion matrix heatmap using matplotlib."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    display_labels = [CLASS_DISPLAY_NAMES.get(c, c) for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)

    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Common Skin ViT — Normalized Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Diagnosis", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_common_skin_model(
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    patience: int = 5,
    unfreeze_blocks: int = 2,
    smoke_test: bool = False,
    device_str: Optional[str] = None
):
    """Main training execution function."""
    if smoke_test:
        smoke_results_dir = RESULTS_DIR / "smoke_test"
        smoke_plots_dir = smoke_results_dir / "plots"
        smoke_models_dir = MODELS_DIR / "smoke_test"

        smoke_results_dir.mkdir(parents=True, exist_ok=True)
        smoke_plots_dir.mkdir(parents=True, exist_ok=True)
        smoke_models_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_save_path = smoke_models_dir / "common_skin_smoke_model.pth"
        history_csv_path = smoke_results_dir / "common_skin_smoke_training_history.csv"
        metrics_json_path = smoke_results_dir / "common_skin_smoke_metrics.json"
        curves_plot_path = smoke_plots_dir / "common_skin_smoke_training_curves.png"
        cm_plot_path = smoke_plots_dir / "common_skin_smoke_confusion_matrix.png"
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        checkpoint_save_path = MODELS_DIR / "common_skin_model.pth"
        history_csv_path = RESULTS_DIR / "common_skin_training_history.csv"
        metrics_json_path = RESULTS_DIR / "common_skin_metrics.json"
        curves_plot_path = PLOTS_DIR / "common_skin_training_curves.png"
        cm_plot_path = PLOTS_DIR / "common_skin_confusion_matrix.png"

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=======================================================")
    print(f"COMMON SKIN DISEASE ViT-B/16 TRAINING PIPELINE")
    print(f"Device: {device} | Smoke Test: {smoke_test}")
    print(f"Target Checkpoint: {checkpoint_save_path}")
    print(f"=======================================================")

    # 1. Load DataLoaders (Use batch_size=2 for smoke test to prevent CPU OOM)
    effective_batch_size = 2 if smoke_test else batch_size
    train_loader, val_loader, test_loader, stats = build_common_skin_dataloaders(
        batch_size=effective_batch_size
    )
    print(f"Data: {stats['train_samples']} Train | {stats['val_samples']} Val | {stats['test_samples']} Test (Batch size: {effective_batch_size})")

    # 2. Load Class Weights from metadata if available
    class_weights_tensor = None
    meta_path = ROOT_DIR / "data" / "manifests" / "common_skin_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        weights = meta.get("class_weights")
        if weights and len(weights) == NUM_COMMON_CLASSES:
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            print("Loaded class weights for Weighted Cross-Entropy Loss.")

    # 3. Instantiate ViT Model
    print(f"Instantiating ViT-B/16 (vit_base_patch16_224)...")
    # For smoke test on CPU without network, pretrained=False if needed
    model = CommonSkinViTClassifier(
        num_classes=NUM_COMMON_CLASSES,
        pretrained=not smoke_test,
        freeze_backbone=not smoke_test,
        unfreeze_blocks=unfreeze_blocks,
        drop_rate=0.1
    ).to(device)

    param_counts = model.count_parameters()
    print(f"Parameters: Total={param_counts['total_parameters']:,} | Trainable={param_counts['trainable_parameters']:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_macro_f1 = 0.0
    patience_counter = 0
    history_records = []
    best_checkpoint_path = MODELS_DIR / "common_skin_model.pth"

    actual_epochs = 1 if smoke_test else epochs
    max_train_batches = 3 if smoke_test else len(train_loader)
    max_val_batches = 2 if smoke_test else len(val_loader)

    start_time = time.time()

    for epoch in range(1, actual_epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_preds, train_trues, train_probs = [], [], []

        for b_idx, (images, targets, _) in enumerate(train_loader):
            if b_idx >= max_train_batches:
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            train_preds.extend(preds)
            train_trues.extend(targets.cpu().numpy())
            train_probs.extend(probs)

        scheduler.step()
        train_loss /= len(train_trues)
        train_metrics = compute_metrics(
            np.array(train_trues), np.array(train_preds), np.array(train_probs), COMMON_SKIN_CLASSES
        )

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_preds, val_trues, val_probs = [], [], []

        with torch.no_grad():
            for b_idx, (images, targets, _) in enumerate(val_loader):
                if b_idx >= max_val_batches:
                    break
                images, targets = images.to(device), targets.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, targets)

                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                val_preds.extend(preds)
                val_trues.extend(targets.cpu().numpy())
                val_probs.extend(probs)

        val_loss /= len(val_trues)
        val_metrics = compute_metrics(
            np.array(val_trues), np.array(val_preds), np.array(val_probs), COMMON_SKIN_CLASSES
        )

        print(
            f"Epoch {epoch:02d}/{actual_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        history_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Checkpointing based on validation Macro F1
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_macro_f1 or smoke_test:
            best_macro_f1 = val_f1
            patience_counter = 0
            save_common_skin_checkpoint(
                model=model,
                save_path=checkpoint_save_path,
                epoch=epoch,
                best_macro_f1=best_macro_f1,
                val_metrics=val_metrics,
                optimizer=optimizer,
                smoke_test=smoke_test
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training history to: {history_csv_path}")

    # Plot loss and F1 curves
    plot_training_history(history_df, curves_plot_path)

    # --- TEST EVALUATION ---
    print("\n--- Evaluating Best Model on Held-Out Test Set ---")
    model.eval()
    test_trues, test_preds, test_probs, test_metadata = [], [], [], []
    max_test_batches = 2 if smoke_test else len(test_loader)

    with torch.no_grad():
        for b_idx, (images, targets, meta) in enumerate(test_loader):
            if b_idx >= max_test_batches:
                break
            images = images.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            test_preds.extend(preds)
            test_trues.extend(targets.numpy())
            test_probs.extend(probs)
            for i in range(len(preds)):
                test_metadata.append({
                    "case_id": meta["case_id"][i],
                    "target_class": meta["target_class"][i],
                    "skin_tone": meta["skin_tone"][i],
                    "filename": meta["filename"][i]
                })

    test_metrics = compute_metrics(
        np.array(test_trues), np.array(test_preds), np.array(test_probs), COMMON_SKIN_CLASSES
    )

    # Plot Test Confusion Matrix
    cm_arr = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm_arr, COMMON_SKIN_CLASSES, cm_plot_path)

    # Breakdown by skin tone if available
    skin_tone_breakdown = {}
    test_trues_arr = np.array(test_trues)
    test_preds_arr = np.array(test_preds)
    tones = set(m["skin_tone"] for m in test_metadata)
    for tone in tones:
        indices = [i for i, m in enumerate(test_metadata) if m["skin_tone"] == tone]
        if indices:
            sub_acc = float(accuracy_score(test_trues_arr[indices], test_preds_arr[indices]))
            skin_tone_breakdown[tone] = {
                "count": len(indices),
                "accuracy": sub_acc
            }

    final_evaluation = {
        "model_architecture": "vit_base_patch16_224",
        "num_classes": NUM_COMMON_CLASSES,
        "classes": COMMON_SKIN_CLASSES,
        "class_display_names": CLASS_DISPLAY_NAMES,
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
        "per_class_metrics": test_metrics["per_class"],
        "skin_tone_breakdown": skin_tone_breakdown,
        "confusion_matrix": test_metrics["confusion_matrix"]
    }

    with open(metrics_json_path, "w") as f:
        json.dump(final_evaluation, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_json_path}")

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f"TRAINING COMPLETE IN {elapsed:.1f}s")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"=======================================================")

    return final_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Common Skin ViT Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="Number of transformer blocks to fine-tune")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid CPU smoke test")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu, cuda)")
    args = parser.parse_args()

    train_common_skin_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        unfreeze_blocks=args.unfreeze_blocks,
        smoke_test=args.smoke_test,
        device_str=args.device
    )
