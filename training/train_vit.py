"""
Training and Evaluation Pipeline for Vision Transformer (ViT-B/16) Skin Lesion Classification.
Phase 4 Implementation.

Features:
- Pretrained ViT-B/16 ('vit_base_patch16_224') fine-tuning with discriminative learning rates.
- Class-weighted Cross-Entropy loss for imbalanced 7-class dermatological categories.
- Linear warmup + Cosine Annealing learning rate scheduling.
- Early stopping monitoring validation Macro F1-score (patience = 5).
- Comprehensive evaluation: Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision, Recall,
  One-vs-Rest Multiclass ROC-AUC, and Confusion Matrix.
- Evaluation curves (loss, accuracy, F1, confusion matrix, ROC curves) and genuine sample visual cards.
- Checkpoint recovery and resume support.
- Fully isolated held-out test evaluation.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.vit import (
    build_vit_model,
    VisionTransformerClassifier,
    DISEASE_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_DISPLAY_NAMES,
    VIT_MODEL_NAME
)
NUM_CLASSES = len(DISEASE_CLASSES)
from training.dataset import (
    get_classification_dataloaders,
    compute_class_weights,
    get_default_data_root
)

# Default Hyperparameters
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_HEAD_LR = 1e-4
DEFAULT_BACKBONE_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seeds for reproducible initialization and splitting."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """Executes one training epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    count = 0
    use_amp = (scaler is not None and device.type == "cuda")

    for i, (images, labels, _) in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        running_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        count += 1

    if count == 0:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    epoch_loss = running_loss / count
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "loss": float(epoch_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1)
    }


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates model on validation or test sets.
    Returns:
    - metrics_dict: comprehensive dictionary of scores
    - all_preds: 1D array of predicted class indices
    - all_targets: 1D array of true class labels
    - all_probs: 2D array of softmax probabilities of shape (N, num_classes)
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    count = 0

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            running_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            count += 1

    if count == 0:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}, np.array([]), np.array([]), np.array([])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    epoch_loss = running_loss / count
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )

    per_class_metrics = {}
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        per_class_metrics[c_name] = {
            "display_name": CLASS_DISPLAY_NAMES[c_name],
            "precision": round(float(per_class_p[c_idx]), 4),
            "recall": round(float(per_class_r[c_idx]), 4),
            "f1_score": round(float(per_class_f1[c_idx]), 4),
            "support": int(per_class_support[c_idx])
        }

    # Multiclass ROC-AUC (One-vs-Rest)
    try:
        roc_auc_ovr = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc_ovr = 0.0

    metrics = {
        "loss": float(epoch_loss),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc_ovr": float(roc_auc_ovr),
        "melanoma_recall": float(per_class_r[CLASS_TO_IDX["mel"]]),
        "per_class": per_class_metrics
    }
    return metrics, all_preds, all_targets, all_probs


def generate_evaluation_artifacts(
    history: List[Dict[str, Any]],
    test_metrics: Dict[str, Any],
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    all_probs: np.ndarray,
    test_loader: torch.utils.data.DataLoader,
    results_dir: Path,
    data_root: Path
):
    """Generates loss/accuracy/F1 plots, confusion matrix, ROC curves, and test visual sample cards."""
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "vit_plots"
    samples_dir = results_dir / "vit_samples"
    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training history CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "vit_training_history.csv", index=False)

    # 2. Metrics JSON
    with open(results_dir / "vit_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # 3. Plots
    if len(history_df) >= 1:
        epochs = history_df["epoch"]

        # Loss Curve
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["train_loss"], label="Train Loss", color="#1b4332", lw=2)
        plt.plot(epochs, history_df["val_loss"], label="Val Loss", color="#d90429", lw=2, linestyle="--")
        plt.title("ViT-B/16 Training vs Validation Loss", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Cross-Entropy Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curve.png", dpi=150)
        plt.close()

        # Accuracy Curve
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["train_acc"], label="Train Accuracy", color="#2d6a4f", lw=2)
        plt.plot(epochs, history_df["val_acc"], label="Val Accuracy", color="#40916c", lw=2, linestyle="--")
        plt.title("ViT-B/16 Training vs Validation Accuracy", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "acc_curve.png", dpi=150)
        plt.close()

        # Macro F1 Curve
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["val_macro_f1"], label="Validation Macro F1", color="#52b788", lw=2)
        plt.title("Validation Macro F1-Score across Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "f1_curve.png", dpi=150)
        plt.close()

    # 4. Confusion Matrix Heatmap (Pure Matplotlib)
    if len(all_targets) > 0 and len(all_preds) > 0:
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
        plt.figure(figsize=(8, 7))
        plt.imshow(cm, interpolation="nearest", cmap="Greens")
        plt.title("ViT-B/16 Confusion Matrix (Held-Out Test Set)", fontsize=12, fontweight="bold")
        plt.colorbar()

        tick_marks = np.arange(NUM_CLASSES)
        labels_display = [f"{c.upper()}\n({CLASS_DISPLAY_NAMES[c][:8]})" for c in DISEASE_CLASSES]
        plt.xticks(tick_marks, labels_display, fontsize=8)
        plt.yticks(tick_marks, labels_display, fontsize=8)

        # Annotate numbers
        thresh = cm.max() / 2.0
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                plt.text(
                    c, r, format(cm[r, c], "d"),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontsize=9
                )

        plt.xlabel("Predicted Class", fontweight="bold")
        plt.ylabel("True Class", fontweight="bold")
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrix.png", dpi=150)
        plt.close()

    # 5. Multiclass ROC Curves (One-vs-Rest)
    if len(all_targets) > 0 and len(all_probs) > 0:
        plt.figure(figsize=(8, 6))
        for c_idx, c_name in enumerate(DISEASE_CLASSES):
            bin_targets = (all_targets == c_idx).astype(int)
            if bin_targets.sum() > 0:
                fpr, tpr, _ = roc_curve(bin_targets, all_probs[:, c_idx])
                auc_c = roc_auc_score(bin_targets, all_probs[:, c_idx])
                plt.plot(fpr, tpr, lw=1.5, label=f"{c_name.upper()} (AUC = {auc_c:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontweight="bold")
        plt.ylabel("True Positive Rate (Recall)", fontweight="bold")
        plt.title("ViT-B/16 Multiclass One-vs-Rest ROC Curves", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "roc_curves.png", dpi=150)
        plt.close()

    # 6. Test sample prediction cards (3 correct + 3 misclassified)
    test_ds = test_loader.dataset
    correct_indices = [i for i, (t, p) in enumerate(zip(all_targets, all_preds)) if t == p]
    misclassified_indices = [i for i, (t, p) in enumerate(zip(all_targets, all_preds)) if t != p]

    # Prioritize Melanoma misclassification if present
    selected_indices = correct_indices[:3] + misclassified_indices[:3]

    for rank, idx in enumerate(selected_indices):
        true_lbl = all_targets[idx]
        pred_lbl = all_preds[idx]
        probs_sample = all_probs[idx]
        conf = probs_sample[pred_lbl]

        # Fetch image
        row = test_ds.df.iloc[idx]
        img_id = row["image_id"]
        rel_img = str(row["image_path"]).replace("\\", "/")
        full_img_path = data_root / rel_img

        bgr = cv2.imread(str(full_img_path))
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        # Plot sample card
        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.3]})
        ax_img.imshow(rgb)
        status_tag = "CORRECT" if true_lbl == pred_lbl else "MISCLASSIFIED"
        tag_color = "#2d6a4f" if true_lbl == pred_lbl else "#b91c1c"
        ax_img.set_title(
            f"{status_tag} | {img_id}\nTrue: {DISEASE_CLASSES[true_lbl].upper()} ({CLASS_DISPLAY_NAMES[DISEASE_CLASSES[true_lbl]]})\n"
            f"Pred: {DISEASE_CLASSES[pred_lbl].upper()} ({conf*100:.1f}%)",
            fontsize=9, color=tag_color, fontweight="bold"
        )
        ax_img.axis("off")

        # Bar chart of top probabilities
        y_pos = np.arange(NUM_CLASSES)
        ax_bar.barh(y_pos, probs_sample, color="#40916c", alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([f"{c.upper()} ({CLASS_DISPLAY_NAMES[c][:10]})" for c in DISEASE_CLASSES], fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Probability", fontsize=9)
        ax_bar.set_xlim([0.0, 1.0])
        ax_bar.set_title("Class-wise Prediction Probabilities", fontsize=10, fontweight="bold")
        ax_bar.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        sample_tag = "correct" if true_lbl == pred_lbl else "misclassified"
        fig.savefig(samples_dir / f"test_{sample_tag}_{rank+1}_{img_id}.png", dpi=150)
        plt.close(fig)

    print(f"✔ Generated ViT evaluation plots in {plots_dir}")
    print(f"✔ Generated test sample prediction cards in {samples_dir}")


def run_training(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    head_lr: float = DEFAULT_HEAD_LR,
    backbone_lr: float = DEFAULT_BACKBONE_LR,
    patience: int = DEFAULT_PATIENCE,
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    resume_path: Optional[Path] = None,
    freeze_backbone: bool = True,
    unfreeze_blocks: int = 2,
    smoke_test: bool = False
) -> List[Dict[str, Any]]:
    """Main training execution function for ViT-B/16."""
    set_seed(RANDOM_SEED)

    if data_root is None:
        data_root = get_default_data_root()
    if output_dir is None:
        output_dir = BASE_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir if smoke_test else (BASE_DIR / "training" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    manifests_dir = data_root / "manifests"

    # 1. Compute class weights for imbalanced Cross-Entropy
    train_manifest = manifests_dir / "classification_train.csv"
    class_weights = compute_class_weights(train_manifest, num_classes=NUM_CLASSES).to(device)
    print(f"Computed Class Weights: {[round(w, 4) for w in class_weights.tolist()]}")

    # 2. DataLoaders
    print("Loading classification DataLoaders...")
    loaders = get_classification_dataloaders(
        manifests_dir=manifests_dir,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
        use_weighted_sampler=False,
        apply_hair_removal=False
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    print(f"Train samples: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 3. Model
    print(f"Initializing ViT-B/16 (freeze_backbone={freeze_backbone}, unfreeze_blocks={unfreeze_blocks})...")
    model = build_vit_model(
        pretrained=True,
        num_classes=NUM_CLASSES,
        freeze_backbone=freeze_backbone,
        unfreeze_blocks=unfreeze_blocks,
        device=device
    )
    p_info = model.get_parameter_count()
    print(f"Total Params: {p_info['total_parameters']:,} | Trainable: {p_info['trainable_parameters']:,} | Frozen: {p_info['frozen_parameters']:,}")

    # 4. Criterion, Optimizer, Scheduler, Scaler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    param_groups = model.get_parameter_groups(head_lr=head_lr, backbone_lr=backbone_lr)
    optimizer = optim.AdamW(param_groups, weight_decay=DEFAULT_WEIGHT_DECAY)

    # Warmup (2 epochs) + Cosine Annealing
    warmup_epochs = 2
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    use_amp = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_macro_f1 = -1.0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    # Checkpoint recovery
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_macro_f1 = checkpoint.get("best_val_macro_f1", -1.0)
        print(f"Resumed at epoch {start_epoch} (Best Val Macro F1: {best_val_macro_f1:.4f})")

    if smoke_test:
        print("\n--- RUNNING SMOKE TEST (1 mini-epoch, 2 batches) ---")
        epochs = 1
        max_batches = 2
    else:
        max_batches = None

    best_checkpoint_path = output_dir / "vit_model.pth"

    print("\nStarting ViT training loop...")
    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        ep_start = time.time()

        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_batches=max_batches
        )

        val_res, _, _, _ = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches
        )

        scheduler.step()
        ep_duration = time.time() - ep_start
        current_head_lr = optimizer.param_groups[-1]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": round(train_res["loss"], 4),
            "train_acc": round(train_res["accuracy"], 4),
            "train_macro_f1": round(train_res["macro_f1"], 4),
            "val_loss": round(val_res["loss"], 4),
            "val_acc": round(val_res["accuracy"], 4),
            "val_bal_acc": round(val_res["balanced_accuracy"], 4),
            "val_macro_f1": round(val_res["macro_f1"], 4),
            "val_weighted_f1": round(val_res["weighted_f1"], 4),
            "val_melanoma_recall": round(val_res["melanoma_recall"], 4),
            "lr_head": current_head_lr,
            "time_sec": round(ep_duration, 1)
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] | "
            f"Train Loss: {train_res['loss']:.4f}, Acc: {train_res['accuracy']*100:.1f}%, F1: {train_res['macro_f1']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f}, Acc: {val_res['accuracy']*100:.1f}%, Macro F1: {val_res['macro_f1']:.4f}, "
            f"Mel Recall: {val_res['melanoma_recall']*100:.1f}% | LR: {current_head_lr:.1e}"
        )

        # Save best model based on validation Macro F1
        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]
            patience_counter = 0

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_accuracy": val_res["accuracy"],
                "best_val_balanced_acc": val_res["balanced_accuracy"],
                "architecture": VIT_MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "classes": DISEASE_CLASSES,
                "input_size": [224, 224],
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                },
                "training_config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "head_lr": head_lr,
                    "backbone_lr": backbone_lr,
                    "freeze_backbone": freeze_backbone,
                    "unfreeze_blocks": unfreeze_blocks
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"  ★ New best ViT checkpoint saved to {best_checkpoint_path} (Val Macro F1: {best_val_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience = {patience}).")
                break

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Macro F1: {best_val_macro_f1:.4f}")

    # 5. Final Evaluation on Held-Out Test Set
    if best_checkpoint_path.exists():
        print("\nLoading best checkpoint for held-out test evaluation...")
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_metrics, all_preds, all_targets, all_probs = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches
        )
        test_metrics["best_val_macro_f1"] = best_val_macro_f1
        test_metrics["total_training_time_sec"] = round(total_time, 1)

        print("\n" + "=" * 55)
        print("FINAL HELD-OUT TEST EVALUATION METRICS (ViT-B/16)")
        print("=" * 55)
        print(f"Test Loss:             {test_metrics['loss']:.4f}")
        print(f"Overall Accuracy:      {test_metrics['accuracy']*100:.2f}%")
        print(f"Balanced Accuracy:     {test_metrics['balanced_accuracy']*100:.2f}%")
        print(f"Macro Precision:       {test_metrics['macro_precision']:.4f}")
        print(f"Macro Recall:          {test_metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score:        {test_metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score:     {test_metrics['weighted_f1']:.4f}")
        print(f"Multiclass ROC-AUC:    {test_metrics['roc_auc_ovr']:.4f}")
        print(f"Melanoma Recall:       {test_metrics['melanoma_recall']*100:.2f}%")
        print("=" * 55)

        generate_evaluation_artifacts(
            history=history,
            test_metrics=test_metrics,
            all_preds=all_preds,
            all_targets=all_targets,
            all_probs=all_probs,
            test_loader=test_loader,
            results_dir=results_dir,
            data_root=data_root
        )

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer (ViT-B/16) for Skin Lesion Classification")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--backbone-lr", type=float, default=DEFAULT_BACKBONE_LR)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 mini-epoch smoke test")

    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    resume_path = Path(args.resume) if args.resume else None

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        patience=args.patience,
        data_root=data_root,
        output_dir=output_dir,
        resume_path=resume_path,
        freeze_backbone=args.freeze_backbone,
        unfreeze_blocks=args.unfreeze_blocks,
        smoke_test=args.smoke_test
    )


if __name__ == "__main__":
    main()
