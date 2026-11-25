"""
Training and Evaluation Pipeline for ResNet18-UNet Lesion Segmentation.
Phase 3 Implementation.

Configuration:
- Max epochs: 20
- Batch size: 8
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Loss: Combined BCEWithLogitsLoss + DiceLoss
- Metric: Dice Score (F1), IoU (Jaccard), Pixel Precision, Pixel Recall
- Early Stopping: Patience = 5 epochs on validation Dice score
- Learning Rate Scheduler: ReduceLROnPlateau on validation Dice
- Mixed Precision: torch.cuda.amp enabled when CUDA is available
- Checkpoint: Saves best model to models/unet_model.pth with architecture metadata
- Results: CSV history, metrics JSON, evaluation curves, and held-out test visual comparisons
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
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.unet import build_unet, ResNet18UNet
from training.dataset import get_segmentation_dataloaders, get_default_data_root

# Default hyperparameters
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
RANDOM_SEED = 42


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation:
    Dice = 2 * |P * G| / (|P| + |G| + eps)
    Loss = 1 - Dice
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined loss: BCEWithLogitsLoss + DiceLoss
    Provides stable gradients from cross entropy and direct overlap optimization from Dice.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total = (self.bce_weight * bce) + (self.dice_weight * dice)
        return total, bce, dice


def compute_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Computes binary segmentation evaluation metrics on a batch:
    - Dice Score (F1)
    - IoU (Jaccard Index)
    - Pixel Precision
    - Pixel Recall
    """
    preds = (probs >= threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall)
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """Runs one training epoch."""
    model.train()
    running_loss, running_bce, running_dice = 0.0, 0.0, 0.0
    running_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    count = 0

    use_amp = scaler is not None and device.type == "cuda"

    for i, (images, masks, _) in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            autocast_ctx = torch.amp.autocast("cuda") if hasattr(torch, "amp") else torch.cuda.amp.autocast()
            with autocast_ctx:
                logits = model(images)
                total_loss, bce, dice = criterion(logits, masks)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            total_loss, bce, dice = criterion(logits, masks)
            total_loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        batch_m = compute_metrics(probs, masks, threshold=0.5)

        running_loss += total_loss.item()
        running_bce += bce.item()
        running_dice += dice.item()
        for k in running_metrics:
            running_metrics[k] += batch_m[k]
        count += 1

    if count == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "loss": running_loss / count,
        "bce": running_bce / count,
        "dice_loss": running_dice / count,
        "dice": running_metrics["dice"] / count,
        "iou": running_metrics["iou"] / count,
        "precision": running_metrics["precision"] / count,
        "recall": running_metrics["recall"] / count
    }


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """Runs evaluation on validation or test sets."""
    model.eval()
    running_loss = 0.0
    running_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    count = 0

    with torch.no_grad():
        for i, (images, masks, _) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            total_loss, _, _ = criterion(logits, masks)
            probs = torch.sigmoid(logits)

            batch_m = compute_metrics(probs, masks, threshold=threshold)

            running_loss += total_loss.item()
            for k in running_metrics:
                running_metrics[k] += batch_m[k]
            count += 1

    if count == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "loss": running_loss / count,
        "dice": running_metrics["dice"] / count,
        "iou": running_metrics["iou"] / count,
        "precision": running_metrics["precision"] / count,
        "recall": running_metrics["recall"] / count
    }


def find_best_threshold(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> float:
    """Finds the optimal decision threshold that maximizes Dice score on the validation set only."""
    best_th = 0.5
    best_dice = -1.0
    criterion = CombinedBCEDiceLoss()

    for th in thresholds:
        m = evaluate(model, val_loader, criterion, device, threshold=th)
        if m["dice"] > best_dice:
            best_dice = m["dice"]
            best_th = th

    print(f"Optimal validation threshold: {best_th:.2f} (Val Dice: {best_dice:.4f})")
    return best_th


def generate_evaluation_artifacts(
    history: List[Dict[str, Any]],
    test_metrics: Dict[str, float],
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
    results_dir: Path,
    data_root: Path
):
    """Generates loss/metric plots and genuine test sample visual comparisons."""
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    samples_dir = results_dir / "samples"
    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training history CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "unet_training_history.csv", index=False)

    # 2. Metrics JSON
    with open(results_dir / "unet_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # 3. Plots
    if len(history_df) > 1:
        epochs = history_df["epoch"]

        # Loss Plot
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["train_loss"], label="Train Loss", color="#2d6a4f", lw=2)
        plt.plot(epochs, history_df["val_loss"], label="Val Loss", color="#d90429", lw=2, linestyle="--")
        plt.title("U-Net Loss: Train vs Validation", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (BCE + Dice)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curve.png", dpi=150)
        plt.close()

        # Dice Score Plot
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["val_dice"], label="Validation Dice Score", color="#1b4332", lw=2)
        plt.title("Validation Dice Score across Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score (F1)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "dice_curve.png", dpi=150)
        plt.close()

        # IoU Plot
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, history_df["val_iou"], label="Validation IoU (Jaccard)", color="#40916c", lw=2)
        plt.title("Validation IoU across Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "iou_curve.png", dpi=150)
        plt.close()

    # 4. Genuine test sample comparisons (6 diverse samples)
    model.eval()
    samples_saved = 0
    with torch.no_grad():
        for images, masks, image_ids in test_loader:
            images_d = images.to(device)
            probs, binary_masks = model.predict_mask(images_d, threshold=threshold)

            images_np = images.cpu().numpy()
            gt_masks_np = masks.cpu().numpy()
            pred_masks_np = binary_masks.cpu().numpy()

            for i in range(len(images)):
                if samples_saved >= 6:
                    break

                img_id = image_ids[i]
                # Denormalize image for display
                img_display = images_np[i].transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_display = (img_display * std + mean) * 255.0
                img_display = np.clip(img_display, 0, 255).astype(np.uint8)

                gt_mask = (gt_masks_np[i, 0] * 255).astype(np.uint8)
                pred_mask = (pred_masks_np[i, 0] * 255).astype(np.uint8)

                # Generate transparent green lesion overlay
                overlay = img_display.copy()
                green_tint = np.zeros_like(img_display)
                green_tint[:, :, 1] = 255  # Green channel
                mask_bool = pred_mask > 127
                overlay[mask_bool] = cv2.addWeighted(img_display[mask_bool], 0.6, green_tint[mask_bool], 0.4, 0)

                # Draw contour around prediction
                contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

                # Create 1x4 comparison figure
                fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
                axes[0].imshow(img_display)
                axes[0].set_title(f"Original: {img_id}", fontsize=10)
                axes[0].axis("off")

                axes[1].imshow(gt_mask, cmap="gray")
                axes[1].set_title("Ground Truth Mask", fontsize=10)
                axes[1].axis("off")

                axes[2].imshow(pred_mask, cmap="gray")
                axes[2].set_title(f"Predicted Mask (th={threshold:.2f})", fontsize=10)
                axes[2].axis("off")

                axes[3].imshow(overlay)
                axes[3].set_title("Lesion Overlay", fontsize=10)
                axes[3].axis("off")

                plt.tight_layout()
                fig.savefig(samples_dir / f"test_sample_{samples_saved + 1}_{img_id}.png", dpi=150)
                plt.close(fig)
                samples_saved += 1

            if samples_saved >= 6:
                break

    print(f"✔ Generated {samples_saved} genuine test visual comparisons in {samples_dir}")


def run_training(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    resume_path: Optional[Path] = None,
    smoke_test: bool = False
):
    """Main execution function for U-Net training and evaluation."""
    set_seed(RANDOM_SEED)

    if data_root is None:
        data_root = get_default_data_root()
    if output_dir is None:
        output_dir = BASE_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = BASE_DIR / "training" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")
    if device.type == "cuda":
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # 1. Load DataLoaders
    print("Loading segmentation DataLoaders...")
    dataloaders = get_segmentation_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,  # 0 for safe multiprocessing on Windows
        apply_hair_removal=False
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    print(f"Train samples: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 2. Build model
    print("Instantiating ResNet18-UNet model...")
    model = build_unet(pretrained=True, device=device)

    # 3. Criterion, Optimizer, Scheduler, Scaler
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    use_amp = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_dice = -1.0
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
        best_val_dice = checkpoint.get("best_val_dice", -1.0)
        print(f"Resumed at epoch {start_epoch} (Best Val Dice: {best_val_dice:.4f})")

    if smoke_test:
        print("\n--- RUNNING SMOKE TEST (1 mini-epoch, 2 batches) ---")
        epochs = 1
        max_batches = 2
    else:
        max_batches = None

    best_checkpoint_path = output_dir / "unet_model.pth"

    print("\nStarting training loop...")
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
        val_res = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=0.5,
            max_batches=max_batches
        )
        ep_duration = time.time() - ep_start

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_res["dice"])

        record = {
            "epoch": epoch,
            "train_loss": round(train_res["loss"], 4),
            "train_dice": round(train_res["dice"], 4),
            "train_iou": round(train_res["iou"], 4),
            "val_loss": round(val_res["loss"], 4),
            "val_dice": round(val_res["dice"], 4),
            "val_iou": round(val_res["iou"], 4),
            "val_precision": round(val_res["precision"], 4),
            "val_recall": round(val_res["recall"], 4),
            "lr": current_lr,
            "time_sec": round(ep_duration, 1)
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] | "
            f"Train Loss: {train_res['loss']:.4f}, Dice: {train_res['dice']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f}, Dice: {val_res['dice']:.4f}, IoU: {val_res['iou']:.4f} | "
            f"LR: {current_lr:.1e}"
        )

        # Save best model based on validation Dice Score
        if val_res["dice"] > best_val_dice:
            best_val_dice = val_res["dice"]
            patience_counter = 0

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dice": best_val_dice,
                "best_val_iou": val_res["iou"],
                "architecture": "ResNet18UNet",
                "input_size": [256, 256],
                "preprocessing": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                "default_threshold": 0.5,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"  ★ New best model checkpoint saved to {best_checkpoint_path} (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience = {patience}).")
                break

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Dice: {best_val_dice:.4f}")

    # 4. Final Evaluation on Held-Out Test Set
    if best_checkpoint_path.exists():
        print("\nLoading best model checkpoint for evaluation on held-out test dataset...")
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        # Select optimal threshold on validation set
        optimal_th = find_best_threshold(model, val_loader, device)

        # Update checkpoint with optimal threshold
        best_ckpt["optimal_threshold"] = optimal_th
        torch.save(best_ckpt, best_checkpoint_path)

        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, criterion, device, threshold=optimal_th)
        test_metrics["optimal_threshold"] = optimal_th
        test_metrics["best_val_dice"] = best_val_dice
        test_metrics["total_training_time_sec"] = round(total_time, 1)

        print("\n--- FINAL HELD-OUT TEST EVALUATION METRICS ---")
        print(f"Test Loss:            {test_metrics['loss']:.4f}")
        print(f"Test Dice Score:      {test_metrics['dice']:.4f}")
        print(f"Test IoU (Jaccard):   {test_metrics['iou']:.4f}")
        print(f"Test Pixel Precision: {test_metrics['precision']:.4f}")
        print(f"Test Pixel Recall:    {test_metrics['recall']:.4f}")
        print(f"Decision Threshold:   {optimal_th:.2f}")

        # Generate plots and sample comparisons
        generate_evaluation_artifacts(
            history=history,
            test_metrics=test_metrics,
            model=model,
            test_loader=test_loader,
            device=device,
            threshold=optimal_th,
            results_dir=results_dir,
            data_root=data_root
        )

    return history


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18-UNet for Skin Lesion Segmentation")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 mini-epoch smoke test")

    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    resume_path = Path(args.resume) if args.resume else None

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        data_root=data_root,
        output_dir=output_dir,
        resume_path=resume_path,
        smoke_test=args.smoke_test
    )


if __name__ == "__main__":
    main()
