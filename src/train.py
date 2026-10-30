"""
Training pipeline for Plant Disease Classification.
Features:
  - Custom CNN and Transfer Learning backbones
  - Early Stopping with best-weight restoration
  - Model Checkpointing (saves best and latest checkpoints)
  - Learning Rate Scheduling (ReduceLROnPlateau)
  - Training and validation loss/accuracy curve plotting
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from src.data_loader import get_data_loaders
from src.model import build_model


class EarlyStopping:
    """
    Early stopping monitor. Halts training when validation loss stops improving.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            return True  # Improved
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Did not improve
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improved


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_training_history(history: dict, save_path: Path, model_name: str):
    """Plot and save training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.set_title(f"{model_name.upper()} - Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Accuracy curve
    ax1_acc = ax2
    ax1_acc.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="#2ca02c", linewidth=2)
    ax1_acc.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc (%)", color="#d62728", linewidth=2, linestyle="--")
    ax1_acc.set_title(f"{model_name.upper()} - Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax1_acc.set_xlabel("Epoch", fontsize=11)
    ax1_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax1_acc.grid(True, linestyle=":", alpha=0.6)
    ax1_acc.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved learning curves plot to {save_path}")


def train_model(
    model_name: str = "custom_cnn",
    data_dir: str = "./data",
    output_dir: str = "./models",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 4,
    freeze_backbone: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} | Model: {model_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )
    num_classes = len(class_names)
    print(f"[*] Loaded dataset with {num_classes} classes: {class_names}")

    # Build model
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=freeze_backbone,
    ).to(device)

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    early_stopping = EarlyStopping(patience=patience, mode="min")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_path = output_path / f"{model_name}_best.pth"
    start_time = time.time()

    print("\n" + "=" * 65)
    print(f"Beginning Training: {model_name.upper()} ({epochs} Epochs)")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        improved = early_stopping(val_loss)

        status_flag = "(*)" if improved else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s {status_flag}"
        )

        # Model Checkpoint: Save best model weights
        if improved or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "model_name": model_name,
                    "num_classes": num_classes,
                },
                best_model_path,
            )

        if early_stopping.early_stop:
            print(f"\n[!] Early stopping triggered after {epoch} epochs without improvement.")
            break

    total_time = time.time() - start_time
    print(f"\n[+] Training completed in {total_time/60:.2f} mins.")
    print(f"[+] Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc*100:.2f}%")
    print(f"[+] Best model checkpoint saved to: {best_model_path}")

    # Save history and curves
    history_path = output_path / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_path = output_path / f"training_curves_{model_name}.png"
    plot_training_history(history, plot_path, model_name)

    return history, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection Model")
    parser.add_argument("--model", type=str, default="custom_cnn", choices=["custom_cnn", "mobilenet_v2", "resnet50"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
    )


if __name__ == "__main__":
    main()
