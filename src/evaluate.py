"""
Comprehensive Evaluation and Model Comparison Module.
Computes:
  - Accuracy, Precision, Recall, F1-Score (macro, weighted, and per-class)
  - High-resolution Confusion Matrix heatmap
  - Multi-model comparative analysis (Custom CNN vs MobileNetV2 / ResNet50)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from src.data_loader import get_data_loaders
from src.model import build_model


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model weights and metadata from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "custom_cnn")
    num_classes = checkpoint.get("num_classes", len(checkpoint.get("class_names", [])))
    class_names = checkpoint.get("class_names", [])

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, model_name


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(
    checkpoint_path: str or Path,
    data_dir: str = "./data",
    output_dir: str = "./models",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Run evaluation on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file}")

    model, class_names, model_name = load_checkpoint(checkpoint_file, device)
    _, _, test_loader, dataset_classes = get_data_loaders(data_dir=data_dir, batch_size=batch_size)

    if not class_names:
        class_names = dataset_classes

    total_params = count_parameters(model)
    print(f"[*] Evaluating {model_name} on device: {device} | Parameters: {total_params:,}")

    all_preds: List[int] = []
    all_targets: List[int] = []
    latencies: List[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.time()
            outputs = model(images)
            latencies.append((time.time() - t0) / images.size(0) * 1000.0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    avg_latency_ms = float(np.mean(latencies))

    # Detailed report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: {model_name.upper()}")
    print("=" * 65)
    print(f"Test Accuracy:         {acc * 100:.2f}%")
    print(f"Macro Precision:       {prec_macro * 100:.2f}%")
    print(f"Macro Recall:          {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:        {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:     {f1_weighted * 100:.2f}%")
    print(f"Mean Latency (ms):     {avg_latency_ms:.2f} ms / sample")
    print("\nClassification Report:\n", report_text)

    # Plot Confusion Matrix
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.replace("___", "\n") for c in class_names],
        yticklabels=[c.replace("___", "\n") for c in class_names],
        cbar=True,
    )
    plt.title(f"Confusion Matrix - {model_name.upper()}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_save_path = output_path / f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix to {cm_save_path}")

    metrics = {
        "model_name": model_name,
        "parameters": total_params,
        "test_accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "latency_ms_per_sample": avg_latency_ms,
        "classification_report": report_dict,
    }

    metrics_save_path = output_path / f"metrics_{model_name}.json"
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def compare_models(
    model_paths: List[Path],
    data_dir: str = "./data",
    output_dir: str = "./models",
):
    """
    Compare multiple trained models and plot a side-by-side metric comparison.
    """
    results = []
    for p in model_paths:
        if p.exists():
            metrics = evaluate_model(p, data_dir=data_dir, output_dir=output_dir)
            results.append(metrics)

    if len(results) < 2:
        print("[!] At least 2 models are needed for comparison.")
        return results

    # Generate comparison bar chart
    model_labels = [r["model_name"].replace("_", " ").title() for r in results]
    accs = [r["test_accuracy"] * 100 for r in results]
    f1s = [r["macro_f1"] * 100 for r in results]
    latencies = [r["latency_ms_per_sample"] for r in results]

    x = np.arange(len(model_labels))
    width = 0.30

    fig, ax1 = plt.subplots(figsize=(9, 5))
    rects1 = ax1.bar(x - width / 2, accs, width, label="Test Accuracy (%)", color="#2b5c8f")
    rects2 = ax1.bar(x + width / 2, f1s, width, label="Macro F1-Score (%)", color="#38a169")

    ax1.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Comparison: Custom CNN vs Transfer Learning", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    comp_plot_path = Path(output_dir) / "model_comparison_chart.png"
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved model comparison chart to {comp_plot_path}")

    # Save summary table
    summary_path = Path(output_dir) / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Models")
    parser.add_argument("--checkpoint", type=str, default="./models/custom_cnn_best.pth")
    parser.add_argument("--compare", nargs="+", default=None, help="Paths to two or more checkpoints to compare")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./models")
    args = parser.parse_args()

    if args.compare:
        compare_models([Path(p) for p in args.compare], data_dir=args.data_dir, output_dir=args.output_dir)
    else:
        evaluate_model(args.checkpoint, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
