import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import RetinaDataset
from src.data.transforms import val_test_transform
from src.models.efficientnet import create_model


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
    fig.colorbar(cax)

    classes = ["No DR (0)", "DR (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            label = cell_labels[i][j]
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{label}\n{count}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax.set_title("Centralized EfficientNet-B0 - Confusion Matrix", fontsize=12, pad=15)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(labels, probs, roc_auc, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("Centralized EfficientNet-B0 - ROC Curve", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate(check_only=False):
    # Paths
    test_csv_path = PROJECT_ROOT / "data" / "processed" / "test_split.csv"
    images_dir = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"
    model_path = PROJECT_ROOT / "saved_models" / "centralized_best.pth"

    metrics_save_path = PROJECT_ROOT / "results" / "metrics" / "centralized_test_metrics.json"
    cm_plot_path = PROJECT_ROOT / "results" / "plots" / "centralized_confusion_matrix.png"
    roc_plot_path = PROJECT_ROOT / "results" / "plots" / "centralized_roc_curve.png"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = create_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_df = pd.read_csv(test_csv_path)
    test_dataset = RetinaDataset(test_df, images_dir, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Verification mode: test single forward pass and exit without full test evaluation
    if check_only:
        sample_images, sample_labels = next(iter(test_loader))
        with torch.no_grad():
            sample_outputs = model(sample_images.to(device))
        assert sample_outputs.shape == torch.Size([16, 2])
        print(f"[PASS] Model checkpoint loaded from: {model_path.as_posix()}")
        print(f"[PASS] Best checkpoint epoch: {checkpoint.get('epoch')}")
        print(f"[PASS] Test samples: {len(test_dataset)}")
        print(f"[PASS] Verification forward pass output shape: {list(sample_outputs.shape)}")
        print("[PASS] Evaluation script is verified and ready for execution.")
        return

    # Evaluation loop
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating Test Set", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            # Softmax to get predicted probability for class 1 (DR)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Metrics calculation
    test_loss = total_loss / len(test_dataset)
    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    sensitivity = recall_score(all_labels, all_preds, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)

    # Save metrics JSON
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        "number_of_test_images": int(len(test_dataset)),
        "test_loss": round(float(test_loss), 4),
        "accuracy": round(float(accuracy), 2),
        "precision": round(float(precision), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    with open(metrics_save_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Save plots
    plot_confusion_matrix(cm, cm_plot_path)
    plot_roc_curve(all_labels, all_probs, roc_auc, roc_plot_path)

    # Terminal summary
    print("\n" + "=" * 40)
    print("CENTRALIZED TEST RESULTS")
    print("=" * 40)
    print(f"Test Images: {len(test_dataset)}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("")
    print("Confusion Matrix:")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print("")
    print(f"Metrics Saved: {metrics_save_path.as_posix()}")
    print(f"Confusion Matrix Saved: {cm_plot_path.as_posix()}")
    print(f"ROC Curve Saved: {roc_plot_path.as_posix()}")
    print("=" * 40)


if __name__ == "__main__":
    is_check = "--check" in sys.argv
    evaluate(check_only=is_check)
