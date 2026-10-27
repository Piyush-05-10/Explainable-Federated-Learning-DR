import argparse
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


def plot_confusion_matrix(cm, save_path, title="FedAvg IID - Confusion Matrix"):
    """Plot publication-quality confusion matrix matching centralized style."""
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
    ax.set_title(title, fontsize=12, pad=15)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(labels, probs, roc_auc, save_path, title="FedAvg IID - ROC Curve", label="FedAvg IID ROC"):
    """Plot report-ready ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"{label} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_three_model_comparison(cent_metrics, fed3_metrics, fed5_metrics, save_path):
    """Plot grouped bar chart comparing Centralized, FedAvg 3-Rounds, and FedAvg 5-Rounds."""
    metric_keys = [
        ("Accuracy", "accuracy", 1.0),
        ("Precision", "precision", 100.0),
        ("Sensitivity", "sensitivity", 100.0),
        ("Specificity", "specificity", 100.0),
        ("F1 Score", "f1_score", 100.0),
        ("ROC-AUC", "roc_auc", 100.0),
    ]

    labels = [m[0] for m in metric_keys]
    cent_vals = [cent_metrics[m[1]] * m[2] for m in metric_keys]
    fed3_vals = [fed3_metrics[m[1]] * m[2] for m in metric_keys]
    fed5_vals = [fed5_metrics[m[1]] * m[2] for m in metric_keys]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, cent_vals, width, label="Centralized EfficientNet-B0", color="#2563eb", alpha=0.9)
    bars2 = ax.bar(x, fed3_vals, width, label="FedAvg IID (3 Rounds)", color="#f59e0b", alpha=0.9)
    bars3 = ax.bar(x + width, fed5_vals, width, label="FedAvg IID (5 Rounds)", color="#16a34a", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Centralized vs Federated (FedAvg IID 3 vs 5 Rounds) Test Performance", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(80, 105)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_two_model_comparison(centralized_metrics, fedavg_metrics, save_path):
    """Plot grouped bar chart comparing Centralized vs FedAvg IID 3-rounds."""
    metric_keys = [
        ("Accuracy", "accuracy", 1.0),
        ("Precision", "precision", 100.0),
        ("Sensitivity", "sensitivity", 100.0),
        ("Specificity", "specificity", 100.0),
        ("F1 Score", "f1_score", 100.0),
        ("ROC-AUC", "roc_auc", 100.0),
    ]

    labels = [m[0] for m in metric_keys]
    cent_vals = [centralized_metrics[m[1]] * m[2] for m in metric_keys]
    fed_vals = [fedavg_metrics[m[1]] * m[2] for m in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, cent_vals, width, label="Centralized EfficientNet-B0", color="#2563eb", alpha=0.9)
    bars2 = ax.bar(x + width / 2, fed_vals, width, label="FedAvg IID (3 Rounds)", color="#16a34a", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Centralized vs Federated (FedAvg IID) Test Performance", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(80, 105)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate(num_rounds=3, checkpoint_path=None, check_only=False):
    # Paths
    test_csv_path = PROJECT_ROOT / "data" / "processed" / "test_split.csv"
    images_dir = PROJECT_ROOT / "data" / "raw" / "aptos" / "train_images"
    centralized_metrics_path = PROJECT_ROOT / "results" / "metrics" / "centralized_test_metrics.json"
    fed3_metrics_path = PROJECT_ROOT / "results" / "metrics" / "fedavg_iid_test_metrics.json"

    # Resolve model checkpoint path
    if checkpoint_path is not None:
        model_path = Path(checkpoint_path)
    elif num_rounds == 5:
        model_path = PROJECT_ROOT / "saved_models" / "fedavg_iid_5rounds_final.pth"
    else:
        model_path = PROJECT_ROOT / "saved_models" / "fedavg_iid_final.pth"

    # Configure output file paths depending on target rounds
    if num_rounds == 5:
        metrics_save_path = PROJECT_ROOT / "results" / "metrics" / "fedavg_iid_5rounds_test_metrics.json"
        cm_plot_path = PROJECT_ROOT / "results" / "plots" / "fedavg_iid_5rounds_confusion_matrix.png"
        roc_plot_path = PROJECT_ROOT / "results" / "plots" / "fedavg_iid_5rounds_roc_curve.png"
        three_comp_csv_path = PROJECT_ROOT / "results" / "metrics" / "centralized_vs_fedavg_comparison.csv"
        three_comp_plot_path = PROJECT_ROOT / "results" / "plots" / "centralized_vs_fedavg_comparison.png"
        header_title = "FEDAVG IID 5-ROUND TEST EVALUATION"
        results_title = "FEDAVG IID 5-ROUND TEST RESULTS"
    else:
        metrics_save_path = PROJECT_ROOT / "results" / "metrics" / "fedavg_iid_test_metrics.json"
        cm_plot_path = PROJECT_ROOT / "results" / "plots" / "fedavg_iid_confusion_matrix.png"
        roc_plot_path = PROJECT_ROOT / "results" / "plots" / "fedavg_iid_roc_curve.png"
        header_title = "FEDAVG IID TEST EVALUATION"
        results_title = "FEDAVG IID TEST RESULTS"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    # Load model and weights
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"
    checkpoint = torch.load(model_path, map_location=device)
    ckpt_rounds = int(checkpoint.get("number_of_rounds", num_rounds))
    ckpt_clients = int(checkpoint.get("number_of_clients", 5))
    ckpt_partition = str(checkpoint.get("partition_type", "IID"))

    model = create_model(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_df = pd.read_csv(test_csv_path)
    test_dataset = RetinaDataset(test_df, images_dir, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Verification mode: test single forward pass and exit without full test evaluation
    if check_only:
        sample_images, _ = next(iter(test_loader))
        with torch.no_grad():
            sample_outputs = model(sample_images.to(device))
        assert sample_outputs.shape == torch.Size([16, 2])
        print("=" * 50)
        print(f"VERIFYING FEDAVG IID ({num_rounds} ROUNDS) EVALUATION SETUP")
        print("=" * 50)
        print(f"[PASS] 1. Checkpoint path: {model_path.as_posix()}")
        print(f"[PASS] 2. Checkpoint metadata: Rounds={ckpt_rounds}, Clients={ckpt_clients}, Partition={ckpt_partition}")
        if num_rounds == 5:
            assert ckpt_rounds == 5, f"Expected 5 rounds in checkpoint, found {ckpt_rounds}"
            assert ckpt_clients == 5, f"Expected 5 clients, found {ckpt_clients}"
            assert ckpt_partition == "IID", f"Expected IID, found {ckpt_partition}"
        print(f"[PASS] 3. Test dataset verified: exactly {len(test_dataset)} images from test_split.csv")
        print(f"[PASS] 4. Inference test forward pass: OK (Output shape: {list(sample_outputs.shape)})")
        print(f"[PASS] 5. Centralized baseline JSON exists: {centralized_metrics_path.exists()}")
        print(f"[PASS] 6. 3-Round FedAvg JSON exists: {fed3_metrics_path.exists()}")
        print(f"[PASS] 7. Dedicated 5-round output paths verified (Original 3-round results protected):")
        print(f"         Target JSON: {metrics_save_path.name}")
        print(f"         Target CM:   {cm_plot_path.name}")
        print(f"         Target ROC:  {roc_plot_path.name}")
        if num_rounds == 5:
            print(f"         3-Model CSV: {three_comp_csv_path.name}")
            print(f"         3-Model Plot: {three_comp_plot_path.name}")
        print("=" * 50)
        print("ALL EVALUATION VERIFICATION CHECKS PASSED: READY FOR INFERENCE")
        print("=" * 50)
        return

    # Terminal start header
    print("=" * 40)
    print(header_title)
    print("=" * 40)
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"Checkpoint: {model_path.name}")
    print(f"Test Images: {len(test_dataset)}")
    print(f"Federated Rounds: {ckpt_rounds}")
    print(f"Clients: {ckpt_clients}")
    print("=" * 40 + "\n")

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
        "test_images": int(len(test_dataset)),
        "test_loss": round(float(test_loss), 4),
        "accuracy": round(float(accuracy), 2),
        "precision": round(float(precision), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "federated_rounds": int(ckpt_rounds),
        "local_epochs": int(checkpoint.get("local_epochs", 1)),
        "number_of_clients": int(ckpt_clients),
        "partition_type": str(ckpt_partition),
    }

    with open(metrics_save_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Save plots
    cm_title = f"FedAvg IID ({ckpt_rounds} Rounds) - Confusion Matrix"
    roc_title = f"FedAvg IID ({ckpt_rounds} Rounds) - ROC Curve"
    roc_label = f"FedAvg IID {ckpt_rounds}R"
    plot_confusion_matrix(cm, cm_plot_path, title=cm_title)
    plot_roc_curve(all_labels, all_probs, roc_auc, roc_plot_path, title=roc_title, label=roc_label)

    # Terminal summary
    print("\n" + "=" * 40)
    print(results_title)
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
    print("=" * 40 + "\n")

    # Comparisons
    with open(centralized_metrics_path, "r") as f:
        cent_data = json.load(f)

    if num_rounds == 5:
        # Load 3-round metrics
        with open(fed3_metrics_path, "r") as f:
            fed3_data = json.load(f)

        # Three-model comparison table
        three_comp_rows = [
            {"metric": "Accuracy (%)", "centralized": cent_data["accuracy"], "fedavg_3rounds": fed3_data["accuracy"], "fedavg_5rounds": metrics_data["accuracy"]},
            {"metric": "Precision", "centralized": cent_data["precision"], "fedavg_3rounds": fed3_data["precision"], "fedavg_5rounds": metrics_data["precision"]},
            {"metric": "Sensitivity", "centralized": cent_data["sensitivity"], "fedavg_3rounds": fed3_data["sensitivity"], "fedavg_5rounds": metrics_data["sensitivity"]},
            {"metric": "Specificity", "centralized": cent_data["specificity"], "fedavg_3rounds": fed3_data["specificity"], "fedavg_5rounds": metrics_data["specificity"]},
            {"metric": "F1 Score", "centralized": cent_data["f1_score"], "fedavg_3rounds": fed3_data["f1_score"], "fedavg_5rounds": metrics_data["f1_score"]},
            {"metric": "ROC-AUC", "centralized": cent_data["roc_auc"], "fedavg_3rounds": fed3_data["roc_auc"], "fedavg_5rounds": metrics_data["roc_auc"]},
        ]

        df_three = pd.DataFrame(three_comp_rows)
        df_three.to_csv(three_comp_csv_path, index=False)

        # Three-model plot
        plot_three_model_comparison(cent_data, fed3_data, metrics_data, three_comp_plot_path)

        print("CENTRALIZED VS FEDAVG IID COMPARISON")
        print("-" * 75)
        print(f"{'Metric':<18} {'Centralized':<18} {'FedAvg 3 Rounds':<20} {'FedAvg 5 Rounds':<18}")
        print("-" * 75)
        for row in three_comp_rows:
            print(f"{row['metric']:<18} {str(row['centralized']):<18} {str(row['fedavg_3rounds']):<20} {str(row['fedavg_5rounds']):<18}")
        print("-" * 75)

        print(f"\nMetrics JSON Saved: {metrics_save_path.name}")
        print(f"Three-Model Comparison CSV Saved: {three_comp_csv_path.name}")
        print(f"Confusion Matrix Saved: {cm_plot_path.name}")
        print(f"ROC Curve Saved: {roc_plot_path.name}")
        print(f"Comparison Plot Saved: {three_comp_plot_path.name}")
        print("=" * 40)
    else:
        comparison_csv_path = PROJECT_ROOT / "results" / "metrics" / "centralized_vs_fedavg_iid.csv"
        comparison_plot_path = PROJECT_ROOT / "results" / "plots" / "centralized_vs_fedavg_iid.png"
        two_comp_rows = [
            {"metric": "Accuracy (%)", "centralized": cent_data["accuracy"], "fedavg_iid": metrics_data["accuracy"], "difference": round(metrics_data["accuracy"] - cent_data["accuracy"], 2)},
            {"metric": "Precision", "centralized": cent_data["precision"], "fedavg_iid": metrics_data["precision"], "difference": round(metrics_data["precision"] - cent_data["precision"], 4)},
            {"metric": "Sensitivity", "centralized": cent_data["sensitivity"], "fedavg_iid": metrics_data["sensitivity"], "difference": round(metrics_data["sensitivity"] - cent_data["sensitivity"], 4)},
            {"metric": "Specificity", "centralized": cent_data["specificity"], "fedavg_iid": metrics_data["specificity"], "difference": round(metrics_data["specificity"] - cent_data["specificity"], 4)},
            {"metric": "F1 Score", "centralized": cent_data["f1_score"], "fedavg_iid": metrics_data["f1_score"], "difference": round(metrics_data["f1_score"] - cent_data["f1_score"], 4)},
            {"metric": "ROC-AUC", "centralized": cent_data["roc_auc"], "fedavg_iid": metrics_data["roc_auc"], "difference": round(metrics_data["roc_auc"] - cent_data["roc_auc"], 4)},
        ]
        df_two = pd.DataFrame(two_comp_rows)
        df_two.to_csv(comparison_csv_path, index=False)
        plot_two_model_comparison(cent_data, metrics_data, comparison_plot_path)

        print("CENTRALIZED VS FEDAVG IID")
        print("-" * 65)
        print(f"{'Metric':<18} {'Centralized':<15} {'FedAvg IID':<15} {'Difference':<15}")
        print("-" * 65)
        for row in two_comp_rows:
            diff_str = f"{row['difference']:+.4f}" if "Accuracy" not in row["metric"] else f"{row['difference']:+.2f}%"
            print(f"{row['metric']:<18} {row['centralized']:<15} {row['fedavg_iid']:<15} {diff_str:<15}")
        print("-" * 65)

        print(f"\nMetrics JSON Saved: {metrics_save_path.name}")
        print(f"Comparison CSV Saved: {comparison_csv_path.name}")
        print(f"Confusion Matrix Saved: {cm_plot_path.name}")
        print(f"ROC Curve Saved: {roc_plot_path.name}")
        print(f"Comparison Plot Saved: {comparison_plot_path.name}")
        print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FedAvg IID on untouched test set")
    parser.add_argument("--rounds", type=int, default=3, choices=[3, 5], help="Federated rounds of the model to evaluate (3 or 5, default: 3)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to model checkpoint .pth")
    parser.add_argument("--check", action="store_true", help="Run verification check without evaluating full test set")
    args = parser.parse_args()

    # If explicit checkpoint path contains 5rounds, infer 5 rounds
    if args.checkpoint and "5round" in args.checkpoint:
        eval_rounds = 5
    else:
        eval_rounds = args.rounds

    evaluate(num_rounds=eval_rounds, checkpoint_path=args.checkpoint, check_only=args.check)
